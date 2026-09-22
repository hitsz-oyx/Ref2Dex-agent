"""Rebuild auditable inputs for the public DExplore GRAB converter.

The adapter keeps the legacy 30 Hz body/object motion, restores native-rate
GRAB contact arrays, and materializes the exact mesh/template/skeleton files
required by ``data_processing/convert_grab.py``.  Source trees are read-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def derive_body_translation(*, baseline_tensor: Path, reference_tensor: Path):
    """Infer converter-input translation from right-wrist world deltas.

    The public converter applies +90 degrees about X, mapping input ``(x,y,z)``
    to output ``(x,-z,y)``.  Inverting that map gives ``(dx,dz,-dy)``.
    """
    baseline = torch.load(baseline_tensor, map_location="cpu", weights_only=True)
    reference = torch.load(reference_tensor, map_location="cpu", weights_only=True)
    if baseline.shape != reference.shape or baseline.ndim != 2 or baseline.shape[1] != 598:
        raise ValueError("baseline and reference tensors must be matching [T,598]")
    if not torch.isfinite(baseline).all() or not torch.isfinite(reference).all():
        raise ValueError("alignment tensors must be finite")
    delta = reference[:, 51:54] - baseline[:, 51:54]
    correction = torch.stack((delta[:, 0], delta[:, 2], -delta[:, 1]), dim=-1)
    provenance = {
        "method": "right_wrist_world_delta_inverse_rotation_x90",
        "baseline_tensor": _record(Path(baseline_tensor)),
        "reference_tensor": _record(Path(reference_tensor)),
        "output_relative_delta_m": {
            "shape": list(delta.shape),
            "min": delta.amin(0).tolist(), "max": delta.amax(0).tolist(),
            "mean": delta.mean(0).tolist(),
        },
    }
    return correction.numpy(), provenance


def build(*, sequence: str, legacy_root: Path, raw_root: Path,
          raw_object_root: Path, raw_tools_root: Path, skeleton_root: Path,
          output_root: Path, body_translation=None,
          body_translation_delta=None,
          alignment_provenance: dict | None = None) -> dict:
    subject, remainder = sequence.split("_", 1)
    object_name = remainder.split("_", 1)[0]
    legacy_sequence = Path(legacy_root) / sequence
    legacy_motion = legacy_sequence / "motion.npz"
    legacy_object = legacy_sequence / "object.npz"
    raw_grab = Path(raw_root) / subject / f"{remainder}.npz"
    raw_object = Path(raw_object_root) / object_name / "mesh.obj"
    skeleton = Path(skeleton_root) / f"smplx_grab_{subject}.xml"
    for path in (legacy_motion, legacy_object, raw_grab, raw_object, skeleton):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(legacy_motion, allow_pickle=True) as source:
        payload = {key: copy.deepcopy(source[key]) for key in source.files}
    body = copy.deepcopy(payload["body"].item())
    n_frames = int(np.asarray(payload["n_frames"]).item())
    vtemp = Path(str(body["vtemp"]))
    subject_mesh = Path(raw_tools_root).parent / vtemp
    if not subject_mesh.is_file():
        raise FileNotFoundError(subject_mesh)
    with np.load(raw_grab, allow_pickle=True) as source:
        contact = copy.deepcopy(source["contact"].item())
    lengths = {np.asarray(contact[key]).shape[0] for key in ("body", "object")}
    if len(lengths) != 1 or next(iter(lengths)) <= n_frames:
        raise ValueError("raw GRAB native contact must have more frames than 30 Hz motion")
    payload["contact"] = np.asarray(contact, dtype=object)

    if body_translation is not None and body_translation_delta is not None:
        raise ValueError("body translation and delta are mutually exclusive")
    if body_translation_delta is not None:
        delta = np.asarray(body_translation_delta, dtype=np.float32)
        legacy_translation = np.asarray(body["params"]["transl"], dtype=np.float32)
        if delta.shape != (n_frames, 3) or legacy_translation.shape != (n_frames, 3):
            raise ValueError("body translation delta and legacy translation must be [T,3]")
        translation = legacy_translation + delta
        if not np.isfinite(translation).all():
            raise ValueError("corrected body translation must be finite")
        body["params"] = copy.deepcopy(body["params"])
        body["params"]["transl"] = translation
    elif body_translation is not None:
        translation = np.asarray(body_translation, dtype=np.float32)
        if translation.shape == (3,):
            translation = np.broadcast_to(translation, (n_frames, 3)).copy()
        if translation.shape != (n_frames, 3) or not np.isfinite(translation).all():
            raise ValueError("body translation must be finite [3] or [T,3]")
        body["params"] = copy.deepcopy(body["params"])
        body["params"]["transl"] = translation
    payload["body"] = np.asarray(body, dtype=object)

    sequence_root = Path(output_root) / "sequences" / sequence
    sequence_root.mkdir(parents=True, exist_ok=False)
    motion_out = sequence_root / "motion.npz"
    np.savez(motion_out, **payload)
    object_out = sequence_root / "object.npz"
    shutil.copyfile(legacy_object, object_out)
    object_mesh_out = Path(output_root) / "objects" / object_name / f"{object_name}.obj"
    object_mesh_out.parent.mkdir(parents=True, exist_ok=True)
    if not object_mesh_out.exists():
        shutil.copyfile(raw_object, object_mesh_out)
    subject_out = Path(output_root) / str(body["vtemp"])
    subject_out.parent.mkdir(parents=True, exist_ok=True)
    if not subject_out.exists():
        shutil.copyfile(subject_mesh, subject_out)
    skeleton_out = Path(output_root) / "skeletons" / skeleton.name
    skeleton_out.parent.mkdir(parents=True, exist_ok=True)
    if not skeleton_out.exists():
        shutil.copyfile(skeleton, skeleton_out)

    manifest = {
        "schema": "dexplore_reconstructed_motion_input_v1",
        "classification": "reconstructed_baseline", "sequence": sequence,
        "motion_frames_30hz": n_frames, "contact_source": "raw_grab_native_rate",
        "contact_frames": {key: int(np.asarray(contact[key]).shape[0])
                           for key in ("body", "object")},
        "body_world_translation_m": {
            "kind": "per_frame", "shape": list(body["params"]["transl"].shape),
            "min": np.asarray(body["params"]["transl"]).min(0).tolist(),
            "max": np.asarray(body["params"]["transl"]).max(0).tolist(),
            "mean": np.asarray(body["params"]["transl"]).mean(0).tolist(),
        },
        "inputs": {"legacy_motion": _record(legacy_motion), "legacy_object": _record(legacy_object),
                   "raw_grab": _record(raw_grab), "raw_object_mesh": _record(raw_object),
                   "raw_subject_mesh": _record(subject_mesh), "legacy_interact_skeleton": _record(skeleton)},
        "outputs": {"motion": _record(motion_out), "object": _record(object_out),
                    "object_mesh": _record(object_mesh_out), "subject_mesh": _record(subject_out),
                    "skeleton": _record(skeleton_out)},
        "invariants": ["legacy 30 Hz body metadata is preserved except explicit translation",
                       "legacy object is copied byte-for-byte",
                       "raw native-rate contact is preserved without resampling"],
    }
    if alignment_provenance is not None:
        manifest["body_translation_alignment"] = alignment_provenance
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--raw-object-root", type=Path, required=True)
    parser.add_argument("--raw-tools-root", type=Path, required=True)
    parser.add_argument("--skeleton-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-tensor", type=Path)
    parser.add_argument("--reference-tensor", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if (args.baseline_tensor is None) != (args.reference_tensor is None):
        raise ValueError("baseline and reference tensors must be supplied together")
    translation = provenance = None
    if args.baseline_tensor is not None:
        translation, provenance = derive_body_translation(
            baseline_tensor=args.baseline_tensor, reference_tensor=args.reference_tensor)
    manifest = build(sequence=args.sequence, legacy_root=args.legacy_root,
                     raw_root=args.raw_root, raw_object_root=args.raw_object_root,
                     raw_tools_root=args.raw_tools_root, skeleton_root=args.skeleton_root,
                     output_root=args.output_root, body_translation_delta=translation,
                     alignment_provenance=provenance)
    path = args.manifest or args.output_root / f"manifest_{args.sequence}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "sequence": args.sequence}, sort_keys=True))


if __name__ == "__main__":
    main()
