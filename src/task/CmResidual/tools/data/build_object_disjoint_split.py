"""Stage auditable DExplore motion views with disjoint train/held-out objects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_entry(entry: dict) -> dict:
    name = entry["sequence"]
    parts = name.split("_")
    if len(parts) < 3 or not parts[0].startswith("s") or not parts[0][1:].isdigit():
        raise ValueError(f"invalid GRAB sequence name: {name}")
    sequence_dir = Path(entry["sequence_dir"]).resolve()
    source_manifest = Path(entry["input_manifest"]).resolve()
    tensor = sequence_dir / "interaction_hand_inspire.pt"
    if not tensor.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(f"missing tensor or source manifest for {name}")
    if sequence_dir.name != name:
        raise ValueError(f"directory and sequence name differ: {sequence_dir}, {name}")
    provenance = json.loads(source_manifest.read_text(encoding="utf-8"))
    if provenance.get("classification") != "reconstructed_baseline" or provenance.get("sequence") != name:
        raise ValueError(f"unexpected source provenance for {name}")
    return {"sequence": name, "object": parts[1],
            "sequence_dir": str(sequence_dir), "tensor_sha256": _sha256(tensor),
            "input_manifest": str(source_manifest),
            "input_manifest_sha256": _sha256(source_manifest)}


def build(*, spec_path: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    train = [_validate_entry(entry) for entry in spec["train"]]
    heldout = [_validate_entry(entry) for entry in spec["heldout"]]
    if not train or not heldout:
        raise ValueError("train and heldout each need at least one sequence")
    names = [entry["sequence"] for entry in train + heldout]
    if len(names) != len(set(names)):
        raise ValueError("duplicate GRAB sequence")
    train_objects = {entry["object"] for entry in train}
    heldout_objects = {entry["object"] for entry in heldout}
    if train_objects & heldout_objects:
        raise ValueError(f"object leakage: {sorted(train_objects & heldout_objects)}")
    output_root.mkdir(parents=True)
    for partition, entries in (("train", train), ("heldout", heldout)):
        destination = output_root / partition
        destination.mkdir()
        for entry in entries:
            (destination / entry["sequence"]).symlink_to(
                entry["sequence_dir"], target_is_directory=True)
    manifest = {
        "schema": "ref2dex.object_disjoint_motion_split.v1",
        "classification": "reconstructed_baseline",
        "split_spec": str(spec_path.resolve()),
        "split_spec_sha256": _sha256(spec_path),
        "train_objects": sorted(train_objects),
        "heldout_objects": sorted(heldout_objects),
        "train": train, "heldout": heldout,
        "train_motion_root": str((output_root / "train").resolve()),
        "heldout_motion_root": str((output_root / "heldout").resolve()),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(spec_path=args.spec, output_root=args.output_root)
    print(json.dumps({"train_objects": manifest["train_objects"],
                      "heldout_objects": manifest["heldout_objects"],
                      "manifest": str(args.output_root / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
