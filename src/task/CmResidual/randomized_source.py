"""Validate a hash-pinned, object-disjoint source for physical interventions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(*, checkpoint: Path, checkpoint_sha256: str,
             motion_root: Path, manifest_path: Path, manifest_sha256: str,
             partition: str) -> dict:
    if partition not in ("train", "heldout"):
        raise ValueError("partition must be train or heldout")
    if not checkpoint.is_file() or sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint hash mismatch")
    if not manifest_path.is_file() or sha256(manifest_path) != manifest_sha256:
        raise ValueError("split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "ref2dex.object_disjoint_motion_split.v1" or
            set(manifest["train_objects"]) & set(manifest["heldout_objects"])):
        raise ValueError("invalid object-disjoint split")
    expected_root = Path(manifest[f"{partition}_motion_root"]).resolve()
    if motion_root.resolve() != expected_root:
        raise ValueError("motion root does not match selected split partition")
    entries = manifest[partition]
    if not entries:
        raise ValueError("empty split partition")
    actual_names = {path.name for path in motion_root.iterdir() if path.is_dir()}
    expected_names = {entry["sequence"] for entry in entries}
    if actual_names != expected_names:
        raise ValueError("motion root sequence set drift")
    for entry in entries:
        link = motion_root / entry["sequence"]
        if (link.resolve() != Path(entry["sequence_dir"]).resolve() or
                sha256(link / "interaction_hand_inspire.pt") != entry["tensor_sha256"]):
            raise ValueError(f"motion tensor drift: {entry['sequence']}")
    return {"checkpoint_sha256": checkpoint_sha256,
            "motion_manifest_sha256": manifest_sha256,
            "partition": partition, "objects": sorted({entry["object"] for entry in entries})}
