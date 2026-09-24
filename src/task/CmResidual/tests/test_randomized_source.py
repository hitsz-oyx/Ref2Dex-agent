from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.task.CmResidual.randomized_source import sha256, validate
from src.task.CmResidual.tools.data.build_object_disjoint_split import build


def _entry(tmp_path: Path, sequence: str) -> dict:
    directory = tmp_path / "source" / sequence
    directory.mkdir(parents=True)
    (directory / "interaction_hand_inspire.pt").write_bytes(sequence.encode())
    manifest = tmp_path / f"{sequence}.json"
    manifest.write_text(json.dumps({"classification": "reconstructed_baseline",
                                    "sequence": sequence}), encoding="utf-8")
    return {"sequence": sequence, "sequence_dir": str(directory),
            "input_manifest": str(manifest)}


def test_validate_pinned_partition_and_reject_drift(tmp_path: Path):
    spec = {"train": [_entry(tmp_path, "s1_mug_lift")],
            "heldout": [_entry(tmp_path, "s7_apple_lift")]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    root = tmp_path / "split"
    build(spec_path=spec_path, output_root=root)
    checkpoint = tmp_path / "actor.pth"
    checkpoint.write_bytes(b"self-trained actor")
    kwargs = {"checkpoint": checkpoint, "checkpoint_sha256": sha256(checkpoint),
              "motion_root": root / "heldout", "manifest_path": root / "manifest.json",
              "manifest_sha256": sha256(root / "manifest.json"), "partition": "heldout"}
    assert validate(**kwargs)["objects"] == ["apple"]
    with pytest.raises(ValueError, match="motion root"):
        validate(**{**kwargs, "motion_root": root / "train"})
    (root / "heldout/s7_apple_lift/interaction_hand_inspire.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tensor drift"):
        validate(**kwargs)
