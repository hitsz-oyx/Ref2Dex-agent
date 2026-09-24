from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_build_stage_disjoint_views_and_provenance(tmp_path: Path):
    spec = {"train": [_entry(tmp_path, "s1_airplane_lift"),
                      _entry(tmp_path, "s1_mug_lift")],
            "heldout": [_entry(tmp_path, "s7_apple_lift")]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    manifest = build(spec_path=spec_path, output_root=tmp_path / "split")
    assert manifest["train_objects"] == ["airplane", "mug"]
    assert manifest["heldout_objects"] == ["apple"]
    assert (tmp_path / "split/train/s1_mug_lift").is_symlink()
    assert (tmp_path / "split/heldout/s7_apple_lift").is_symlink()


def test_rejects_object_leakage_before_creating_output(tmp_path: Path):
    spec = {"train": [_entry(tmp_path, "s1_apple_lift")],
            "heldout": [_entry(tmp_path, "s7_apple_lift")]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="object leakage"):
        build(spec_path=spec_path, output_root=tmp_path / "split")
    assert not (tmp_path / "split").exists()
