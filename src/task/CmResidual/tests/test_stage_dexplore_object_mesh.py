from __future__ import annotations

from pathlib import Path

import pytest

from src.task.CmResidual.tools.data.stage_dexplore_object_mesh import stage


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    dexplore = tmp_path / "dexplore"
    mesh = raw / "apple/mesh.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"grab-apple-mesh")
    mjcf = dexplore / "dexplore/data/assets/mjcf"
    mjcf.mkdir(parents=True)
    (mjcf / "apple.urdf").write_text(
        '<mesh filename="objects/apple/apple.obj"/>', encoding="utf-8")
    return raw, dexplore, mjcf / "objects/apple/apple.obj"


def test_stage_links_read_only_source_without_overwriting(tmp_path: Path):
    raw, dexplore, target = _fixture(tmp_path)
    first = stage(object_name="apple", raw_object_root=raw, dexplore_root=dexplore)
    second = stage(object_name="apple", raw_object_root=raw, dexplore_root=dexplore)
    assert first["created"] is True and second["created"] is False
    assert target.is_symlink() and target.resolve() == (raw / "apple/mesh.obj").resolve()
    assert (raw / "apple/mesh.obj").read_bytes() == b"grab-apple-mesh"


def test_stage_rejects_unexpected_existing_asset(tmp_path: Path):
    raw, dexplore, target = _fixture(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"another mesh")
    with pytest.raises(ValueError, match="refusing to replace"):
        stage(object_name="apple", raw_object_root=raw, dexplore_root=dexplore)
    assert target.read_bytes() == b"another mesh"


def test_stage_rejects_unsafe_name(tmp_path: Path):
    raw, dexplore, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="invalid"):
        stage(object_name="../apple", raw_object_root=raw, dexplore_root=dexplore)
