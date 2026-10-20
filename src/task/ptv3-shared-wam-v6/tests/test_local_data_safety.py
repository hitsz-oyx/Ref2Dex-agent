"""Safety boundaries for local preparation; no GPU or training."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


path=Path(__file__).resolve().parents[1]/'tools/audit/prepare_local_data.py'
spec=importlib.util.spec_from_file_location('local_data_safety',path)
local=importlib.util.module_from_spec(spec);spec.loader.exec_module(local)


def test_refuses_peak_that_would_cross_operational_floor(monkeypatch,tmp_path):
    monkeypatch.setattr(local.shutil,'disk_usage',lambda _:SimpleNamespace(free=30*local.GIB))
    with pytest.raises(RuntimeError,match='Insufficient headroom'):
        local.capacity(tmp_path,11*local.GIB)
    assert not list(tmp_path.iterdir())


def test_boundary_preserves_twenty_gib_not_just_fifteen(monkeypatch,tmp_path):
    monkeypatch.setattr(local.shutil,'disk_usage',lambda _:SimpleNamespace(free=30*local.GIB))
    assert local.capacity(tmp_path,10*local.GIB)==30*local.GIB


@pytest.mark.parametrize('pose_width',[7,13])
def test_recorded_robot_pose7_and_state13_are_both_supported(tmp_path,pose_width):
    source=tmp_path/'source.npz';n=30
    np.savez(source,hand=np.zeros((n,11,3),np.float32),object=np.zeros((n,pose_width),np.float32),
        q=np.zeros((n,22),np.float32),hand_root=np.zeros((n,13),np.float32))
    job=dict(key='s4-fixture',source_id=4,source_row=dict(source_path=str(source)),
        objects=['fixture'],windows={'train':'unused.npy'},source_sequence='fixture')
    row=local.inspect_job(job)
    assert row['frames']==30 and row['objects']==1
    assert row['source']==str(source)
