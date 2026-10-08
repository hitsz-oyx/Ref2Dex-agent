import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('hocap_probe', Path(__file__).parents[1]/'tools/run/evaluate_hocap_frame_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_unverified_clock_is_explicit_and_never_training_eligible():
    meta=dict(schema='ref2dex.hocap-frame-probe.v1',training_allowed=False,
              fps=None,source_fps_verified=False,nominal_feature_fps=30)
    probe.validate_frame_manifest(meta,True)
    with pytest.raises(ValueError):probe.validate_frame_manifest(meta,False)
    for patch in ({'training_allowed':True},{'fps':30},{'source_fps_verified':True}):
        with pytest.raises(ValueError):probe.validate_frame_manifest(dict(meta,**patch),True)


def test_panels_preserve_sequence_coverage_motion_and_unique_windows():
    rows=np.array([[s,0,t,c] for s in range(3) for t,c in [(3,0),(11,0),(19,0),(27,1)]])
    moving,natural=probe.pick_panel(rows,3,228)
    assert np.all(rows[moving,3]==0)
    assert len(moving)==6 and len(natural)==3
    assert len(np.unique(np.concatenate((moving,natural))))==9
    assert set(rows[moving,0])==set(rows[natural,0])=={0,1,2}
    repeat=probe.pick_panel(rows,3,228)
    assert np.array_equal(moving,repeat[0]) and np.array_equal(natural,repeat[1])


def test_missing_motion_or_empty_sequence_does_not_fabricate_a_moving_row():
    rows=np.array([[0,0,3,1],[0,0,11,1],[1,0,3,0]])
    moving,natural=probe.pick_panel(rows,3,228)
    assert moving.tolist()==[2]
    assert len(natural)==1 and rows[natural[0],0]==0
