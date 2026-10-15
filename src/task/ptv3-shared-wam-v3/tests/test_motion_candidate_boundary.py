from pathlib import Path
import importlib.util
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'tools/audit/prepare_strict_motion_admission.py'
spec=importlib.util.spec_from_file_location('strict_candidate_test',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_native_bad_transition_partly_enters_interpolated_forecast():
    metrics={k:dict(full_capture=dict(maximum=dict(value=0.)),retained_window_context=dict(maximum=dict(value=0.))) for k in mod.CAPS}
    metrics['hand_speed_mps']['full_capture']['maximum']['value']=8.1102276
    # Native306->308 is fast; query283 ends307, so only native endpoint306 is covered.
    covered=np.zeros(599,bool);covered[280:308]=True
    assert not (covered[306]&covered[308])
    assert mod.threshold_candidate(metrics)
    # Inclusive existing isolation catches the interpolated half306->307.
    bad=np.zeros(599,bool);bad[306:309]=True;p=np.r_[0,np.cumsum(bad)]
    assert p[283+25]-p[283-3]>0
    assert p[281+25]-p[281-3]==0


def test_existing_exclusions_cannot_be_overwritten_by_rebuilding_entry():
    mod.require_unfiltered_base({})
    with pytest.raises(ValueError,match='unfiltered quality base'):mod.require_unfiltered_base(dict(quality_admission={'index_path':'old.npz'}))
