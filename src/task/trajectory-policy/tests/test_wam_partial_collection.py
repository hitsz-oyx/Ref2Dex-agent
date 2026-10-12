"""A bounded collection may show hold45 but cannot claim final placement."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


path=Path(__file__).resolve().parents[1]/'tools/run/execute_wam_native_bridge.py'
spec=importlib.util.spec_from_file_location('wam_partial_collection_test',path)
bridge=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def packet(count=65):
    pose=np.repeat(np.eye(4)[None],count,axis=0)
    pose[:,2,3]=.78
    pose[1:,2,3]+=.05
    return dict(object_pose=pose,object_velocity=np.zeros((count,6)),
        surface_gap=np.zeros(count),support_gap=np.full(count,.05),
        table_footprint=np.ones(count,bool))


def test_partial_hold_does_not_expose_placement_or_success():
    result=bridge.partial_grasp_outcome(packet())
    assert result['maximum_held_frames']==64
    assert result['first_stable_tick']==45
    assert not {'success','controlled_final_place','terminal_settled_frames'}&set(result)
    assert result['metric']=='partial_hold45_only_no_placement'


def test_short_grasp_statistics_handle_under45_and_contact_break():
    result=bridge.partial_grasp_outcome(packet(25))
    assert result['maximum_held_frames']==24 and result['first_stable_tick'] is None
    data=packet()
    data['surface_gap'][32]=.02
    assert bridge.partial_grasp_outcome(data)['maximum_held_frames']==32


def test_supported_object_and_reset_do_not_count_as_held():
    data=packet()
    data['support_gap'][:]=0
    assert bridge.partial_grasp_outcome(data)['maximum_held_frames']==0


def test_nonfinite_partial_geometry_rejected():
    data=packet()
    data['object_velocity'][1,0]=np.nan
    with pytest.raises(ValueError):
        bridge.partial_grasp_outcome(data)
