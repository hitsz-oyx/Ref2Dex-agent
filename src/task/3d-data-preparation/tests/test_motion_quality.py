import json
from pathlib import Path
import numpy as np
from pose_mesh_data.motion_quality import motion_features,events,rejected_ticks

POLICY=json.loads((Path(__file__).resolve().parents[1]/'configs/unidex_motion_v1.json').read_text())


def inputs():
    t=np.arange(100)/30
    obj=np.zeros((100,7));obj[:,6]=1;obj[:,2]=.6
    hand=np.zeros((100,11,3));hand[:,:,2]=.6
    return t,obj,hand


def test_impulse_cuts_tail_but_preserves_complete_prefix():
    t,obj,hand=inputs();obj[79:,0]+=.6
    f=motion_features(t,obj,hand);e=events(f,POLICY)
    assert 79 in e['translational_impulse']
    ticks=np.array([3,11,19,27,35,43,51,54,55,59,67,75,80])
    assert np.array_equal(ticks[rejected_ticks(ticks,e['translational_impulse'])],[55,59,67,75,80])


def test_gravity_drop_and_floor_stop_are_retained():
    t,obj,hand=inputs();obj[:,2]=np.maximum(.1,.6-.5*9.81*t**2)
    assert not any(len(v) for v in events(motion_features(t,obj,hand),POLICY).values())


def test_initial_flight_is_flagged_without_treating_constant_speed_as_impulse():
    t,obj,hand=inputs();obj[:,0]=t*6
    ev=events(motion_features(t,obj,hand),POLICY)
    assert not len(ev['translational_impulse'])
    assert ev['initial_fast_flight'].tolist()==[1,2]
    obj[:,0]=0;obj[50:,0]=.2;hand[50:,:,0]=.2
    assert not len(events(motion_features(t,obj,hand),POLICY)['translational_impulse'])


def test_quaternion_signs_do_not_create_spin_events():
    t,obj,hand=inputs();obj[:,5]=np.sin(t);obj[:,6]=np.cos(t)
    a=motion_features(t,obj,hand);obj[::2,3:]*=-1
    b=motion_features(t,obj,hand)
    assert np.array_equal(a['angular_speed_radps'],b['angular_speed_radps'])
    assert not len(events(b,POLICY)['extreme_rotation'])


def test_shared_extreme_impulse_is_not_hidden_by_hand_speed():
    t,obj,hand=inputs();obj[50:,0]=.3;hand[50:,:,0]=.3
    e=events(motion_features(t,obj,hand),POLICY)
    assert not len(e['translational_impulse'])
    assert 50 in e['severe_shared_impulse']


def test_invalid_policy_fails_closed():
    import pytest
    t,obj,hand=inputs()
    with pytest.raises(ValueError):
        events(motion_features(t,obj,hand),{**POLICY,'speed_mps':float('nan')})
