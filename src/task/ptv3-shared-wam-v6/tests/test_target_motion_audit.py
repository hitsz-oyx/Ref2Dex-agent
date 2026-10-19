"""Future-horizon runaway and coordinate/time semantics of the read-only audit."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'tools/audit/audit_target_motion.py'
spec=importlib.util.spec_from_file_location('wam_target_motion_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_runaway_after_clean_anchor_is_detected_through_full_future():
    poses=np.broadcast_to(np.eye(4),(60,1,4,4)).copy()
    poses[36:,0,1,3]=.25+np.arange(24)*.4
    rows=np.array([[0,0,3],[0,0,11],[0,0,19],[0,0,35]])
    maximum,speed,tail=audit.window_motion(poses,np.ones((60,1),bool),rows,np.ones(60,bool))
    # The frame35 anchor is healthy; frame36 is the runaway onset. A check
    # limited to H4 would miss windows19/27/35 whose future includes it.
    assert np.allclose(maximum[:2],0)
    assert np.array_equal(tail,[False,False,True,True])
    assert maximum[-1]>8 and speed[-1]>10
    shifted=poses.copy();shifted[:,:,:3,3]+=[100,20,-50]
    angle=.7;c,s=np.cos(angle),np.sin(angle)
    world=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shifted[:,:,:3,3]=shifted[:,:,:3,3]@world.T
    assert np.allclose(audit.window_motion(shifted,np.ones((60,1),bool),rows,np.ones(60,bool))[0],maximum)


def test_observed_transition_clock_and_missing_pose_mask():
    poses=np.broadcast_to(np.eye(4),(60,1,4,4)).copy()
    poses[:,0,0,3]=np.arange(60)/30.
    observed=np.zeros(60,bool);observed[::2]=True
    rows=np.array([[0,0,3]])
    valid=np.ones((60,1),bool)
    maximum,speed,_=audit.window_motion(poses,valid,rows,observed)
    assert np.allclose(maximum,.8)
    assert np.allclose(speed,1.)
    valid[:]=False
    assert np.allclose(audit.window_motion(poses,valid,rows,observed)[1],0.)
