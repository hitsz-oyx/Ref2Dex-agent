"""Physical edge cases that would make a static-goal filter unsafe."""
import importlib.util
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('curation_audit',Path(__file__).resolve().parents[1]/'tools/audit/audit_native_curation.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_rotating_and_returning_object_is_not_idle():
    poses=np.repeat(np.eye(4)[None],25,0)
    angle=np.sin(np.arange(25)*np.pi/24)
    poses[:,0,0]=poses[:,1,1]=np.cos(angle)
    poses[:,0,1]=-np.sin(angle);poses[:,1,0]=np.sin(angle)
    goal,peak=audit.object_motion(np.array([[.1,0,0],[-.1,0,0]]),poses)
    assert goal[0]<1e-6 and peak[0]>90
    row=np.array([[goal[0],peak[0],0,0,200,200,200,200,0]])
    assert audit.masks(row)['endpoint_static_excursion'][0]
    assert audit.masks(row)['stratum/manipulation'][0]


def test_static_contact_and_approach_are_preserved_before_idle():
    rows=np.array([[0,0,0,0,5,5,5,5,0],[0,0,30,20,120,70,70,70,50],
                   [0,0,40,20,200,200,200,5,0]])
    masks=audit.masks(rows)
    assert masks['stratum/contact_or_approach'].tolist()==[True,True,False]
    assert not masks['stratum/idle_away'].any()
    assert masks['static_goal_far_moving_other_near'][2]


def test_surface_distance_masks_missing_hand_and_is_rigid_invariant():
    points=np.array([[0,0,0],[1,0,0]],float)
    poses=np.repeat(np.eye(4)[None],25,0);poses[:,:3,3]=[2,3,4]
    poses[:,:2,:2]=[[0,-1],[1,0]]
    hand=np.zeros((25,2,11,3));hand[:,0]=[2,3,4];hand[:,1]=[2,3,4.02]
    valid=np.repeat([[False,True]],25,axis=0)
    dist=audit.surface_distance(points,poses,hand,valid,np.ones(25,bool))
    assert np.allclose(dist,20)
    peak,mean=audit.hand_motion(hand,valid)
    assert peak[0]==0 and mean[0]==0
