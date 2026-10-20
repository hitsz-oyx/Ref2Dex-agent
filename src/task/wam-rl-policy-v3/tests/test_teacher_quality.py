import numpy as np
from wam_rl_policy_v3.teacher_quality import failure_masks,measured_failures,table_support


def test_failure_prebuffer_excludes_transitions_and_entire_h4_future_support():
    bad=np.zeros(150,bool);bad[100]=True
    commands,windows,unsafe=failure_masks(bad)
    assert commands[:49].all() and not commands[49:101].any() and commands[101:].all()
    assert windows[:26].all() and not windows[26:104].any() and windows[104:].all()
    assert unsafe.sum()==51


def test_supported_release_is_distinct_from_airborne_loss_after_grasp():
    pose=np.repeat(np.eye(4)[None],6,0)
    coupled=np.array([False,True,True,False,False,False])
    lost=np.array([True,False,False,True,True,True])
    support=np.array([False,False,False,True,False,False])
    r=measured_failures(pose,coupled,lost,support,np.zeros(6),np.ones(6,bool))
    assert r['unsupported_loss_after_coupling'].tolist()==[False,False,False,False,True,True]


def test_support_requires_table_geometry_and_current_net_force():
    pose=np.repeat(np.eye(4)[None],3,0);pose[:,2,3]=[.82,.9,.75]
    points=np.array([[-.01,-.01,-.02],[.01,.01,-.02],[0,0,.02]])
    table=np.array([0,0,.8,0,0,0,1]);bounds=np.array([[-.5,-.5,-.05],[.5,.5,0]])
    support,gap,inside=table_support(pose,points,table,bounds,np.array([[0,0,1]]*3))
    assert support.tolist()==[True,False,False] and inside.all()
    assert gap[2]<-.02


def test_support_handles_native_table_with_thin_y_axis_and_downward_local_normal():
    pose=np.repeat(np.eye(4)[None],3,0);pose[:,2,3]=[.825,.9,.825];pose[2,0,3]=.5
    points=np.array([[-.01,-.01,-.02],[.01,.01,-.02],[0,0,.02]])
    # DExplore's tabletop mesh lies in local XZ; its local -Y faces world +Z.
    table=np.array([0,0,.8,0,np.sqrt(.5),-np.sqrt(.5),0])
    bounds=np.array([[-.225,-.00274,-.27],[.225,.00274,.27]])
    support,gap,inside=table_support(pose,points,table,bounds,np.array([[0,0,1]]*3))
    assert support.tolist()==[True,False,False]
    assert inside.tolist()==[True,True,False]
    np.testing.assert_allclose(gap[:2],[.00226,.07726],atol=1e-8)
