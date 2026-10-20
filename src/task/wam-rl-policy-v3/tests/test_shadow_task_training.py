import numpy as np
import torch
from scipy.spatial.transform import Rotation

from wam_rl_policy_v3.shadow_task_contract import measured_velocity,conditioning,CURRENT_SCHEMA
from wam_rl_policy_v3.bc_mlp import TaskShadowMLP
from wam_rl_policy_v3.scheduled_current_cache import RecordInputs
from wam_rl_policy_v3.shadow_task_probe import TaskGoalScheduler


def test_native_yaw_velocity_is_decoded_without_future_or_force_sensors():
    yaw=Rotation.from_euler('z',.8).as_matrix()
    local=Rotation.from_rotvec([.1,.2,.3]).as_matrix()
    pose=np.eye(4);pose[:3,:3]=yaw@local
    obs=np.zeros(300);obs[194:198]=Rotation.from_matrix(local).as_quat()
    obs[198:201]=[.3,-.2,.1];obs[201:204]=np.array([.4,.5,-.2])*.2
    np.testing.assert_allclose(measured_velocity(pose,obs),np.r_[yaw@obs[198:201],yaw@np.array([.4,.5,-.2])],atol=1e-7)


def test_offline_cache_uses_the_exact_online_task10_and_wam_goal(tmp_path):
    rng=np.random.default_rng(7);n=20
    poses=np.repeat(np.eye(4)[None],n,0);poses[:,0,3]=np.arange(n)*.001
    surface=rng.normal(size=(512,3))*.008
    normals=surface/np.linalg.norm(surface,axis=-1,keepdims=True)
    geometry=(surface,normals,surface[:32],surface[:128],normals[:128])
    hand=np.tile(poses[:,:3,3,None].transpose(0,2,1),(1,11,1));hand[:,:,2]+=.004
    obs=np.zeros((n,300));obs[:,197]=1;obs[:,198]=.06
    goal=np.tile([.2,0,0],(n,1));action=np.zeros((n-1,24))
    np.savez(tmp_path/'seq.npz',poses=poses,hand=hand,observation=obs,action=action,goal=goal,time=np.arange(n)/60)
    row=dict(source=dict(path='seq.npz',ticks=list(range(n-1))),lo=0,hi=n-1,geometry=0)
    data=RecordInputs(tmp_path,[row],[geometry],task_goal=True)[0]
    state=dict(pose=poses[:1],goal=goal[:1],velocity=np.zeros((1,6)))
    online=TaskGoalScheduler(state,0,geometry,force_feedback=False)
    for t in range(n-1):
        state['velocity']=measured_velocity(poses[t],obs[t])[None];online.feedback(state,0,t>0)
        result=online.update(poses[t],t/60,hand_points=hand[t])
        np.testing.assert_array_equal(data['task'][t],result['task'])
        if t>=6:np.testing.assert_array_equal(data['goals'][t-6],result['goal_delta'])
    assert not online.force_valid


def test_learned_task_columns_survive_current_to_scheduled_expansion():
    torch.manual_seed(241)
    current=TaskShadowMLP()
    with torch.no_grad():current.net[-1].weight.normal_(0,.01)
    scheduled=TaskShadowMLP(True)
    scheduled.initialize_task_current(dict(schema=CURRENT_SCHEMA,model=current.state_dict()))
    x=torch.randn(8,575)
    torch.testing.assert_close(scheduled(x),current(x[:,:310]),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(scheduled.net[0].weight[:,300:310],current.net[0].weight[:,300:310],rtol=0,atol=0)
    assert not scheduled.net[0].weight[:,310:].any()
    assert conditioning()['fields']==dict(robot=300,task=10,tau=264,tau_valid=1)
