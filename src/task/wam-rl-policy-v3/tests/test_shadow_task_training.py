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


def test_uneven_three_rank_bc_gradients_equal_the_single_global_batch():
    import copy
    from wam_rl_policy_v3.task_bc_training import global_mean_loss
    torch.manual_seed(7)
    model=torch.nn.Linear(7,3);x=torch.randn(17,7);y=torch.randn(17,3);scale=torch.rand(3)+.1
    ((model(x)-y)/scale).abs().mean().backward()
    grads=[]
    for ids in torch.tensor_split(torch.arange(17),3):
        local=copy.deepcopy(model);local.zero_grad()
        global_mean_loss(local(x[ids]),y[ids],scale,17,3).backward()
        grads.append([p.grad.clone() for p in local.parameters()])
    for i,parameter in enumerate(model.parameters()):
        torch.testing.assert_close(sum(g[i] for g in grads)/3,parameter.grad,rtol=1e-6,atol=1e-7)


def test_trained575_actor_loads_goal_weights_directly_without_zero_expansion():
    from wam_rl_policy_v3.scheduled_executor import BCWarmStartActor
    from wam_rl_policy_v3.shadow_task_contract import SCHEDULED_SCHEMA
    model=TaskShadowMLP(True)
    with torch.no_grad():
        model.net[0].weight[:,300:310].fill_(.123)
        model.net[-1].weight.normal_(0,.01)
    packet=dict(schema=SCHEDULED_SCHEMA,model=model.state_dict(),task_columns_trained=True,
                config=dict(conditioning=conditioning()))
    actor=BCWarmStartActor(packet)
    assert actor.task_model and actor.actor.input_dim==575
    torch.testing.assert_close(actor.actor.net[0].weight[:,300:310],model.net[0].weight[:,300:310],rtol=0,atol=0)
    inputs=torch.randn(8,575)
    torch.testing.assert_close(actor(inputs),model(inputs),rtol=0,atol=0)


def test_current310_keeps_new_episode_observation_and_goal_without_wam_prediction():
    from types import SimpleNamespace
    from test_episode_boundaries import DelayedResetBase
    from wam_rl_policy_v3.shadow_task_probe import TaskCurrentOfficialEnv
    base=DelayedResetBase();poses=np.repeat(np.eye(4)[None],2,axis=0)
    points=np.random.default_rng(7).normal(size=(32,3))*.01
    runtime=SimpleNamespace(device='cpu',geometry=[(points,None,points)]*2)
    runtime.predict=lambda *args:(_ for _ in ()).throw(AssertionError('Current310 must not run WAM'))
    snapshot=lambda:dict(pose=poses.copy(),hand=np.ones((2,11,3)),goal=np.tile([0,0,.2],(2,1)),velocity=np.zeros((2,6)))
    env=TaskCurrentOfficialEnv(base,snapshot,runtime,termination_policy='native')
    observation=env.reset()
    assert observation.shape==(2,310) and env.inputs is None
    np.testing.assert_array_equal(observation.numpy()[:,307:310],[[1,0,0],[1,0,0]])
    for tick in range(3):
        action=torch.ones(2,24)*.25;action[:,0]=observation[:,0]+.25
        observation,reward,done,info=env.step(action)
        if tick==1:assert not info['transition_valid'][0] and info['transition_valid'][1]
    assert base.trace[1][0][0].abs().max()==0
    assert base.trace[2][0][0,0]==1.25
    assert env.schedulers[0].controller.observer.last_time==1/60
    assert env.schedulers[1].controller.observer.last_time==3/60
