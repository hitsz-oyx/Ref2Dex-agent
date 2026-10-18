import copy,os
import pytest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from wam_rl_policy_v2.batched_geometry import native_tau,online_batch,workspace_valid
from wam_rl_policy_v2.tau_contract import v4_trajectory_to_student
from wam_rl_policy_v2.wam_v4_runtime import v4_online_samples,collate_v4
from wam_rl_policy_v2.goal_scheduler import OrderedGoalScheduler,update_schedulers


def fixture(n=8):
    rng=np.random.RandomState(241)
    objects=np.zeros((n,4,7));objects[:,:,:3]=rng.randn(n,4,3)*.03
    objects[:,:,3:]=Rotation.from_rotvec(rng.randn(n*4,3)*.05).as_quat().reshape(n,4,4)
    hands=rng.randn(n,4,11,3)*.05
    geometries=[]
    for i in range(n):
        surface=rng.randn(512,3).astype('float32')*.03;normals=rng.randn(512,3).astype('float32');normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
        geometries.append((surface,normals,surface[:32].copy(),surface[:128].copy(),normals[:128].copy()))
    goals=rng.randn(n,32,3)*.01
    return objects,hands,geometries,goals


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if os.environ.get('REF2DEX_TEST_CUDA')=='1' else []))
def test_device_features_match_reference(device):
    objects,hands,geometries,goals=fixture()
    old=collate_v4(v4_online_samples(objects,hands,geometries,goal_delta=goals))
    new=online_batch(objects,hands,geometries,goals,device)
    assert old.keys()==new.keys()
    for key in old:
        torch.testing.assert_close(new[key].cpu(),old[key],rtol=1e-6,atol=2e-6)
    config=dict(voxel_origin_m=[-2]*3,voxel_m=.01)
    assert workspace_valid(new,config).all()
    new['start_hand'][0,0,0,0]=-2.06
    assert workspace_valid(new,config).tolist()==[False]+[True]*7


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if os.environ.get('REF2DEX_TEST_CUDA')=='1' else []))
def test_device_tau_matches_reference(device):
    rng=np.random.RandomState(41);tau=rng.randn(12,24,2,11,3).astype('float32')*.1
    rotation=Rotation.random(12,random_state=rng).as_matrix()
    old=np.stack([v4_trajectory_to_student(t,object_rotation=r) for t,r in zip(tau,rotation)])
    np.testing.assert_allclose(native_tau(torch.tensor(tau,device=device),torch.tensor(rotation,device=device)).cpu().numpy(),old,rtol=0,atol=1e-7)


def test_batched_scheduler_preserves_state_and_decisions():
    rng=np.random.RandomState(42);points=rng.randn(32,3)*.04;n=12
    start=np.repeat(np.eye(4)[None],n,0);goal=start.copy();goal[:,2,3]=.2
    single=[OrderedGoalScheduler([s,g],[0,2],points) for s,g in zip(start,goal)];batched=copy.deepcopy(single)
    for tick in range(18):
        actual=start.copy();actual[:,:3,:3]=Rotation.from_rotvec(rng.randn(n,3)*.5).as_matrix();actual[:,:3,3]=rng.randn(n,3)*.12
        if tick==0:actual=start.copy()
        hands=rng.randn(n,11,3)*.05;times=np.full(n,tick/60)
        if tick==10:single[0].reset();batched[0].reset()
        old=[s.update(p,t,hand_points=h) for s,p,t,h in zip(single,actual,times,hands)]
        new=update_schedulers(batched,actual,times,hands)
        for a,b in zip(old,new):
            for key in ['goal_pose','goal_delta','confirmed_progress','goal_point_m','lookahead_m']:
                np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-8)
            for key in ['mode','reason','confirmed_index','arrival_witness','complete','timed_out']:assert a[key]==b[key]


def test_pipeline_partial_reset_and_workspace_failure(monkeypatch):
    from wam_rl_policy_v2.batched_scheduled_inputs import BatchedScheduledInputs
    from wam_rl_policy_v2.wam_v4_runtime import V4Runtime
    objects,hands,geometries,goals=fixture(3)
    class Runtime:
        device=torch.device('cpu')
        config=dict(voxel_origin_m=[-2]*3,voxel_m=.01,hand_in_ptv3=True)
        geometry=geometries
        workspace_valid=V4Runtime.workspace_valid
        def predict(self,batch):
            return dict(trajectory=batch['start_hand'][:,None].repeat(1,24,1,1,1)*.1)
    start=np.repeat(np.eye(4)[None],3,0);goal=start.copy();goal[:,2,3]=.2
    pipelines=[]
    for mode in ['legacy','tau','inputs','scheduler','all']:
        monkeypatch.setenv('REF2DEX_BATCH_PIPELINE',mode)
        pipelines.append(BatchedScheduledInputs(Runtime(),[OrderedGoalScheduler([a,b],[0,2],g[2]) for a,b,g in zip(start,goal,geometries)]))
    for tick in range(14):
        times=np.full(3,tick/60);observed=hands[:,-1].copy()
        if tick>=9:times[0]=(tick-9)/60
        if tick==9:
            for pipe in pipelines:pipe.reset([0])
        if tick==12:observed[1,0,0]=-2.06
        values=[pipe.observation(np.zeros((3,300)),start,observed,times) for pipe in pipelines]
        for v in values[1:]:
            np.testing.assert_allclose(np.asarray(v['observation']),values[0]['observation'],atol=2e-6,rtol=1e-6)
            np.testing.assert_array_equal(v['workspace_invalid'],values[0]['workspace_invalid'])
