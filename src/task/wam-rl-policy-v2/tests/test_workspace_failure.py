import numpy as np
import torch
from types import SimpleNamespace
from wam_rl_policy_v2.wam_v4_runtime import V4Runtime
from wam_rl_policy_v2.official_bridge import ScheduledOfficialEnv


def test_workspace_rejects_observed_negative_grid_without_clipping():
    runtime=V4Runtime.__new__(V4Runtime)
    runtime.config=dict(voxel_origin_m=[-2.,-2.,-2.],voxel_m=.01,hand_in_ptv3=True)
    sample=dict(xyz=np.zeros((512,3),np.float32),point_valid=np.ones(512,bool),features=np.zeros((512,18),np.float32),
                start_hand=np.zeros((2,11,3),np.float32),start_hand_valid=np.array([[True]*11,[False]*11]))
    assert runtime.workspace_valid([sample]).tolist()==[True]
    sample['start_hand'][0,0,0]=-2.06
    assert runtime.workspace_valid([sample]).tolist()==[False]
    assert sample['start_hand'][0,0,0]==np.float32(-2.06)
    sample['start_hand_valid'][0,0]=False
    assert runtime.workspace_valid([sample]).tolist()==[True]


def test_workspace_failure_terminates_only_affected_environment():
    env=ScheduledOfficialEnv.__new__(ScheduledOfficialEnv)
    task=SimpleNamespace(reset_buf=torch.zeros(2,dtype=torch.long),successes=torch.zeros(2))
    env.base=SimpleNamespace(step=lambda actions,id:(torch.zeros(2,300),torch.zeros(2),torch.zeros(2,dtype=torch.long),{}))
    env.task=task;env.pending=np.zeros(2,bool);env.workspace_invalid=np.array([False,True])
    env.augment=lambda native,ids:torch.zeros(2,565)
    env.workspace_failures=env.reset_count=env.steps=env.completed_episodes=env.successful_episodes=env.held_episodes=0
    env.longest=np.zeros(2)
    _,_,done,_=env.step(torch.zeros(2,24))
    assert done.tolist()==[0,1]
    assert task.reset_buf.tolist()==[0,1]
    assert env.pending.tolist()==[False,True]
    assert env.workspace_failures==1
