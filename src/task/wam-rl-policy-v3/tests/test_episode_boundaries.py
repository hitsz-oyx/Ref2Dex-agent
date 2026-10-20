"""Regression cases for control, task and reporting semantics."""
import numpy as np
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler
from wam_rl_policy_v3.episode_metrics import episode_outcome
from wam_rl_policy_v3.execution_identity import validate_execution
import pytest
import torch
from types import SimpleNamespace
from wam_rl_policy_v3.official_bridge import ScheduledOfficialEnv, prepare_official


def test_position_goal_does_not_restore_initial_orientation():
    start = np.eye(4); target = start.copy(); target[2, 3] = .1
    points = np.random.RandomState(0).normal(size=(32, 3)) * .03
    scheduler = OrderedGoalScheduler([start, target], [0, 2], points,
                                     goal_type='position_only')
    scheduler.update(start, 0)
    actual = target.copy()
    actual[:3, :3] = Rotation.from_euler('z', 20, degrees=True).as_matrix()
    for tick in range(1, 20):
        result = scheduler.update(actual, tick / 60)
        np.testing.assert_allclose(result['goal_pose'][:3, :3], actual[:3, :3])
        assert result['goal_rotation_rad'] < 1e-8
    assert result['complete']
    # A task with an actual orientation target retains the SE(3) constraint.
    pose_scheduler = OrderedGoalScheduler([start, target], [0, 2], points)
    assert not pose_scheduler._reached(actual, target)


def test_position_goal_matches_optimized_scheduler():
    from wam_rl_policy_v3.goal_scheduler import update_schedulers
    start=np.eye(4);target=start.copy();target[2,3]=.1
    points=np.random.RandomState(3).normal(size=(32,3))*.03
    slow=OrderedGoalScheduler([start,target],[0,2],points,goal_type='position_only')
    fast=OrderedGoalScheduler([start,target],[0,2],points,goal_type='position_only')
    for tick in range(30):
        actual=start.copy();actual[2,3]=tick*.003
        actual[:3,:3]=Rotation.from_euler('z',tick*2,degrees=True).as_matrix()
        hand=np.zeros((11,3))
        a=slow.update(actual,tick/60,hand_points=hand)
        b=update_schedulers([fast],actual[None],np.array([tick/60]),hand[None],fast_geometry=True)[0]
        np.testing.assert_allclose(a['goal_delta'],b['goal_delta'],atol=1e-10)
        assert a['confirmed_index']==b['confirmed_index'] and a['mode']==b['mode']


def test_hold_drop_and_workspace_do_not_become_final_success():
    record = episode_outcome(True, 65, 0, False, False)
    assert record['native_success'] and record['ever_held60'] and record['drop_after_hold']
    assert not record['final_held60'] and not record['final_task_success']
    record = episode_outcome(True, 80, 80, True, True)
    assert record['native_success'] and record['final_held60']
    assert not record['final_task_success'] and not record['final_stable_success']
    assert record['termination_reason'] == 'wam_workspace'
    recovered = episode_outcome(True,80,80,False,True,dropped_after_hold=True)
    assert recovered['drop_after_hold'] and recovered['final_stable_success']


def test_resume_rejects_changed_wam_and_unbound_legacy_checkpoint():
    current = dict(wam='new', geometry={'jar': 'a'}, reset_protocol='same-step.v2')
    with pytest.raises(ValueError, match='migration'):
        validate_execution(None, current)
    with pytest.raises(ValueError, match='wam'):
        validate_execution(dict(current, wam='old'), current)
    change = validate_execution(None, current, migration_reason='explicit legacy warmstart')
    assert change['changed']
    assert not validate_execution(current, current)['changed']
    subset = dict(current, geometry={})
    assert not validate_execution(current, subset, evaluating=True)['changed']
    with pytest.raises(ValueError, match='geometry'):
        validate_execution(current, subset)


class DelayedResetBase:
    def __init__(self):
        self.num_envs = self.num_environments = 2
        self.num_states = 0; self.num_actions = 24
        self.state_space = self.action_space = None; self.rl_device = 'cpu'
        self.task = SimpleNamespace(reset_buf=torch.zeros(2,dtype=torch.long),successes=torch.zeros(2))
        self.episodes = torch.zeros(2,dtype=torch.long)
        self.trace = []; self.calls = 0

    def reset(self):
        return torch.zeros(2,300)

    def step(self, actions, tick):
        self.episodes[self.task.reset_buf.bool()] += 1
        self.task.reset_buf[:] = 0
        self.trace.append((actions.clone(), self.episodes.clone()))
        observation = torch.zeros(2,300); observation[:,0] = self.episodes
        if self.calls == 0: self.task.reset_buf[0] = 1
        self.calls += 1
        return observation, torch.ones(2), self.task.reset_buf.clone(), {}


def test_reset_initialization_is_not_a_policy_transition_or_an_old_action():
    base = DelayedResetBase()
    poses = np.repeat(np.eye(4)[None],2,axis=0)
    env = ScheduledOfficialEnv(base, lambda:dict(pose=poses,goal=np.ones((2,3))),
                               SimpleNamespace(device='cpu'), native_only=True)
    observation = env.reset()
    validity = []
    for _ in range(3):
        actions = torch.full((2,24), .75); actions[:,0] = observation[:,0] + .25
        before = actions.clone()
        observation, reward, done, info = env.step(actions)
        torch.testing.assert_close(actions,before)
        validity.append(info['transition_valid'].tolist())
        if len(validity)==2:
            assert done.tolist()==[1,0] and reward.tolist()==[0,1]
    assert validity == [[True,True],[False,True],[True,True]]
    assert not base.trace[1][0][0].any()  # reset env executes only initialization
    assert base.trace[1][0][1,0] == .25  # continuing env keeps its own action
    assert base.trace[2][0][0,0] == 1.25  # first policy action uses new observation
    assert env.completed_episodes==1 and env.initialization_steps==1


def test_masked_official_storage_excludes_reset_rows_and_preserves_terminal_return(tmp_path):
    from pathlib import Path
    import importlib.util
    root = Path(__file__).resolve().parents[4]
    official = prepare_official(root,tmp_path,masked_resets=True)
    spec = importlib.util.spec_from_file_location('masked_storage',official/'algorithms/rl/ppo/storage.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    storage = module.RolloutStorage(2,8,(300,),(0,),(24,))
    for tick in range(8):
        valid = torch.tensor([tick!=1,True])
        done = torch.tensor([tick in (0,1),False])
        reward = torch.tensor([5. if tick==0 else 0.,1.])
        storage.add_transitions(torch.zeros(2,300),torch.zeros(2,0),torch.zeros(2,24),
            reward,done,torch.zeros(2,1),torch.zeros(2),torch.zeros(2,24),torch.zeros(2,24),valid)
    storage.compute_returns(torch.zeros(2,1),.99,.95)
    assert storage.returns[0,0,0]==5.  # genuine terminal transition remains
    indices = [i for batch in storage.mini_batch_generator(4) for i in batch]
    assert len(indices)==15 and 2 not in indices  # row t=1, env=0 is initialization
    assert storage.advantages[1,0,0]==0.
    selected = storage.advantages[storage.transition_valid]
    torch.testing.assert_close(selected.mean(),torch.tensor(0.),atol=1e-6,rtol=0)
    torch.testing.assert_close(selected.std(),torch.tensor(1.))


def test_drop_flag_survives_recovery():
    base = DelayedResetBase()
    poses = np.repeat(np.eye(4)[None],2,axis=0)
    goals = np.tile([0.,0.,.15],(2,1))
    env = ScheduledOfficialEnv(base,lambda:dict(pose=poses,goal=goals),
                               SimpleNamespace(device='cpu'),native_only=True)
    env.reset()
    for z,frames in [(.15,65),(0.,1),(.15,70)]:
        poses[:,2,3]=z
        for _ in range(frames):env.augment(torch.zeros(2,300),[])
    assert env.streak[0]>=60 and env.dropped_after_hold[0]


def test_common_workspace_matches_native_and_wam_during_startup():
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    from types import MethodType
    points = np.random.RandomState(4).normal(size=(32,3))*.02
    geometry = (np.zeros((512,3),np.float32),np.zeros((512,3)),points,np.zeros((128,3)),np.zeros((128,3)))
    runtime = SimpleNamespace(device=torch.device('cpu'),geometry=[geometry]*2,
                              config=dict(voxel_origin_m=[-2.,-2.,-2.],voxel_m=.01,hand_in_ptv3=True))
    runtime.workspace_valid=MethodType(V4Runtime.workspace_valid,runtime)
    poses=np.repeat(np.eye(4)[None],2,axis=0)
    hands=np.zeros((2,11,3));hands[0,0,0]=-3
    snapshot=lambda:dict(pose=poses,goal=np.ones((2,3)),hand=hands)
    results=[]
    for native in (True,False):
        env=ScheduledOfficialEnv(DelayedResetBase(),snapshot,runtime,native_only=native,
                                 termination_policy='common_workspace')
        env.reset();results.append(env.workspace_invalid.copy())
    np.testing.assert_array_equal(results,[[True,False],[True,False]])
