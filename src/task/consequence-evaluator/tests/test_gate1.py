"""Independent recoverable outcome and frozen GT candidate selection."""
from pathlib import Path
import runpy
import sys
import numpy as np
import pytest
TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.gate1 import candidate_plan, choose_candidate, episode_outcome, paired_counts
from consequence_evaluator.native_backend import canonical_device, resolve_backend, resolve_legacy_backend


def fixture():
    height = np.zeros(231)
    height[20:181] = .06
    height[181:196] = np.linspace(.06, 0, 15)
    poses = np.tile(np.eye(4), (231, 1, 1)); poses[:, 2, 3] = .8 + height
    gap = np.full(231, .003); gap[195:] = .05
    velocity = np.zeros((231, 6)); velocity[181:195, 2] = -.15
    return dict(object_pose=poses, surface_gap=gap, object_velocity=velocity,
                support_gap=height.copy(), table_footprint=np.ones(231, bool))


def test_recovered_loss_can_succeed_and_is_reported_separately():
    packet = fixture()
    packet['surface_gap'][100:120] = .08
    packet['object_velocity'][100:120, 2] = -.5
    result = episode_outcome(packet)
    assert result['success'] and result['intermediate_loss_events'] == 1
    assert result['last_stable_tick'] == 188
    assert result['maximum_held_frames'] == 80


def test_final_uncontrolled_fall_or_floor_is_not_success():
    packet = fixture()
    assert episode_outcome(packet)['success']
    packet['surface_gap'][185:195] = .08
    packet['object_velocity'][185:195, 2] = -.5
    assert not episode_outcome(packet)['success']
    packet = fixture(); packet['support_gap'][196:] = -.7
    assert not episode_outcome(packet)['success']
    packet = fixture(); packet['surface_gap'][20:181] = .08
    assert not episode_outcome(packet)['success']


def test_chooser_preserves_baseline_for_ties_and_small_gains():
    assert choose_candidate([.1, .1, .1]) == 0
    assert choose_candidate([.1, .109, .08]) == 0
    assert choose_candidate([.1, .12, .15]) == 2
    assert choose_candidate([.1, .15, .15]) == 1
    with pytest.raises(ValueError): choose_candidate([.1, np.nan, .1])
    for index in range(3):
        plan = candidate_plan(index)
        assert plan.shape == (24, 18)
        assert not plan[:, [7, 9, 11, 13, 16, 17]].any()
        assert np.abs(plan).max() <= .2 + 1e-7
    assert np.array_equal(candidate_plan(1), -candidate_plan(2))


def test_paired_report_counts_rescue_and_harm_separately():
    pairs = [dict(baseline=dict(success=False), rolling=dict(success=True)),
             dict(baseline=dict(success=True), rolling=dict(success=False)),
             dict(baseline=dict(success=True), rolling=dict(success=True))]
    result = paired_counts(pairs)
    assert result == dict(episodes=3, baseline_success=2, rolling_success=2, rescued=1, harmed=1)


def test_runner_import_does_not_initialize_torch_before_isaac():
    # Separate interpreter is the meaningful order check; pytest itself uses Torch.
    import subprocess
    code = "import sys,runpy;runpy.run_path(%r);assert 'torch' not in sys.modules" % str(TASK / 'tools/run/run_gate1_gt_progress.py')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_legacy_actor_batch_preserves_observation_and_only_first_control():
    import torch
    from consequence_evaluator.gate1 import legacy_batched_actor_action
    raw = torch.arange(1442, dtype=torch.float32)[None]
    original = raw.clone()
    class Player:
        is_rnn = False
        def get_action(self, observation, deterministic):
            assert deterministic is True
            assert observation['obs'].shape == (64, 1442)
            assert torch.equal(observation['obs'], original.repeat(64, 1))
            output = torch.arange(64 * 18, dtype=torch.float32).reshape(64, 18)
            return output
    control = legacy_batched_actor_action(Player(), {'obs': raw})
    assert control.shape == (1, 18)
    assert torch.equal(control, torch.arange(18, dtype=torch.float32)[None])
    assert torch.equal(raw, original)


def test_legacy_group_actor_batch_contract():
    import torch
    from consequence_evaluator.gate1 import legacy_group_actor_action
    raw = torch.arange(2 * 18, dtype=torch.float32).reshape(2, 18)
    class Player:
        is_rnn = False
        def get_action(self, observation, deterministic):
            assert deterministic is True
            assert observation['obs'].shape == (8, 18)
            return observation['obs']
    result = legacy_group_actor_action(Player(), {'obs': raw}, copies=4)
    assert result.shape == (2, 18)
    assert torch.equal(result, raw)


def test_host_backend_separates_gpu_physx_from_cpu_tensor_pipeline():
    backend = resolve_backend('host')
    assert backend.name == 'gpu_physx_cpu_pipeline'
    assert backend.sim_device == 'cuda:0'
    assert backend.pipeline == 'cpu'
    assert backend.physx_use_gpu is True
    assert backend.physx_num_threads == 1
    assert backend.tensor_device == 'cpu'
    assert backend.actor_device == 'cuda:0'
    assert backend.argv() == (
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'cpu',
        '--num_threads', '1')


def test_host_backend_alias_and_old_gpu_cpu_flag_are_unambiguous():
    assert resolve_backend('gpu_physx_cpu_pipeline') == resolve_backend('host')
    assert resolve_legacy_backend(None, 'gpu') == resolve_backend('gpu')
    assert resolve_legacy_backend(None, 'cpu') == resolve_backend('cpu')
    assert resolve_legacy_backend('host', 'host') == resolve_backend('host')
    with pytest.raises(ValueError, match='different contracts'):
        resolve_legacy_backend('host', 'gpu')


def test_backend_canonicalizes_isaac_cuda_shorthand():
    assert canonical_device('cuda') == 'cuda:0'
    assert canonical_device('cuda:2') == 'cuda:2'
    assert canonical_device('cpu') == 'cpu'
