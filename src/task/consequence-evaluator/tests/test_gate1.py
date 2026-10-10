"""Independent recoverable outcome and frozen GT candidate selection."""
from pathlib import Path
import runpy
import sys
import numpy as np
import pytest
TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.gate1 import candidate_plan, choose_candidate, episode_outcome, paired_counts


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
