import numpy as np
import torch

from wam_rl_policy_v3.action_contract import canonical_action, step_with_isolated_action
from wam_rl_policy_v3.hold_metrics import summarize_realtime_hold


def test_simulator_cannot_rewrite_the_recorded_canonical_action():
    """Mirror UniDexGrasp pre_physics_step's in-place wrist transform."""
    action = torch.tensor([[1., 0., 0.] + [0.] * 21])
    canonical = canonical_action(action)

    def mutating_sim_step(received):
        # Equivalent to actions[:,0:3] = pose_vec(actions[:,0:3]) at yaw=90deg.
        received[:, 0:3] = torch.tensor([[0., 1., 0.]])
        received[:, 3:6] = torch.tensor([[0., 1., 0.]])

    step_with_isolated_action(mutating_sim_step, action)
    assert torch.equal(canonical[:, :6], torch.tensor([[1., 0., 0., 0., 0., 0.]]))
    assert torch.equal(action[:, :6], torch.tensor([[1., 0., 0., 0., 0., 0.]]))


def test_realtime_hold_is_not_the_cumulative_upstream_flag():
    distances = [.10, .01, .15, .20]
    summary = summarize_realtime_hold(distances)
    assert summary["success_frames"] == 1
    assert summary["longest_hold_frames"] == 1
    assert summary["final_success"] is False
    assert summary["ever_success"] is True
    assert summary["first_success_index"] == 1
    # The old upstream aggregate would have counted the three later frames.
    cumulative = np.asarray(distances) <= .05
    cumulative = np.maximum.accumulate(cumulative.astype(int))
    assert int(cumulative.sum()) == 3


def test_realtime_hold_counts_contiguous_runs_and_drops():
    summary = summarize_realtime_hold([.2, .04, .03, .02, .1, .01, .0], .05)
    assert summary["success_frames"] == 5
    assert summary["longest_hold_frames"] == 3
    assert summary["success_onsets"] == 2
    assert summary["drop_events_after_success"] == 1
    assert summary["final_success"] is True
