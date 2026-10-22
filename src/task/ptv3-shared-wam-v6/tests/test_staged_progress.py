"""Adversarial evidence injections exercise the same observer used online."""
from types import SimpleNamespace
import numpy as np
import pytest
from ptv3_shared_wam_v6.staged_progress import StagedProgress, StagedDenseReward, STAGES


def sample(stages, tick, *, z=0., contact=True, valid=True, coupled=True,
           released=False, authorized=False, lost=False, abort=False):
    features = SimpleNamespace(palm_gap_m=.02, mean_tip_gap_m=.01,
                               hand_clearance_m=.04 if released else .01)
    feedback = SimpleNamespace(force_valid=valid, workspace_ok=True,
        contact_observed=contact, grasp_established=coupled, grasp_lost=lost,
        supported=False, released=released)
    forces = np.ones((5, 3)) if contact else np.zeros((5, 3))
    return stages.update(z, tick/30, features, feedback, forces,
                         release_authorized=authorized, physical_abort=abort)


def hold(stages):
    sample(stages, 0)
    for tick in range(1, 32):
        sample(stages, tick, z=.05)


def test_lift_without_contact_is_rejected():
    stages = StagedProgress(0.)
    for tick in range(100):
        result = sample(stages, tick, z=.1, contact=False)
        assert not stages.progress[1:].any()
    assert not any(result['achieved'][k] for k in STAGES[1:])


def test_contact_without_lift_earns_no_lift_hold_release():
    stages = StagedProgress(0.)
    for tick in range(100):
        result = sample(stages, tick)
    assert result['achieved']['contact']
    assert not any(result['achieved'][k] for k in STAGES[2:])
    assert not stages.progress[2:].any()


def test_intermittent_hold_does_not_accumulate_to_one_second():
    stages = StagedProgress(0.)
    sample(stages, 0)
    for tick in range(1, 100):
        result = sample(stages, tick, z=.05, valid=tick % 15 != 0)
    assert result['achieved']['lift'] and not result['achieved']['hold']
    assert result['longest_hold_s'] < .5


def test_lift_before_first_contact_cannot_be_credited_afterwards():
    stages = StagedProgress(0.)
    sample(stages, 0, z=.1, contact=False)
    for tick in range(1, 50):
        result = sample(stages, tick, z=.1)
    assert result['achieved']['contact'] and not result['achieved']['lift']


@pytest.mark.parametrize('bad', ['force_invalid', 'attached', 'unauthorized', 'abort'])
def test_fake_release_is_rejected(bad):
    stages = StagedProgress(0.)
    hold(stages)
    for tick in range(32, 50):
        result = sample(stages, tick, z=.05, released=True,
            contact=bad == 'attached', coupled=bad == 'attached',
            valid=bad != 'force_invalid', authorized=bad != 'unauthorized', abort=bad == 'abort')
    assert not result['achieved']['release'] and stages.progress[-1] == 0


def test_positive_sequence_uses_elapsed_time_and_real_detachment():
    stages = StagedProgress(0.)
    sample(stages, 0)
    for tick in range(1, 31):
        assert not sample(stages, tick, z=.05)['achieved']['hold']
    result = sample(stages, 31, z=.05)
    assert result['first_time_s']['lift'] == 1/30
    assert result['first_time_s']['hold'] == 31/30
    for tick in range(32, 37):
        result = sample(stages, tick, z=.05, contact=False, coupled=False,
                        released=True, authorized=True)
    assert all(result['achieved'].values())
    assert result['current']['release'] and not result['current']['hold']


def test_missing_sample_breaks_continuity_and_nonmonotone_time_is_rejected():
    stages = StagedProgress(0.)
    sample(stages, 0)
    sample(stages, 1, z=.05)
    result = sample(stages, 100, z=.05)
    assert not result['achieved']['hold'] and result['continuous_hold_s'] == 0
    with pytest.raises(ValueError, match='increasing'):
        sample(stages, 100, z=.05)


def test_repeated_partial_progress_has_bounded_payment_and_terminal_is_once():
    stages = StagedProgress(0.)
    sample(stages, 0, contact=False)
    reward = StagedDenseReward(stages)
    decision = SimpleNamespace(failure_reason=None, task_complete=False, terminated=False)
    for tick in range(1, 151):
        sample(stages, tick, z=.05 if tick > 1 and tick % 15 else 0., contact=tick % 15 != 0)
        total, parts, increments = reward.step(stages, decision, np.zeros(12))
        assert total == pytest.approx(sum(parts.values()))
        assert parts['shaping'] == pytest.approx(sum(increments.values()))
    assert reward.totals['shaping'] <= 1.
    assert reward.totals['event'] == pytest.approx(.2)  # contact + lift only
    assert not stages.snapshot()['achieved']['hold']
    _, parts, _ = reward.step(stages, decision, np.zeros(12), failed=True)
    assert parts['sparse'] == -1.
    with pytest.raises(ValueError, match='live episode'):
        reward.step(stages, decision, np.zeros(12))


def test_fixed_demo_panel_counts_and_time_distributions():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'tools/audit/evaluate_staged_progress.py'
    spec = importlib.util.spec_from_file_location('staged_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.demo()
    summary = module.summarize(records)
    assert [summary[k]['successes'] for k in STAGES] == [6, 4, 3, 2, 1]
    assert all(r['observed_controls'] == 150 for r in records)
    assert summary['release']['first_passage_s']['n'] == 1
    assert summary['hold']['from_previous_stage_s']['median'] == pytest.approx(1.)
    assert all(sum(r['reward_parts_sum'][k] for k in ('shaping','event')) <= 1.5 for r in records)
