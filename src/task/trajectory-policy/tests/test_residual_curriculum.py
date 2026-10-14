import numpy as np

from trajectory_policy.residual_curriculum import (
    curriculum_phases, diagnostic_phases, query_budget, contact_hold_summary,
)


def test_contact_curriculum_keeps_startup_and_emphasizes_not_yet_held_contact():
    assert curriculum_phases('uniform', 4) == (0, 32, 48, 62)
    assert curriculum_phases('contact', 4) == (0, 48, 50, 52)
    assert diagnostic_phases(curriculum_phases('contact', 4)) == (0, 32, 48, 50, 52, 62)


def test_query_budget_includes_bootstrap_and_every_real_evaluation():
    assert query_budget(20, 4, (0, 32, 48, 62)) == 2578
    assert query_budget(40, 4, (0, 32, 48, 62), 20) == 96 + 40*97 + 2*542
    assert query_budget(40, 4, (0, 48, 50, 52), 20) == 192 + 40*97 + 2*542


def test_new_hold_metric_separates_contact_acquisition_from_inherited_hold():
    steps, envs = 50, 3
    obj = np.broadcast_to(np.eye(4), (steps + 1, envs, 4, 4)).copy()
    obj[:, :, 2, 3] = .84
    obj[0, 0, 2, 3] = .80
    gap = np.full((steps + 1, envs), .001)
    gap[30:, 2] = .2
    packed = dict(obj=obj, surface_gap=gap, support_gap=np.ones_like(gap),
                  table_footprint=np.zeros_like(gap, dtype=bool))
    rows = contact_hold_summary(packed, [52, 62, 52], [.8]*envs)
    assert rows[0]['new_hold45'] and rows[0]['maximum_held_frames'] == 50
    assert rows[1]['initially_held'] and not rows[1]['new_hold45']
    assert rows[2]['maximum_held_frames'] == 29 and not rows[2]['new_hold45']
