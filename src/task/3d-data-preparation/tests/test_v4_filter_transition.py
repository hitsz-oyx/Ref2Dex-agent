import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/audit'
sys.path.insert(0, str(TOOLS))
import rebuild_v4_window_filter_transition as transition
from pose_mesh_data.v4_interaction import V4SamplingPolicy


def test_transition_rechecks_only_old_blocks_and_can_restore_moving_objects(monkeypatch):
    policy = V4SamplingPolicy()
    monkeypatch.setattr(transition, 'retain_window',
                        lambda kind, identity, policy: kind == 'moving_object_noncontact')
    restored, counts = transition.transition_blocked(
        [3, 8, 13], lambda index: (('moving_object_noncontact', str(index))
                                   if index == 8 else ('static_idle', str(index))), policy)
    assert restored == [8]
    assert counts == {'static_idle': 2, 'moving_object_noncontact': 1}


def test_transition_requires_a_real_completed_filter():
    # The CLI must not silently accept a probe output as a training filter.
    with pytest.raises(SystemExit):
        sys.argv = ['probe', '--main3-root', 'x', '--native-root', 'y', '--split', 'train',
                    '--old-filter', 'z', '--probe-limit', '1', '--output', 'o']
        transition.main()
