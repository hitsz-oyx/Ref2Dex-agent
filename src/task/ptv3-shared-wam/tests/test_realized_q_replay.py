"""Oracle playback must preserve the action-to-next-state clock."""
import importlib.util
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / 'tools/run/execute_realized_q_replay.py'
spec = importlib.util.spec_from_file_location('realized_q_replay', path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_next_q_targets_and_same_tick_pd_have_different_clocks():
    data = dict(q=np.repeat(np.arange(20)[:, None], 18, axis=1),
                pd_targets=np.repeat((100 + np.arange(19))[:, None], 18, axis=1))
    np.testing.assert_array_equal(replay.reference_chunk(data, 3, 'q')[:, 0], np.arange(4, 12))
    np.testing.assert_array_equal(replay.reference_chunk(data, 3, 'PD')[:, 0], np.arange(103, 111))
    # A K4 refill must consume the same exact eight oracle states as K1.
    chunked = np.concatenate([replay.reference_chunk(data, t, 'q')[:4] for t in (3, 7)])
    np.testing.assert_array_equal(chunked, data['q'][4:12])
    # Only unconsumed suffixes are padded near the recorded terminal frame.
    np.testing.assert_array_equal(replay.reference_chunk(data, 18, 'q')[0], data['q'][19])


def test_damping_feedforward_uses_only_the_joint_plan_and_preserves_PD_control():
    data = dict(q=np.repeat((np.arange(20, dtype=np.float32)**2)[:, None], 18, axis=1),
                pd_targets=np.repeat((100 + np.arange(19))[:, None], 18, axis=1))
    expected = data['q'][4:12] + 3*(data['q'][4:12]-data['q'][3:11])
    np.testing.assert_array_equal(replay.reference_chunk(data, 3, 'q', True), expected)
    np.testing.assert_array_equal(replay.reference_chunk(data, 3, 'PD', True), data['pd_targets'][3:11])
