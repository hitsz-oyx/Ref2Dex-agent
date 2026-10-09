import importlib.util
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "structured_candidate_bank_audit",
    Path(__file__).resolve().parents[1] / "tools/audit/audit_structured_candidate_bank.py",
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def test_pairwise_rms_and_current_object_frame_transform_are_batchwise():
    value = np.asarray([[0., 0.], [3., 4.]], dtype="float32")
    np.testing.assert_allclose(_audit.pairwise_rms(value), [[0., np.sqrt(12.5)], [np.sqrt(12.5), 0.]])

    pose = np.tile(np.eye(4, dtype="float32"), (26, 2, 1, 1))
    pose[:, 1, :3, 3] = [1., 2., 3.]
    hand = np.zeros((26, 2, 11, 3), dtype="float32")
    hand[1:25, 0, :, 0] = np.arange(1., 25.)[:, None]
    hand[1:25, 1, :, 0] = np.arange(1., 25.)[:, None] + 1.
    future = _audit.current_frame_hand_future(pose, hand, 0)
    assert future.shape == (24, 2, 11, 3)
    expected = np.broadcast_to(np.arange(1., 25.)[:, None], (24, 11))
    np.testing.assert_allclose(future[:, 0, :, 0], expected)
    np.testing.assert_allclose(future[:, 1, :, 0], expected)


def test_reset_report_detects_bitwise_state_contract():
    data = {
        "object_pose": np.tile(np.eye(4, dtype="float32"), (3, 2, 1, 1)),
        "hand_keypoints": np.zeros((3, 2, 11, 3), dtype="float32"),
        "dof_position": np.zeros((3, 2, 18), dtype="float32"),
        "dof_velocity": np.zeros((3, 2, 18), dtype="float32"),
        "pair": np.zeros((3, 2), dtype=bool),
    }
    assert _audit.reset_report(data)["exact"]
    data["dof_position"][0, 1, 4] = 1e-3
    report = _audit.reset_report(data)
    assert not report["exact"] and not report["dof_position"]["bitwise_equal"]
