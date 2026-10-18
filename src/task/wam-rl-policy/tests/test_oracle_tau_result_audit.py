import importlib.util
import json
from pathlib import Path

import numpy as np


def test_archive_audit_recomputes_latched_success_and_marks_missing_terminal(tmp_path):
    tool = Path(__file__).resolve().parents[1] / 'tools/audit/audit_oracle_tau_results.py'
    spec = importlib.util.spec_from_file_location('oracle_result_audit', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        seed=453, variant='p2', policy='fixture.pt', success_frames=3)))
    obs = np.zeros((4, 300), np.float32)
    obs[:, 204] = [.10, .01, .15, .20]
    action = np.zeros((4, 24), np.float32)
    np.savez(tmp_path / 'trajectory.npz', obs=obs, action=action,
             action_prefix=action, replay_error=np.zeros(4),
             object_pose=np.zeros((4, 13)))
    result = module.audit_run(tmp_path)
    assert result['upstream_reported_success_frames'] == 3
    assert result['last_record_hold']['success_frames'] == 1
    assert result['last_record_hold']['longest_hold_frames'] == 1
    assert not result['last_record_hold']['final_success']
    assert result['terminal_after_last_action'] is None


def test_teacher_endpoint_uses_world_frame_and_reads_both_packet_generations(tmp_path):
    tool = Path(__file__).resolve().parents[1] / 'tools/audit/audit_oracle_tau_results.py'
    spec = importlib.util.spec_from_file_location('oracle_endpoint_audit', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('z', 90, degrees=True)
    current = np.zeros((2, 13))
    current[:, :3] = [.1, .2, .3]
    current[:, 3:7] = rotation.as_quat()
    goal = np.array([.1, .21, .32])
    obs = np.zeros((2, 300))
    obs[:, 191:194] = rotation.inv().apply(current[:, :3])
    obs[:, 194:198] = [0., 0., 0., 1.]
    obs[:, 204:207] = rotation.inv().apply(goal - current[:, :3])
    for tick in range(2):
        directory = tmp_path / f'oracle/tick-{tick:03d}'
        directory.mkdir(parents=True)
        future = current[tick].copy()
        future[:3] = goal + np.array([tick * .1, 0, 0])
        values = {'object_pose': future} if tick == 0 else {
            'object_pose': current[tick], 'future_object_pose': future}
        np.savez(directory / 'tau.npz', **values)
    result = module.teacher_hold_endpoints(tmp_path, obs, current)
    assert result['in_goal_queries'] == 2
    assert result['teacher_in_goal_after8'] == 1
    np.testing.assert_allclose(result['teacher_endpoint_distances_m'], [0., .1], atol=1e-7)
