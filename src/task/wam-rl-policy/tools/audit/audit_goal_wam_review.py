"""Independently check a real WAM review cache against original observations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'inference-manifest.json').read_text())
    catalog = json.loads((directory/'predictions.json').read_text())
    source = Path(manifest.get('input_lineage', {}).get('original_source', manifest['source']))
    original = (source/'last-episode.npz').exists()
    if not original:
        source = Path(manifest['source'])
    frames = predictions = 0
    largest_delta_error = largest_tau_error = 0.
    with np.load(source/'last-episode.npz', allow_pickle=False) as recorded:
        for case in catalog['cases']:
            env = case['identity'].get('original_env', case['identity']['env']) if original else case['identity']['env']
            with np.load(directory/case['file'], allow_pickle=False) as data:
                for name in data.files:
                    assert np.isfinite(data[name]).all(), (case['name'], name)
                tick = data['tick']
                frames += len(tick)
                predictions += len(tick)*3
                np.testing.assert_array_equal(data['hand'], recorded['hand'][tick, env])
                np.testing.assert_allclose(data['object_pose'][:, :3, 3], recorded['object'][tick, env, :3])
                np.testing.assert_allclose(data['object_pose'][:, :3, :3],
                    Rotation.from_quat(recorded['object'][tick, env, 3:7]).as_matrix())
                np.testing.assert_allclose(data['predicted_hand'][:, :, 0],
                    np.broadcast_to(data['hand'][:, None], data['predicted_hand'][:, :, 0].shape))
                for f, t in enumerate(tick):
                    pose = data['object_pose'][f]
                    current = data['flow'] @ pose[:3, :3].T+pose[:3, 3]
                    for variant in range(3):
                        goal = data['goals'][f, variant]
                        future = data['flow'] @ goal[:3, :3].T+goal[:3, 3]
                        delta = (future-current) @ pose[:3, :3]
                        error = float(np.max(np.abs(delta-data['goal_delta'][f, variant])))
                        largest_delta_error = max(largest_delta_error, error)
                        assert error < 2e-7
                        assert np.linalg.norm(goal[:3, 3]-pose[:3, 3]) <= .050001
                        assert Rotation.from_matrix(pose[:3, :3].T @ goal[:3, :3]).magnitude() <= .350001
                        assert np.linalg.norm(future-current, axis=-1).mean() <= .050001
                        np.testing.assert_allclose(data['predicted_scene'][f, variant, 0], current, atol=1e-7)
                    indices = int(t)+np.arange(25)*2
                    valid = indices < len(recorded['time'])
                    np.testing.assert_array_equal(data['observed_future_valid'][f], valid)
                    np.testing.assert_array_equal(data['observed_hand'][f, valid], recorded['hand'][indices[valid], env])
                # The native executor consumes first 8 ticks at60Hz, not the
                # first 8 of24 WAM rows at30Hz. Check in world positions.
                at = np.arange(1, 9)/2
                fraction = (at % 1)[None, None, :, None, None]
                expected = (data['predicted_hand'][:, :, np.floor(at).astype(int)]*(1-fraction)
                    + data['predicted_hand'][:, :, np.ceil(at).astype(int)]*fraction
                    - data['hand'][:, None, None])
                tau_error = float(np.max(np.abs(expected-data['student_tau'])))
                largest_tau_error = max(largest_tau_error, tau_error)
                assert tau_error < 2e-7
                assert data['contact'].min() >= 0 and data['contact'].max() <= 1
                assert data['parts'].min() >= 0 and data['parts'].max() <= 1
                valid = data['interaction_valid'].astype(bool)
                assert np.max(np.abs(data['parts'].sum(-1)[valid]-1)) < .01
    report = dict(status='PASS', cases=len(catalog['cases']), frames=frames, predictions=predictions,
        max_goal_anchor_delta_error_m=largest_delta_error, max_native_tau_error_m=largest_tau_error,
        scope='observed-state identity, SE3 budgets, t0, world coordinates, native clock, future masks and C probabilities')
    (directory/'data-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().directory)))
