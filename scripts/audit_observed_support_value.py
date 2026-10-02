"""Independent causal observations, SDK transforms, full NN and paired statistics."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.audit_truth_successor_value import decode
from src.task.CmResidual.observed_support_value import read_panel


def matrices(quaternion):
    q = torch.as_tensor(quaternion, dtype=torch.float64)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), -1).reshape(q.shape[:-1] + (3, 3))


def independent_panel(directory, base):
    raw = decode(directory, base)
    mask = raw['tick'] >= 2
    data = {k: v[mask] for k, v in raw.items()}
    trace = torch.load(directory / 'trace.pt', map_location='cpu', weights_only=False)
    metadata = json.loads((directory / 'physical_metadata.json').read_text())
    tick, env = data['tick'], data['environment']
    current = trace['object_root'][tick - 1, env].double()
    older = trace['object_root'][tick - 2, env].double()
    inverse = matrices(current[:, 3:7]).transpose(-1, -2)
    older_inverse = matrices(older[:, 3:7]).transpose(-1, -2)
    position = torch.matmul(inverse[:, None],
                            (trace['hand_body_position'][tick - 1, env].double() - current[:, None, :3])[..., None]).squeeze(-1)
    prior_position = torch.matmul(older_inverse[:, None],
                                  (trace['hand_body_position'][tick - 2, env].double() - older[:, None, :3])[..., None]).squeeze(-1)
    orient = torch.matmul(inverse[:, None], matrices(trace['hand_body_quaternion'][tick - 1, env]))
    force = torch.cat((trace['object_force'][tick - 1, env, None], trace['hand_force'][tick - 1, env]), 1).double()
    weight = torch.tensor([p['mass'] for p in metadata['object_body_properties']], dtype=torch.float64)
    weight *= torch.linalg.vector_norm(torch.tensor(metadata['gravity'], dtype=torch.float64))
    force = torch.matmul(inverse[:, None], force[..., None]).squeeze(-1) / weight[env, None, None]
    # Saved clearance differences were float32 in the primary decoder.
    clearance_delta = (trace['clearance'][tick - 1, env] - trace['clearance'][tick - 2, env]).double()
    increments = torch.stack((current[:, 2] - older[:, 2], clearance_delta), -1) / .005
    data['extra'] = torch.cat((position.reshape(-1, 15), orient[..., :2].reshape(-1, 30),
                              force.reshape(-1, 18), increments,
                              (position - prior_position).reshape(-1, 15)), -1).float().numpy()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    root = args.directory.resolve()
    manifest = json.loads((root / 'run_manifest.json').read_text())
    report = json.loads((root / 'fit/results.json').read_text())
    assert report['run_status'] == 'COMPLETED'
    assert sha(root / 'fit/models.pt') == report['model_sha256']
    assert sha(root / 'fit/predictions.pt') == report['prediction_sha256']
    models = torch.load(root / 'fit/models.pt', map_location='cpu', weights_only=False)
    saved = torch.load(root / 'fit/predictions.pt', map_location='cpu', weights_only=False)
    base = torch.load(manifest['base_checkpoint'], map_location='cpu', weights_only=False)
    fit = independent_panel(root / 's597', base)
    test = independent_panel(root / 's598', base)
    maxima = {}
    # Independently audit both cohorts; exact FIT statistics use the same saved
    # float32 observations after their separate numerical reconstruction check.
    fit_primary, test_primary = read_panel(root / 's597'), read_panel(root / 's598')
    for name, independent, primary in [('fit', fit, fit_primary), ('test', test, test_primary)]:
        for key in ('target', 'tick', 'environment', 'cluster'):
            assert np.array_equal(independent[key], primary[key].numpy()), (name, key)
        maxima[name + '_sdk_extra'] = float(np.abs(independent['extra'] - primary['extra'].numpy()).max())
        maxima[name + '_compact'] = float(np.abs(independent['current'] - primary['current'].numpy()).max())
        assert maxima[name + '_sdk_extra'] <= 2e-5
        assert maxima[name + '_compact'] <= 2e-6
    for key in ('target', 'tick', 'environment', 'cluster'):
        assert np.array_equal(test[key], saved['rows'][key].numpy())
    assert np.array_equal(saved['extra'].numpy(), test_primary['extra'].numpy())
    assert np.array_equal(saved['current'].numpy(), test_primary['current'].numpy())
    mean = fit_primary['extra'].numpy().mean(0)
    std = fit_primary['extra'].numpy().std(0).clip(.001)
    assert np.array_equal(mean, models['extra_mean']) and np.array_equal(std, models['extra_std'])
    table = np.full((3, 16), fit['target'].mean(), dtype=np.float32)
    for motion in range(3):
        for timebin in range(16):
            take = (fit['motion'] == motion) & (fit['timebin'] == timebin)
            if take.any():
                table[motion, timebin] = fit['target'][take].mean()
    assert np.array_equal(table, models['phase_template'])
    for variant, state in models['parameters'].items():
        extra = np.clip((saved['extra'].numpy() - mean) / std, -10, 10)
        if variant == 'compact_value':
            extra = np.zeros_like(extra)
        z = np.concatenate((saved['current'].numpy(), extra), -1).astype(np.float64)
        for layer in ('0', '2'):
            z = np.maximum(z @ state[layer + '.weight'].numpy().astype(np.float64).T + state[layer + '.bias'].numpy(), 0)
        z = z @ state['4.weight'].numpy().astype(np.float64).T + state['4.bias'].numpy()
        prediction = (1 / (1 + np.exp(-z.clip(-700, 700)))).ravel()
        maxima[variant + '_forward'] = float(np.abs(prediction - saved['predictions'][variant]).max())
        assert maxima[variant + '_forward'] <= 2e-5
    assert np.array_equal(table[test['motion'], test['timebin']], saved['predictions']['motion_time'])
    risk = {k: (v.astype(np.float64) - test['target'].astype(np.float64)) ** 2 for k, v in saved['predictions'].items()}
    brier = {k: float(v.mean()) for k, v in risk.items()}
    groups = test['cluster']
    counts = np.bincount(groups, minlength=192)
    indices = np.random.default_rng(3543).integers(0, 192, size=(2000, 192))
    gates = {}
    for control in ('compact_value', 'motion_time'):
        diff = np.bincount(groups, weights=risk['support_value'] - risk[control], minlength=192)
        ci = np.quantile(diff[indices].sum(1) / counts[indices].sum(1), [.025, .975]).tolist()
        assert np.allclose(ci, report['paired_difference_interval95'][control], atol=1e-12, rtol=0)
        gates[control] = dict(gain1percent=brier['support_value'] <= .99 * brier[control], paired_upper95_negative=ci[1] < 0)
    assert gates == report['gates']
    assert all(abs(brier[k] - report['brier'][k]) <= 1e-12 for k in brier)
    label = 'PROMISING' if all(all(g.values()) for g in gates.values()) else 'UNPROMISING'
    assert label == report['label']
    result = dict(run_status='COMPLETED', label=label, independent_reconstruction_maximum=maxima,
                  both_cohorts_current_sdk_geometry_force_flow_and_task_history_rebuilt=True,
                  full_test_numpy_model_replay=True, fit_statistics_shared_float32_decoder=True,
                  all_controls_cluster_statistics_and_gates_rebuilt=True, optimizer_not_replayed=True,
                  current_observed_information_only=True, source_result_sha256=sha(root / 'fit/results.json'),
                  wall_seconds=time.monotonic() - start)
    (root / 'value_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
