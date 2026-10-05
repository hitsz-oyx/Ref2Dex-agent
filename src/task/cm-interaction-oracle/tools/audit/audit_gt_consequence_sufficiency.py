#!/usr/bin/env python3
"""File/statistical replay of ref8 outputs; no neural inference or refitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
from consequence_sufficiency import HEADS, PRIMARY


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(dataset, run):
    p = torch.load(dataset, weights_only=False, map_location='cpu')
    d = torch.load(run / 'diagnostic.pt', weights_only=False, map_location='cpu')
    r = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    bridge = json.loads((run / 'arm_bridge.json').read_text())
    assert manifest['run_status'] == 'COMPLETED' and digest(dataset) == manifest['dataset_sha256']
    assert all(digest(ROOT / k) == v for k, v in manifest['code_sha256'].items())
    tr = p['trajectory'].numpy(); before = p['before'].numpy(); rest = p['rest_height'].numpy()
    n = len(tr); y = np.zeros((n, 8)); pair = tr[:, :, 71] > .5
    lift = tr[:, :, 2] - rest[:, None]
    lost = np.zeros(n, dtype=int); failure = np.zeros((n, 32), dtype=bool)
    for step in range(32):
        lost = np.where(pair[:, step], 0, lost + 1)
        failure[:, step] = (lift[:, step] < .02) | (lost >= 6)
    for h, offset in ((16, 0), (32, 3)):
        y[:, offset] = pair[:, 8:h].mean(1)
        y[:, offset+1] = ((lift[:, 8:h] >= .03) & pair[:, 8:h]).mean(1)
        y[:, offset+2] = failure[:, 8:h].any(1)
    y[:, 6] = (lift[:, 8:] < .02).any(1)
    y[:, 7] = (lift[:, 8:] >= .03).mean(1)
    target_error = float(np.max(np.abs(y - d['target']))); assert target_error < 1e-7
    force = tr[:, 7, 48:63].reshape(-1, 5, 3)
    i = np.column_stack((np.log1p(np.linalg.norm(force, axis=-1)), np.log1p(np.abs(tr[:, 7, 63:66])), tr[:, 7, 66:72]))
    signed = np.arcsinh(tr[:, 7, 48:66])
    mediator_error = max(float(np.max(np.abs(i - d['I'].numpy()))), float(np.max(np.abs(signed - d['signed'].numpy()))))
    assert mediator_error < 2e-6
    # Independent relative-quaternion construction for E12.
    q0 = before[:, 3:7].astype(float); q1 = tr[:, 7, 3:7].astype(float)
    q0 /= np.linalg.norm(q0, axis=1)[:, None]; q1 /= np.linalg.norm(q1, axis=1)[:, None]
    xyz = q0[:, 3:] * q1[:, :3] - q1[:, 3:] * q0[:, :3] - np.cross(q1[:, :3], q0[:, :3])
    w = (q0 * q1).sum(1); sign = np.where(w < 0, -1, 1); xyz *= sign[:, None]; w *= sign
    norm = np.linalg.norm(xyz, axis=1); angle = 2 * np.arctan2(norm, w)
    rotation = xyz * (angle / np.maximum(norm, 1e-8))[:, None]
    e = np.column_stack((tr[:, 7, :3] - before[:, :3], rotation, tr[:, 7, 7:13] - before[:, 7:13]))
    effect_error = float(np.max(np.abs(e - d['E'].numpy()))); assert effect_error < 2e-6
    train, test = d['train'], d['test']; clusters = d['clusters']
    assert not set(clusters[train]) & set(clusters[test]) and len(train) + len(test) == n
    mean, scale = d['normalizers']['y_mean'].numpy(), d['normalizers']['y_scale'].numpy()
    assert np.max(np.abs(mean - y[train].mean(0))) < 2e-7
    assert np.max(np.abs(scale - np.maximum(y[train].std(0), .001))) < 2e-7
    raw_h = torch.cat((p['history'].flatten(1), p['actor_obs'], p['context'], p['base_action']), -1).numpy()
    assert np.allclose(d['normalizers']['h_mean'].numpy(), raw_h[train].mean(0), atol=2e-5, rtol=2e-5)
    assert np.allclose(d['normalizers']['h_scale'].numpy(), np.maximum(raw_h[train].std(0), .001), atol=2e-5, rtol=2e-5)
    projection = d['normalizers']['projection'].numpy()
    orth_error = float(np.max(np.abs(projection.astype(float).T @ projection.astype(float) - np.eye(32))))
    # Long float32 GPU SVD vectors need a dimension-aware numerical envelope;
    # projected features are subsequently train-standardized in every model.
    orth_tolerance = 4 * np.finfo(projection.dtype).eps * projection.shape[0]
    assert orth_error < orth_tolerance
    normalized = d['normalized_target']; errors = {}; max_metric_error = 0.
    for name, pred in d['predictions'].items():
        error = (((pred - mean) / scale - normalized) ** 2)[test]
        errors[name] = error
        max_metric_error = max(max_metric_error, float(np.max(np.abs(error.mean(0) - r['models'][name]['test_normalized_mse_per_head']))))
        assert sum(v.numel() for v in d['models'][name].values()) == 12776
    assert max_metric_error < 2e-6 and len(set(d['initial_hashes'].values())) == 1
    max_bootstrap_error = 0.
    for name, comparison in r['comparisons'].items():
        new, base = name.split('_vs_')
        for metric, axes in (('primary', PRIMARY), ('physical_failure', (6,)), ('contact', (3,))):
            base_err, new_err = errors[base][:, axes].mean(1), errors[new][:, axes].mean(1)
            unique = np.unique(clusters[test])
            a = np.array([base_err[clusters[test] == c].sum() for c in unique])
            b = np.array([new_err[clusters[test] == c].sum() for c in unique])
            draws = np.random.default_rng(232).integers(len(unique), size=(2000, len(unique)))
            gain = 1 - b[draws].sum(1) / a[draws].sum(1)
            recalculated = [1 - new_err.mean() / base_err.mean(), *np.quantile(gain, [.025, .975, .95])]
            stored = [comparison[metric][k] for k in ('gain', 'lower95', 'upper95', 'upper_one_sided95')]
            max_bootstrap_error = max(max_bootstrap_error, float(np.max(np.abs(np.array(stored) - recalculated))))
    assert max_bootstrap_error < 2e-6
    # Full design OLS with explicit categorical columns, independent of FWL replay.
    joint = np.column_stack((d['E'].numpy(), d['I'].numpy(), d['signed'].numpy(), y))
    label = d['arms']; onehot = (label[:, None] == np.arange(1, 15)[None]).astype(float)
    coefficients = np.linalg.lstsq(np.column_stack((d['design'], onehot)), joint, rcond=1e-10)[0][-14:]
    ols_error = float(np.max(np.abs(coefficients - d['full_contrasts']))); assert ols_error < 1e-8
    bridge_error = 0.
    for name, axes in dict(E=slice(0,12), I=slice(12,26), EI=slice(0,26), EI_signed=slice(0,44)).items():
        pred = []
        for source, destination in ((0,1),(1,0)):
            a, b = d['half_contrasts'][source], d['half_contrasts'][destination]
            for j in range(14):
                keep = np.arange(14) != j; x = a[keep, axes]; sy = np.maximum(x.std(0), .001)
                x = x / sy; _, _, vt = np.linalg.svd(x, full_matrices=False); z = x @ vt[:3].T
                # Ridge by augmented least squares, distinct from normal-equation implementation.
                w = np.linalg.lstsq(np.vstack((z, np.eye(3))), np.vstack((a[keep,44:], np.zeros((3,8)))), rcond=None)[0]
                pred.append(b[j,axes] / sy @ vt[:3].T @ w)
        bridge_error = max(bridge_error, float(np.max(np.abs(np.asarray(pred) - bridge[name]['predictions']))))
    assert bridge_error < 1e-10
    assert r['GT_prognosis_status'] == 'PROMISING' and r['status'] == 'UNCLEAR'
    assert not r['small_remaining_action'] and r['arm_bridge_pass']
    return dict(status='PASS', target_max_error=target_error, I_signed_max_error=mediator_error,
        E_max_error=effect_error, projection_orthogonality_error=orth_error,
        projection_orthogonality_tolerance=orth_tolerance,
        metrics_max_error=max_metric_error, bootstrap_max_error=max_bootstrap_error,
        direct_OLS_max_error=ols_error, independent_ridge_bridge_max_error=bridge_error,
        dataset_sha256=digest(dataset), source_commit=manifest['git_commit'],
        diagnostic_sha256=digest(run/'diagnostic.pt'), audit_code_sha256=digest(__file__),
        limits='Replay of fixed saved predictions and estimated contrasts; no causal mediation or fit-seed uncertainty claim')


def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((run/'result.json').read_text()); bridge=json.loads((run/'arm_bridge.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    names = ('H','Ha','HE','HI','HEI','HaEI','HEI_signed','HaEI_signed')
    axes[0].barh(names, [r['models'][n]['test_primary_mse'] for n in names])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Held-out normalized primary error (lower better)')
    axes[0].set_title('GT prognosis: fixed environment holdout')
    c = r['comparisons']['HaEI_vs_HEI']; labels=('primary','physical_failure')
    point=np.array([c[n]['gain'] for n in labels])*100
    low=np.array([c[n]['lower95'] for n in labels])*100; high=np.array([c[n]['upper95'] for n in labels])*100
    axes[1].errorbar(point, np.arange(2), xerr=(point-low,high-point), fmt='o', capsize=4)
    axes[1].set_yticks(np.arange(2), labels); axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].axvline(5, color='grey', linestyle='--'); axes[1].set_xlabel('Adding a to GT E/I: error gain (%)')
    axes[1].set_title('95% environment-bootstrap intervals')
    for axis, name in ((3,'contact32'),(6,'physical height failure')):
        pred=np.asarray(bridge['EI']['predictions'])[:,axis]; y=np.asarray(bridge['EI']['target'])[:,axis]
        axes[2].scatter(y*100,pred*100,label=name,alpha=.7)
    axes[2].plot([-30,30],[-30,30], color='grey',linestyle='--')
    axes[2].set_xlabel('Other-half measured arm contrast (pp)'); axes[2].set_ylabel('Predicted contrast (pp)')
    axes[2].set_title('Cross-half, held-arm EI bridge (descriptive)'); axes[2].legend(fontsize=8)
    fig.suptitle('ref8: GT prognosis PROMISING; full consequence sufficiency UNCLEAR')
    fig.tight_layout(); fig.savefig(run/'gt_consequence_sufficiency.png', dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True); parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    output=args.run_dir/'statistical_replay.json'
    if output.exists(): raise FileExistsError(output)
    result=audit(args.dataset,args.run_dir)
    output.write_text(json.dumps(result,indent=2)+'\n'); plot(args.run_dir)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
