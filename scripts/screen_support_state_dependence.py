#!/usr/bin/env python3
"""Fixed posthoc positional policy headroom; statistics only, no model fitting."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gradient', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, gradient, out = [p.resolve() for p in (args.source, args.gradient, args.output)]
    if out.exists() or ROOT not in out.parents:
        raise ValueError('unique isolated output required')
    files = [source/'dataset.pt', source/'results.json', source/'collection_audit.json',
             gradient/'analysis.pt', gradient/'results.json', gradient/'run_manifest.json',
             Path(__file__), ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-support-state-dependence.md']
    hashes = {str(f): sha(f) for f in files}
    out.mkdir()
    begin = time.monotonic()
    manifest = dict(run_status='RUNNING', input_sha256=hashes, reused_data=True,
                    device_reason='CPU counts and independent NumPy gradient statistics; no model inference',
                    wall_limit_seconds=60, storage_limit_bytes=5<<20)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        d = torch.load(source/'dataset.pt', map_location='cpu', weights_only=False)
        old = json.loads((source/'results.json').read_text())
        if sha(source/'dataset.pt') != old['dataset_sha256']:
            raise ValueError('audited data hash')
        grad_manifest = json.loads((gradient/'run_manifest.json').read_text())
        if grad_manifest['run_status'] != 'COMPLETED':
            raise ValueError('gradient run terminal')
        if any(sha(Path(f)) != h for f,h in grad_manifest['input_sha256'].items()):
            raise ValueError('gradient protected input drift')
        ga = torch.load(gradient/'analysis.pt', map_location='cpu', weights_only=False)
        gr = json.loads((gradient/'results.json').read_text())
        # Independent NumPy reductions over all 520 recorded coordinates.
        maximum_error = 0.
        gs = ga['test_seed'].numpy()
        gm = ga['test_motion'].numpy()
        masks = [('pooled', np.ones(len(gs), dtype=bool))]
        masks += [('seed'+str(s), gs == s) for s in (525, 526)]
        masks += [('motion'+str(m), gm == m) for m in range(3)]
        for name, mask in masks:
            expected = gr['pooled'] if name == 'pooled' else gr['seeds'][name[4:]] if name.startswith('seed') else gr['motions'][name[6:]]
            arrays = {k: v.numpy()[mask] for k,v in ga['gradients'].items()}
            moments = {k: float(np.einsum('nij,nij->n', v,v).mean()) for k,v in arrays.items()}
            for k in moments:
                error = abs(moments[k]-expected['second_moment'][k])
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    raise ValueError('independent moment reduction differs')
            for k in ('state_only', 'global_motion_arm'):
                value = float(np.linalg.norm(arrays['cm'].mean(0)-arrays[k].mean(0)))
                if abs(value-expected['finite_sample_mean_gradient_difference_norm'][k]) > 1e-12:
                    raise ValueError('mean-gradient norm differs')
        audit = dict(run_status='COMPLETED', gradient_run_status='COMPLETED',
                     protected_input_count=len(grad_manifest['input_sha256']),
                     maximum_second_moment_error=maximum_error, all_520_coordinates=True,
                     all1536_rows=True, method='independent NumPy einsum and mean-gradient norms',
                     gradient_label_unchanged=gr['label'])
        (out/'gradient_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        x, motion, arm, seed = [d[k].numpy() for k in ('state','motion','arm','seed')]
        reward = d['physical105'].numpy().astype(np.float64)
        fit, test = seed < 525, seed >= 525
        if fit.sum() != 4608 or test.sum() != 1536:
            raise ValueError('complete cohorts')
        chosen = {k: np.zeros(len(seed), dtype=np.int64) for k in ('position','global','unchanged')}
        policies = {}
        for mo in range(3):
            f = fit & (motion == mo)
            med = np.median(x[f,33:35],axis=0)
            bins = (x[:,33] >= med[0]).astype(int)+2*(x[:,34] >= med[1]).astype(int)
            count = np.zeros((4,8),dtype=int)
            positive = np.zeros_like(count)
            for b in range(4):
                for a in range(8):
                    ids = f & (bins == b) & (arm == a)
                    count[b,a] = ids.sum()
                    positive[b,a] = reward[ids].sum()
            probability = (positive+1)/(count+2)
            best = probability.argmax(1)
            total = count.sum(0)
            global_probability = (positive.sum(0)+1)/(total+2)
            global_best = int(global_probability.argmax())
            chosen['position'][motion == mo] = best[bins[motion == mo]]
            chosen['global'][motion == mo] = global_best
            policies[str(mo)] = dict(fit_median_xy=med.tolist(), fit_count=count.tolist(),
                fit_success=positive.tolist(), bin_arms=best.tolist(), global_arm=global_best,
                global_fit_probability=global_probability.tolist())
        def summarize(ids):
            estimates = {}
            for name, policy in chosen.items():
                matched = ids & (arm == policy)
                estimates[name] = dict(ipw_success=float(8*reward[matched].sum()/ids.sum()),
                    assignment_matches=int(matched.sum()), successful_matches=int(reward[matched].sum()),
                    chosen_arm_count=np.bincount(policy[ids],minlength=8).tolist())
            return dict(n=int(ids.sum()), estimates=estimates,
                position_minus_control={k: estimates['position']['ipw_success']-estimates[k]['ipw_success'] for k in ('global','unchanged')})
        pooled = summarize(test)
        seeds = {str(s): summarize(test & (seed == s)) for s in (525,526)}
        motions = {str(m): summarize(test & (motion == m)) for m in range(3)}
        gates = dict(pooled_gain5pp_both=all(v >= .05 for v in pooled['position_minus_control'].values()),
                     each_seed_positive_both=all(v > 0 for s in seeds.values() for v in s['position_minus_control'].values()))
        result = dict(run_status='COMPLETED', label='PROMISING' if all(gates.values()) else 'UNPROMISING',
            reused_data=True, posthoc=True, no_new_physics=True, no_model_training=True,
            fit_rows=4608, test_rows=1536, known_assignment_probability=.125,
            policies=policies, pooled=pooled, seeds=seeds, motions=motions, gates=gates,
            boundary='exploratory fixed-policy IPW estimates, not executed success, Cm utility or Validation')
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        if any(sha(Path(f)) != h for f,h in hashes.items()):
            raise ValueError('protected input drift')
        if time.monotonic()-begin > 60 or sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > 5<<20:
            raise RuntimeError('budget')
        manifest.update(run_status='COMPLETED', label=result['label'], inputs_unchanged=True)
        print(json.dumps(result))
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        manifest['wall_seconds']=time.monotonic()-begin
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
