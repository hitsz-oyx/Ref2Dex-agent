"""Saved-array, outcome-oracle decomposition; never a deployable prediction."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def aggregate(errors_mm, env, mask):
    return float(np.mean([errors_mm[mask & (env == e)].mean()
                          for e in np.unique(env[mask])]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--capacity', type=Path, required=True)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert root in a.output.resolve().parents and not a.output.exists()
    sources = [a.parent/'results.json', a.parent/'run_manifest.json',
               a.parent/'fit/full_outputs.npz', a.parent/'fit/state_only_outputs.npz',
               a.parent/'fit/shuffled_outputs.npz', a.capacity/'capacity/fields.npz',
               a.capacity/'capacity/causal_oracle.npz',
               a.execution/'qualified/held_rows.npz', a.execution/'qualified/features.npz',
               Path(__file__).resolve(),
               root/'docs/experiments/probes/P-20261003-rigid-coupling-error-decomposition.md']
    begin = time.monotonic()
    hashes = {str(f):sha(f) for f in sources}
    parent = json.loads(sources[0].read_text())
    assert parent['run_status'] == 'COMPLETED' and parent['label'] == 'UNPROMISING'
    with np.load(a.capacity/'capacity/fields.npz') as f:
        anchor, target, endpoints = f['anchor'], f['target'], f['causal']
    with np.load(a.parent/'fit/full_outputs.npz') as f:
        w, scores, winner, prediction = f['coefficients'], f['scores'], f['winner'], f['prediction']
    with np.load(a.capacity/'capacity/causal_oracle.npz') as f:
        oracle_w, oracle_prediction = f['coefficients'], f['prediction']
    with np.load(a.execution/'qualified/held_rows.npz') as f:
        env = f['env']
    with np.load(a.execution/'qualified/features.npz') as f:
        near = f['stationary'][...,15].min(-1)*.05 < .02
    candidate = anchor[:,None] + w.astype(np.float64)[...,None,None]*(endpoints-anchor[:,None])
    candidate_error = np.linalg.norm(candidate-target[:,None],axis=-1).mean(-1)*1000
    n = np.arange(len(env))
    assert np.array_equal(winner, scores.argmin(1))
    assert np.max(np.abs(candidate[n,winner]-prediction)) < 1e-12
    oracle_error = np.linalg.norm(oracle_prediction-target,axis=-1).mean(-1)*1000
    assert np.min(candidate_error-oracle_error[:,None]) >= -1e-4
    fixed_choice_oracle = anchor + oracle_w[n,winner,None,None]*(endpoints[n,winner]-anchor)
    fixed_choice_error = np.linalg.norm(fixed_choice_oracle-target,axis=-1).mean(-1)*1000
    errors = dict(deployed=candidate_error[n,winner],
                  best_learned_coefficient=candidate_error.min(1),
                  oracle_coefficient_at_deployed_winner=fixed_choice_error,
                  full_outcome_oracle=oracle_error,
                  full_model_state_candidates=candidate_error[n,scores[:,:2].argmin(1)],
                  best_full_model_state_candidates=candidate_error[:,:2].min(1))
    reports = {name:{k:aggregate(v,env,mask) for k,v in errors.items()}
               for name,mask in [('all',np.ones(len(env),dtype=bool)),('near',near),('far',~near)]}
    assert abs(reports['all']['deployed']-parent['reports']['full']['episode_epe_mm']) < 1e-9
    for r in reports.values():
        r['selection_regret_mm'] = r['deployed']-r['best_learned_coefficient']
        r['coefficient_gap_mm'] = r['best_learned_coefficient']-r['full_outcome_oracle']
    gates = dict(overall=reports['all']['best_learned_coefficient'] <= .9*parent['reports']['state_only']['episode_epe_mm'],
                 near=reports['near']['best_learned_coefficient'] <= .95*parent['diagnostics']['near']['state_only']['episode_epe_mm'])
    assert all(sha(Path(f)) == h for f,h in hashes.items())
    elapsed = time.monotonic()-begin
    assert elapsed < 60
    result = dict(run_status='COMPLETED',experiment_id='P-20261003-rigid-coupling-error-decomposition',
                  run_id=a.output.name,source_sha256=hashes,reports=reports,gates=gates,
                  label='PROMISING' if all(gates.values()) else 'UNPROMISING',
                  oracle_selection_not_deployable=True,parent_label_unchanged=parent['label'],
                  new_optimizer_updates=0,new_native_ticks=0,execution_device='cpu',wall_seconds=elapsed)
    data = json.dumps(result,indent=2)+'\n'
    assert len(data.encode()) < 1048576
    a.output.mkdir()
    (a.output/'results.json').write_text(data)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
