#!/usr/bin/env python3
"""Endpoint and first-episode audit for the one frozen baseline continuation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_hold_plateau_substrate import score_phase,summarize
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.hold_plateau_training import canonical_model_state


def checkpoint_audit(manifest):
    final=Path(manifest['final_checkpoint'])
    p=torch.load(final,map_location='cpu',weights_only=False)
    source=torch.load(manifest['source_checkpoint'],map_location='cpu',weights_only=False)
    own=p['hold_plateau']
    if p['epoch']!=300 or p['frame']!=460800 or own['sampled_interactions']!=460800:
        raise ValueError('new epoch/frame/interaction budget drift')
    if not own['no_cm'] or not own['new_optimizer']:
        raise ValueError('baseline identity drift')
    expected=dict(model=fingerprint(canonical_model_state(source['model'])),observation_rms=fingerprint(source['running_mean_std']),amp_rms=fingerprint(source['amp_input_mean_std']))
    if own['initial_fingerprints']!=expected:
        raise ValueError('source network/normalizer initialization drift')
    np.testing.assert_allclose(own['fixed_exploration_std'][:3],.005,rtol=1e-6)
    np.testing.assert_allclose(own['fixed_exploration_std'][3:],np.exp(-2.9),rtol=1e-6)
    sigma=next(v for k,v in p['model'].items() if k.endswith('sigma')).exp().numpy()
    np.testing.assert_allclose(sigma,own['fixed_exploration_std'],rtol=1e-6)
    for mapping in (p['model'],p['running_mean_std'],p['amp_input_mean_std']):
        if any(not torch.isfinite(v).all() for v in mapping.values()):
            raise ValueError('nonfinite final weights/RMS')
    return dict(status='PASS',epoch=p['epoch'],new_interactions=p['frame'],
        initialized_from_self_trained_source=True,source_fingerprints=expected,
        final_model_fingerprint=fingerprint(canonical_model_state(p['model'])),final_rms_fingerprint=fingerprint(p['running_mean_std']),
        final_checkpoint_sha256=sha(final),fixed_exploration_std=own['fixed_exploration_std'],
        reset_counts_start_plateau=own['reset_counts_start_plateau'].tolist(),
        phase_samples=own['phase_samples'],positive_training_bonus_samples=own['positive_bonus_samples'],
        boundary='training labels/elevated initial states are not evaluated success')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--checkpoint-only',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    root=args.directory;manifest=json.loads((root/'run_manifest.json').read_text())
    audit=checkpoint_audit(manifest)
    if args.checkpoint_only:
        (root/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True);return
    if manifest['run_status']!='COLLECTION_COMPLETED' or manifest['panels']!=[[721,500],[721,501]]:
        raise ValueError('nonterminal/frozen test cohort drift')
    rows=[];panels={}
    for actor,evaluation in manifest['panels']:
        directory=root/f't{actor}_s{evaluation}'
        result=json.loads((directory/'results.json').read_text())
        if not result['all_episodes_complete'] or result['complete_episodes']!=192:
            raise ValueError('incomplete first episodes')
        if result['actor_fingerprint']!=audit['final_model_fingerprint'] or result['rms_fingerprint']!=audit['final_rms_fingerprint']:
            raise ValueError('evaluated actor/RMS differs from frozen endpoint')
        for name in ('initial','episodes'):
            if sha(directory/(name+'.pt'))!=result[name+'_sha256']:
                raise ValueError('trajectory drift')
        data=torch.load(directory/'episodes.pt',map_location='cpu',weights_only=False)
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        if not torch.equal(data['motion'],initial['motion']) or not torch.equal(data['initial_height'],initial['initial_height']):
            raise ValueError('native initial identity drift')
        if sorted(e['environment'] for e in data['episodes'])!=list(range(192)):
            raise ValueError('episode coverage drift')
        panel=[]
        for e in data['episodes']:
            env=e['environment'];motion=e['motion'];active=data['active'][:,env].numpy();steps=int(active.sum())
            if not np.array_equal(active,np.arange(len(active))<steps) or steps!=e['steps'] or motion!=int(data['motion'][env]):
                raise ValueError('active/identity trace drift')
            done=data['done'][:,env].numpy()[active]
            if not done[-1] or done[:-1].any() or e['terminate'] or steps!=data['loader_checks'][motion]['reference_frames']-1:
                raise ValueError('first native episode endpoint drift')
            scores=score_phase(data['height'][:,env].numpy()[active],data['contact'][:,env].numpy()[active],
                data['progress'][:,env].numpy()[active],data['initial_height'][env],int(data['phase_start'][motion]),int(data['phase_stop'][motion]))
            if any(e[k]!=scores[k] for k in ('max_phase_hold_steps','stable45','retained75','phase_steps')):
                raise ValueError('independent hold score mismatch')
            row=dict(e);row.update(scores);row.update(training_seed=actor,evaluation_seed=evaluation,
                initial_gap_m=float(initial['gap'][env]),initial_height_m=float(data['initial_height'][env]))
            panel.append(row);rows.append(row)
        panels[str(evaluation)]=summarize(panel)
    pooled=summarize(rows);motions={str(m):summarize([r for r in rows if r['motion']==m]) for m in range(3)}
    gates=dict(pooled75_at_least10percent=pooled['retained75_rate']>=.10,
        **{f'motion{m}_75_at_least5percent':motions[str(m)]['retained75_rate']>=.05 for m in range(3)})
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',
        complete_episodes=len(rows),gates=gates,pooled=pooled,motions=motions,panels=panels,
        final_checkpoint_sha256=audit['final_checkpoint_sha256'],synthetic_task=True,
        boundary='single-training-seed baseline feasibility; no Cm, causal superiority or Validation claim')
    torch.save(dict(rows=rows),root/'analysis.pt')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'trajectory_audit.json').write_text(json.dumps(dict(status='PASS',independently_scored_episodes=len(rows),
        actual_native_progress_and_endpoints_verified=True,plateau_samples_per_episode=90,
        final_endpoint_model_and_normalizers_verified=True),indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
