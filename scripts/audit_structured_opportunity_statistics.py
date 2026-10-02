#!/usr/bin/env python3
"""Independent raw-outcome, allocation, AIPW, bootstrap and decision audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def interval(values, clusters, seed):
    keys = sorted(set(clusters))
    if len(keys) < 2:
        return None
    totals = np.array([values[clusters == k].sum() for k in keys])
    counts = np.array([(clusters == k).sum() for k in keys])
    indices = np.random.default_rng(seed).integers(0, len(keys), (1000, len(keys)))
    return np.percentile(totals[indices].sum(1)/counts[indices].sum(1), [5, 95])


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise ValueError('unique audit output')
    source_path = args.source/'run_manifest.json'
    source = json.loads(source_path.read_text())
    result = json.loads(args.result.read_text())
    if (source['run_status'] != 'COMPLETED' or source['child_exit_code'] != 0
            or result['run_status'] != 'COMPLETED'
            or [p['seed'] for p in source['phases']] != list(range(611, 619))):
        raise ValueError('whole fixed terminal panel')
    for hashes in (source['input_sha256'], result['input_sha256']):
        if any(sha(Path(k)) != v for k, v in hashes.items()):
            raise ValueError('pinned input changed')
    import torch
    parts = []
    raw_score_error = 0.
    for phase in source['phases']:
        if phase['run_status'] != 'COMPLETED' or phase['native_exit_code'] or phase['audit_exit_code']:
            raise ValueError('accepted native and full planner audit required')
        directory = Path(phase['directory'])
        if sha(directory/'records.pt') != phase['result']['record_sha256'] or sha(Path(phase['audit'])) != phase['audit_sha256']:
            raise ValueError('actual source/audit hash')
        b = torch.load(directory/'records.pt', map_location='cpu', weights_only=False)
        def a(key): return b[key].numpy()
        allocation, arm = a('allocation'), a('assignment')
        mapping = np.array([0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4])
        probability = np.bincount(mapping, minlength=5)/16
        if not np.array_equal(mapping[allocation], arm) or not np.array_equal(probability[arm], a('propensity')):
            raise ValueError('known randomized propensity')
        bucket = np.array([int(hashlib.sha256(('12651/%d/%d' % (m,s)).encode()).hexdigest()[:8],16)%100
            for m,s in zip(a('motion_id'), a('start_frame'))])
        if not np.array_equal(bucket, a('split_group_bucket')):
            raise ValueError('split')
        mg = a('mass_kg').astype(float)*b['gravity_magnitude']
        hand = np.linalg.norm(a('future_hand_force').astype(float), axis=-1).max(-1)/mg[:,None] > .1
        obj = np.linalg.norm(a('future_object_force').astype(float), axis=-1)/mg[:,None] > .1
        clr = a('future_clearance') >= .002
        joint = (hand[:,-3:] & obj[:,-3:]).all(1)
        retained = joint & clr[:,-3:].all(1)
        # Preserve native float32 operation order for the saved mm label.
        score = (np.maximum(a('future_state')[:,-3:,38].min(1)-a('rest_z'),0)*retained
                 - np.maximum(a('state')[:,38]-a('rest_z'),0))*1000
        raw_score_error = max(raw_score_error, float(np.abs(score-b['outcome']['supported_change_mm'].numpy()).max()))
        seen = np.maximum.accumulate(np.column_stack((a('initial_clearance')>=.002,clr[:,:-1])), axis=1)
        loss = (seen & ~clr).any(1)
        q = np.concatenate((a('state')[:,None,:18],a('future_state')[:,:-1,:18]),1)
        cup = a('expert_bank')[:,:,1].copy()
        cup[:,:,3:6] = np.clip((a('hold_target')[:,None,3:6]-q[:,:,3:6]-a('pd_offset')[3:6])/a('pd_scale')[3:6],-1,1)
        def pd(raw, position):
            raw = raw.copy(); raw[...,6:] = .5*(raw[...,6:]+1)
            v = a('pd_offset')+a('pd_scale')*raw
            v[...,:6] += position[...,:6]
            for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
                v[...,dst] = v[...,src]*ratio
            return v
        delta = np.abs(a('actual_pd_targets')-pd(cup,q)).max(-1)
        proposed = pd(a('candidate_actions'),a('state')[:,None,:18])
        ep = np.array(b['episode_id'])
        initial = (a('initial_clearance')>=.002)&(a('state')[:,38]-a('rest_z')>=.03)
        for episode in set(ep):
            take = ep == episode
            ticks = a('trigger')[take]
            if len(ticks)>4 or (initial[take].sum()>2) or ((~initial[take]).sum()>2):
                raise ValueError('per-stratum repeated-window cap')
            if not np.array_equal(a('window_index')[take],np.arange(len(ticks))) or (np.diff(ticks)<15).any():
                raise ValueError('ordered distinct observe-plan-execute windows')
        parts.append(dict(arm=arm,slot=allocation,bucket=bucket,ep=ep,
            group=np.array(['%d/%d' % (m,s) for m,s in zip(a('motion_id'),a('start_frame'))]),
            y=np.column_stack((score,loss,joint)).astype(float),
            mu=a('nuisance_forecasts').astype(float).mean(1)[:,:5],
            proposed=np.abs(proposed[:,2]-proposed[:,1]).max(-1)>1e-5,
            steps=(delta>1e-5).sum(1),changed=delta.max(1)>1e-5,initial=initial))
    if raw_score_error>2e-5:
        raise ValueError('raw supported-score mismatch')
    d = {k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    probability = np.array([2,4,4,3,3])/16
    # Independently construct augmentation plus known randomization residual.
    estimates = np.stack([d['mu'][:,arm] + (d['arm']==arm)[:,None]*(d['y']-d['mu'][:,arm])/p
        for arm,p in enumerate(probability)],1)
    split = dict(fit=d['bucket']<50,cal=(d['bucket']>=50)&(d['bucket']<70),held=d['bucket']>=70)
    ref = int(np.argmax(estimates[split['fit'],:2,0].mean(0)))
    if ref != result['fixed_reference']:
        raise ValueError('fit-only fixed control')
    errors = {}
    def equal(key,actual,expected):
        if actual is None or expected is None:
            if actual is not None or expected is not None: raise ValueError(key)
            return
        error = float(np.abs(np.asarray(actual)-np.asarray(expected)).max())
        errors[key] = error
        if error>2e-5: raise ValueError(key+': '+str(error))
    sufficient = True; passed = True
    controls = dict(state_only=1,shuffled=4,always_base=0,best_fixed=ref,direct_score=3)
    for name,take in split.items():
        counts = result['counts'][name]
        arms = np.bincount(d['arm'][take],minlength=8)
        adequate = int(take.sum())>=150 and len(set(d['ep'][take]))>=32 and len(set(d['group'][take]))>=8 and np.all(arms[:5]>=24)
        if counts['adequate'] != adequate or counts['rows'] != int(take.sum()) or counts['arms'] != arms.tolist():
            raise ValueError('source support '+name)
        sufficient &= bool(adequate); passed &= bool(adequate)
        if name == 'fit': continue
        nulls = []
        available = True
        for key,arm,left,right in (('base',0,[0],[1]),('cup',1,[2,3],[4,5])):
            present = np.any(take&np.isin(d['slot'],left)) and np.any(take&np.isin(d['slot'],right))
            available &= bool(present)
            difference = (np.isin(d['slot'],left)*16/len(left)-np.isin(d['slot'],right)*16/len(right))*(d['y'][:,0]-d['mu'][:,arm,0])
            value = float(difference[take].mean()) if present else None
            equal(name+'.null.'+key,value,result['reference_null'][name]['pairs'][key]['difference_mm'])
            if present: nulls.append(2*abs(value))
        threshold = max([1.]+nulls) if available else None
        equal(name+'.threshold',threshold,result['reference_null'][name]['required_gain_mm'])
        actual = take&(d['arm']==2)
        coverage = result['coverage'][name]
        if coverage['actual_non_cup_pd_windows'] != int((actual&d['changed']).sum()) or coverage['actual_non_cup_pd_steps'] != int(d['steps'][actual].sum()):
            raise ValueError('actual control coverage')
        for control,arm in controls.items():
            difference = estimates[:,2]-estimates[:,arm]
            score,risk,joint = difference[take].mean(0)
            ci_group = interval(difference[take,0],d['group'][take],12753)
            ci_ep = interval(difference[take,0],d['ep'][take],12754)
            comparison = result['comparisons'][name][control]
            for key,value in (('score_difference_mm',score),('loss_difference',risk),('joint_difference',joint),('group90',ci_group),('episode90',ci_ep)):
                equal(name+'.'+control+'.'+key,value,comparison[key])
            support = True
            for side in (2,arm):
                match = take&(d['arm']==side)
                support &= int(match.sum())>=24 and len(set(d['ep'][match]))>=12 and len(set(d['group'][match]))>=8
            gates = dict(support=bool(support),null_available=available,score=threshold is not None and score>=threshold,
                group90_positive=ci_group is not None and ci_group[0]>0,episode90_positive=ci_ep is not None and ci_ep[0]>0,
                loss_point_nonregression=risk<=.02,joint_point_nonregression=joint>=-.05,
                proposed_physical_change10pct=d['proposed'][take].mean()>=.1,actual_non_cup_windows12=int((actual&d['changed']).sum())>=12)
            gates['passed'] = all(gates.values())
            if gates != comparison['gate']: raise ValueError('gate '+name+'.'+control)
            sufficient &= support and available; passed &= gates['passed']
    label = 'PROMISING' if passed else ('UNPROMISING' if sufficient else 'UNCLEAR')
    if label != result['label']: raise ValueError('fixed classification')
    report = dict(run_status='COMPLETED',audit_passed=True,label=label,raw_supported_score_max_error_mm=raw_score_error,
        statistic_max_error=max(errors.values()),source_sha256=sha(source_path),result_sha256=sha(args.result),
        auditor_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-started,
        all_fixed_gates_reconstructed=True,scope='independent actual-label and estimator audit; no final policy claim')
    if source['cumulative_seconds']+result['elapsed_seconds']+report['elapsed_seconds']>3600:
        raise TimeoutError('whole fixed slot budget')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
