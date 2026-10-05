#!/usr/bin/env python3
"""Fixed-Y channel/continuous diagnostics; never reconstruct a switching policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
from rolling_y_audit import EPSILON, rolling_y
from oracle_y_utility import utility

CHANNELS = ('contact16', 'held16', 'failure16', 'contact32', 'held32',
            'failure32', 'height_failure32', 'height_fraction32')
ORIENTATION = np.array([1, 1, -1, 1, 1, -1, -1, 1])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_signal(gap, offsets, eligible, event, threshold=EPSILON):
    """Pre-event and joint-risk eligibility, with explicit absent-event censoring."""
    gap = np.asarray(gap)
    mask = np.asarray(eligible, bool) & (np.asarray(offsets) < event) if event is not None else np.zeros(len(offsets), bool)
    def first(condition):
        found = np.flatnonzero(mask & condition)
        return None if not len(found) else dict(offset=int(offsets[found[0]]), lead_steps=int(event-offsets[found[0]]))
    return dict(absolute=first(np.abs(gap) > threshold), correct=first(gap > threshold),
                inverse=first(gap < -threshold), eligible_queries=int(mask.sum()),
                event_missing=event is None)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), minimum=float(values.min()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None,
                histogram={str(int(x)):int((values == x).sum()) for x in np.unique(values)})


def continuous(height, pair, rest, before, offsets):
    """Diagnostic physical quantities, local steps9..32; velocity per control step."""
    result=[]
    for offset in offsets:
        h=height[..., offset:offset+32]-rest[..., None]
        current=before-rest if offset == 0 else height[..., offset-1]-rest
        velocity=np.diff(np.concatenate([current[..., None],h],axis=-1),axis=-1)
        result.append(np.stack([h[...,8:].min(-1)-.02,h[...,31]-current,
                                velocity[...,8:].min(-1),pair[...,offset+8:offset+32].mean(-1)],axis=-1))
    return np.stack(result,axis=-2)


def audit_factual(arrays, source):
    y=arrays['y']; offsets=arrays['offsets']; risk=arrays['risk']; z=arrays['z']
    reconstructed, reconstructed_risk=rolling_y(*[torch.from_numpy(arrays[key]) for key in
        ('height','pair','rest','before_height','before_pair')],tuple(int(x) for x in offsets))
    if not np.array_equal(y,reconstructed.numpy()) or not np.array_equal(risk,reconstructed_risk.numpy()):
        raise ValueError('fixed Y/risk reconstruction mismatch')
    c=continuous(arrays['height'],arrays['pair'],arrays['rest'],arrays['before_height'],offsets)
    u=utility(y)
    if not np.array_equal(u,arrays['u']): raise ValueError('fixed U mismatch')
    reports=[]
    for entry in source['pair_reports']:
        a,g,b=entry['anchor'],entry['good'],entry['bad']; joint=risk[a,g]&risk[a,b]
        event=entry['bad_event']['first_raw_failure_step']; post=entry['bad_event']['first_postqualification_drop_step']
        channels={name:first_signal((y[a,g,:,i]-y[a,b,:,i])*ORIENTATION[i],offsets,joint,event) for i,name in enumerate(CHANNELS)}
        # Units differ; continuous equality is diagnostic, no accuracy/pass threshold is fitted.
        cg=c[a,g]-c[a,b]
        continuous_report={name:dict(initial_good_minus_bad=float(cg[0,i]),
            first_numerically_distinct_offset=next((int(offsets[t]) for t in range(len(offsets)) if joint[t] and abs(cg[t,i])>1e-9),None))
            for i,name in enumerate(('min_height_margin_m','delta_height_m','min_vertical_increment_m_per_step','contact_fraction'))}
        tied=bool(u[a,g,0] == u[a,b,0]); full_tied=bool(np.array_equal(y[a,g,0],y[a,b,0]))
        reports.append(dict(anchor=a,identity=source['identities'][a],good=g,bad=b,initial_utility_tie=tied,
            initial_full_y_tie=full_tied,initial_continuous_height_distinct=bool(np.any(np.abs(cg[0,:3])>1e-9)),
            first_raw_failure_step=event,first_postqualification_drop_step=post,
            initial_failure_after_32=event is not None and event>32,
            initial_failure_in_excluded_steps_1_to_8=event is not None and event<=8,
            channels=channels,utility=first_signal(u[a,g]-u[a,b],offsets,joint,event),
            utility_postqualification_event=first_signal(u[a,g]-u[a,b],offsets,joint,post),continuous=continuous_report))
    summary={name:dict(correct_pairs=sum(p['channels'][name]['correct'] is not None for p in reports),
        absolute_pairs=sum(p['channels'][name]['absolute'] is not None for p in reports),
        inverse_pairs=sum(p['channels'][name]['inverse'] is not None for p in reports),
        correct_lead_distribution=distribution([p['channels'][name]['correct']['lead_steps'] for p in reports if p['channels'][name]['correct'] is not None]),
        initially_tied_correct_pairs=sum(p['initial_utility_tie'] and p['channels'][name]['correct'] is not None for p in reports)) for name in CHANNELS}
    tied=[p for p in reports if p['initial_utility_tie']]
    earliest={name:0 for name in CHANNELS}; no_signal=0
    for p in tied:
        available=[(name,r['correct']['offset']) for name,r in p['channels'].items() if r['correct'] is not None]
        if not available:no_signal+=1;continue
        first=min(t for _,t in available)
        for name,t in available:
            if t == first:earliest[name]+=1
    return dict(causal_scope='different later H; factual-path observability, not common-state candidate ranking',
        anchors=int(len(z)),discordant_pairs=len(reports),channel_threshold=EPSILON,
        channel_summary=summary,earliest_correct_channels_among_initial_utility_ties=earliest,
        earliest_channels_overlap=True,ties=dict(count=len(tied),full_y_equal=sum(p['initial_full_y_tie'] for p in tied),
        continuous_height_distinct=sum(p['initial_continuous_height_distinct'] for p in tied),
        first_bad_raw_event_after_initial_horizon=sum(p['initial_failure_after_32'] for p in tied),
        no_pre_event_channel_signal=no_signal),
        utility_lead_distribution=distribution([p['utility']['correct']['lead_steps'] for p in tied if p['utility']['correct'] is not None]),
        pair_reports=reports)


def rolling_groups(root, read):
    groups=[]
    # Only finalized group result files; ongoing group rounds are excluded.
    for path in sorted(root.glob('s*-g*-result.json')):
        group=read(path); rounds=[]
        for decision in group['decisions']:
            plan=read(root / ('s%d-g%d-t%02d-plan.json'%(group['seed'],group['group'],decision['offset'])))
            y=np.asarray(plan['y'],dtype=np.float32); u=np.asarray(plan['utility'],dtype=np.float32)
            if not np.array_equal(utility(y),u):raise ValueError('rolling plan U mismatch')
            k=y.shape[1]; rows=plan['rows']; choices=np.asarray(plan['choices'])[rows]
            if not np.array_equal(np.argmax(u,axis=1),choices):raise ValueError('rolling selected candidate mismatch')
            certified=bool(plan.get('baseline_upper_bound_certificate'))
            if certified and k != 1:raise ValueError('certificate should store only baseline Y')
            rounds.append(dict(offset=decision['offset'],anchors=len(y),observed_candidates=k,
                baseline_certificate=certified,missing_alternative_labels=certified,
                exact_utility_tied_anchors=int(np.all(u==u[:,:1],axis=1).sum()) if k>1 else None,
                full_y_tied_anchors=int(np.all(y==y[:,:1],axis=(1,2)).sum()) if k>1 else None,
                channel_spread_mean={name:float(np.ptp(y[:,:,i],axis=1).mean()) for i,name in enumerate(CHANNELS)} if k>1 else None))
        groups.append(dict(seed=group['seed'],group=group['group'],anchors=len(group['rows']),rounds=rounds,
            causal_scope='same-current-state fork labels within each round; comparisons across rounds change H',
            candidate_panel_hashes_verified_here=False,source='completed group result and frozen plans; raw panels audited separately'))
    return groups



def continuous_asset(path):
    asset=torch.load(path,map_location='cpu',weights_only=False)
    if asset['schema'] != 'ref2dex.rolling_oracle_asset.v1':raise ValueError('unknown asset schema')
    reports=[]
    for row in asset['rounds']:
        records=row['candidates']
        if any(p is None for p in records):
            reports.append(dict(offset=row['offset'],complete_candidates=False));continue
        y=torch.stack([p['Y32'] for p in records],1).numpy()
        u=torch.stack([p['U'] for p in records],1).numpy()
        tr=torch.stack([p['physical32'] for p in records],1).numpy()
        rests=asset['rest_height'].numpy()
        force=np.linalg.norm(tr[...,48:63].reshape(*tr.shape[:-1],5,3),axis=-1)
        values=dict(min_height_margin_m=(tr[:,:,8:,2]-rests[:,None,None]).min(-1)-.02,
                    mean_hand_force_norm_N=force[:,:,8:].mean(axis=(-1,-2)),
                    min_body_object_distance_m=tr[:,:,8:,66:71].min(axis=(-1,-2)))
        tied=np.all(u==u[:,:1],axis=1)
        reports.append(dict(offset=row['offset'],complete_candidates=True,utility_tied_anchors=int(tied.sum()),
            metrics={name:dict(tied_anchor_spreads=np.ptp(v,axis=1)[tied].tolist(),
                numerically_distinct_tied_anchors=int((np.ptp(v,axis=1)[tied]>1e-9).sum())) for name,v in values.items()}))
    return dict(seed=asset['seed'],group=asset['group'],rounds=reports,
        scope='Continuous same-current-state candidate differences; no candidate Z or failure forecast claim.',
        contact_margin_definition='Raw force norm and proximity, not certified contact/friction margins.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factual-run',type=Path,required=True)
    parser.add_argument('--rolling-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--asset',type=Path)
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();torch.set_num_threads(2); hashes={}
    def read(path):
        raw=path.read_bytes();hashes[str(path.resolve())]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    source=read(args.factual_run/'result.json'); npz=args.factual_run/'rolling.npz';hashes[str(npz.resolve())]=sha(npz)
    with np.load(npz) as arrays:
        factual=audit_factual(arrays,source)
        dense_arrays={key:arrays[key] for key in arrays.files}
        for key in ('y','u','risk','offsets'):dense_arrays[key]=arrays['dense_'+key]
        dense=audit_factual(dense_arrays,source)
    groups=rolling_groups(args.rolling_run,read)
    asset_report=None
    if args.asset:
        hashes[str(args.asset.resolve())]=sha(args.asset)
        asset_report=continuous_asset(args.asset)
    report=dict(status='UNCLEAR',purpose='Decision diagnostic: distinguish delayed horizon entry and coarse binary labels; identify useful warnings before any target redesign.',
        device='cpu',cpu_reason='Only fixed-label reconstruction and statistics; no model/simulation.',
        fixed_contract='Y/U/Z unchanged; epsilon .02 inherited, exact utility tie unchanged.',
        factual=factual,dense_boundary_diagnostic=dense,completed_rolling_groups=groups,continuous_rolling_asset=asset_report,
        limits=['Pairs are correlated within anchor/shared solver group; counts are descriptive.',
        'Continuous metrics use numerical distinction only; no learned or tuned usefulness threshold.',
        'Height margins can differ without different outcomes; does not justify replacing Y.',
        'Height horizon vs coarse threshold explanations are compatible, not uniquely causally identified.',
        'No raw continuous force/proximity or complete flow in rolling.npz; contact is forcepair binary proxy.',
        'Missing alternatives under baseline certificate are unknown, never copied from baseline.',
        'No trajectory stitching, training, PhysX, or rolling Z/gain inference in this audit.'])
    for path,value in hashes.items():
        if sha(path)!=value:raise ValueError('finalized input changed during audit: '+path)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),input_sha256=hashes,
        code_sha256={str(path.resolve()):sha(path) for path in (Path(__file__),ROOT/'src/task/cm-interaction-oracle/src/rolling_y_audit.py',ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py')},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        elapsed_seconds=time.monotonic()-start,command=sys.argv,run_status='COMPLETED',new_simulation=0,new_fits=0)
    for name,obj in [('result.json',report),('manifest.json',manifest)]:
        (args.run_dir/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    if manifest['elapsed_seconds']>600 or sum(p.stat().st_size for p in args.run_dir.iterdir())>200*1024**2:raise ValueError('audit budget exceeded')
    print(json.dumps(dict(ties=factual['ties'],utility_lead=factual['utility_lead_distribution'],channels=factual['channel_summary'],completed_groups=len(groups)),indent=2))

if __name__ == '__main__':main()
