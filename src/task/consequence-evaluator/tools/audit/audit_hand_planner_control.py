"""Full-episode prospective policy comparison; matched IDs are not causal forks."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import subprocess
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/cm-interaction-oracle/src')]
from consequence_evaluator.data import sha
from consequence_evaluator.value_outcomes import task_trace,PARAMETERS
from consequence_evaluator.temporal_value import episode_labels
from consequence_evaluator.supervision import consecutive
from oracle_y_utility import paired_bootstrap,candidate_deltas


def verify_runtime_inputs(manifest):
    """Artifacts stay immutable; earlier source blobs are checked at their commit."""
    for path,digest in manifest['input_sha256'].items():
        if sha(path)==digest:continue
        source=Path(path)
        try:relative=source.resolve().relative_to(ROOT)
        except ValueError:raise ValueError('external immutable input drift: '+path)
        if source.suffix!='.py':raise ValueError('immutable artifact drift: '+path)
        blob=subprocess.check_output(['git','show',manifest['git_commit']+':'+str(relative)],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('unrecoverable runtime source drift: '+path)


def summarize(folder):
    m=json.loads((folder/'manifest.json').read_text())
    if m['status']!='COMPLETED' or m['smoke'] or m['mode'] not in ('baseline','planner'):
        raise ValueError('complete prospective policy run required')
    if sha(folder/'trajectory.npz')!=m['trajectory_sha256']:raise ValueError('trajectory drift')
    verify_runtime_inputs(m)
    if m['mode']=='planner':
        if not m['PW_repeat_checked']:raise ValueError('PW repeatability missing')
        if any(sha(p)!=h for p,h in m['planner_input_sha256'].items()):raise ValueError('planner weights/source drift')
    with np.load(folder/'trajectory.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    if m['mode']=='planner':
        if sha(folder/'decisions.npz')!=m['decisions_sha256']:raise ValueError('online decision drift')
        with np.load(folder/'decisions.npz',allow_pickle=False) as f:q={k:f[k] for k in f.files}
        if not np.isfinite(q['scores']).all() or not np.array_equal(q['candidate'],q['scores'].argmax(1)):
            raise ValueError('executed candidate differs from frozen scores')
        delta=candidate_deltas().numpy()[q['candidate']]
        if not np.array_equal(q['action'][:,:8],np.repeat(delta[:,None],8,axis=1)) or q['action'][:,8:].any():
            raise ValueError('requested24plan does not match selected candidate')
        for i,(tick,e) in enumerate(zip(q['tick'],q['env'])):
            if not np.array_equal(q['object_history'][i],d['object_pose'][tick-3:tick+1,e]):raise ValueError('PW past object geometry mismatch')
            if not np.array_equal(q['hand_history'][i],d['hand_keypoints'][tick-3:tick+1,e]):raise ValueError('bridge past hand geometry mismatch')
        m=dict(m,online_selection_contract=True,decision_windows=len(q['tick']),candidate_counts=np.bincount(q['candidate'],minlength=7).tolist())
    rows=[]
    for e,n in enumerate(d['length']):
        s=slice(0,n+1);valid=np.ones(n+1,bool);valid[0]=False
        if n!=m['native_reference_steps'] or d['done'][:n-1,e].any() or not d['done'][n-1,e]:
            raise ValueError('first full episode terminal coverage failed')
        packet=dict(action=d['action'][:n,e],object_pose=d['object_pose'][s,e])
        diagnostics={k:d[k][s,e] for k in ('surface_gap','object_velocity','support_gap','table_footprint','reference_object_pose')}
        diagnostics.update(contact_valid=valid,initial_height=float(packet['object_pose'][0,2,3]))
        trace=task_trace(packet,diagnostics)
        success,recovery=episode_labels(trace,diagnostics['object_velocity'])
        stable=bool((trace['held_run']>=PARAMETERS['stable_frames']).any())
        first=np.flatnonzero(trace['held_run']>=PARAMETERS['stable_frames'])
        # Recovery windows exclude intended supported final release, inherited
        # from the user-confirmed outcome/mask rule; no evaluator labels change.
        had_recovery=bool(recovery.any());loss_after_stable=bool(len(first) and recovery[first[0]+1:].any())
        completion=np.flatnonzero((trace['held_run']>=PARAMETERS['stable_frames']) & (np.arange(n+1)<trace['place_start']))
        failure_reason=None
        if not success:
            if not len(completion):failure_reason='no_stable_hold_before_place'
            else:
                last=int(completion[-1]);place=int(trace['place_start'])
                missing=trace['valid']&~trace['near']&~trace['supported']
                unsafe=(consecutive(missing)>=PARAMETERS['lost_geometry_frames'])
                unsafe|=missing&(diagnostics['object_velocity'][:,2]<-PARAMETERS['unheld_fall_mps'])
                if trace['drop'][last+1:place].any():failure_reason='lost_between_last_stable_and_place'
                elif unsafe[max(last+1,place):].any():failure_reason='unsafe_unheld_placing'
                else:failure_reason='final_placement_not_settled'
        rows.append(dict(env=e,success=int(success),stable_lift45=stable,recovery=had_recovery,
            loss_after_stable=loss_after_stable,recovered_success=bool(success and had_recovery),
            task_failure_reason=failure_reason,final_supported=bool(trace['supported'][-1]),
            settled_tail_frames=int(consecutive(trace['settled'])[-1]),place_start=int(trace['place_start']),
            maximum_held_frames=int(trace['held_run'].max()),interventions=int(d['interventions'][e]),
            clipped_steps=int(d['clipped'][:n,e].sum())))
    return m,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,action='append',required=True)
    p.add_argument('--planner',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if len(a.baseline)!=len(a.planner):raise ValueError('matched seed runs required')
    comparisons=[];allbase=[];allplan=[];hashes={str(Path(__file__).resolve()):sha(__file__)}
    for b,p in zip(a.baseline,a.planner):
        bm,br=summarize(b);pm,pr=summarize(p)
        for key in ('seed','num_envs','actor_sha256','actor_fingerprint','rms_fingerprint','initial_q_sha256','initial_object_sha256','native_reference_steps','initialization'):
            if bm[key]!=pm[key]:raise ValueError('matched policy start/actor contract differs: '+key)
        x=np.array([r['success'] for r in pr]);y=np.array([r['success'] for r in br])
        comparisons.append(dict(seed=bm['seed'],episodes=len(x),baseline_success=int(y.sum()),planner_success=int(x.sum()),
            matched_ID_rescue=int(((x==1)&(y==0)).sum()),matched_ID_harm=int(((x==0)&(y==1)).sum()),
            descriptive_gain=paired_bootstrap(x-y,seed=291),online_selection_contract=pm['online_selection_contract'],
            decision_windows=pm['decision_windows'],candidate_counts=pm['candidate_counts']))
        allbase.extend(br);allplan.extend(pr)
        for folder in (b,p):
            for name in ('manifest.json','trajectory.npz'):hashes[str(folder/name)]=sha(folder/name)
    def aggregate(rows):
        return dict(episodes=len(rows),success=sum(r['success'] for r in rows),
            success_rate=float(np.mean([r['success'] for r in rows])),stable_lift45=sum(r['stable_lift45'] for r in rows),
            recovery=sum(r['recovery'] for r in rows),loss_after_stable=sum(r['loss_after_stable'] for r in rows),
            recovered_success=sum(r['recovered_success'] for r in rows),
            task_failure_reasons={reason:sum(r['task_failure_reason']==reason for r in rows)
                for reason in sorted({r['task_failure_reason'] for r in rows if r['task_failure_reason'] is not None})},
            mean_interventions=float(np.mean([r['interventions'] for r in rows])),clipped_steps=sum(r['clipped_steps'] for r in rows))
    base=aggregate(allbase);plan=aggregate(allplan)
    gains=np.array([c['planner_success']/c['episodes']-c['baseline_success']/c['episodes'] for c in comparisons])
    status='PROMISING' if (plan['success_rate']-base['success_rate']>=.05 and (gains>0).all()) else 'UNCLEAR'
    result=dict(status=status,baseline=base,planner=plan,per_seed=comparisons,baseline_episodes=allbase,planner_episodes=allplan,
        task_rule='inherited final controlled completion with recovery allowed; same45/15frame thresholds',
        old_Y_unchanged=True,formal_Gate1=False,RL_benefit=False,input_sha256=hashes,
        scope='two exploratory seed worlds; matched initial episode IDs, not bitwise physical counterfactual rescue/harm')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','baseline','planner','per_seed','scope')},indent=2),flush=True)


if __name__=='__main__':main()
