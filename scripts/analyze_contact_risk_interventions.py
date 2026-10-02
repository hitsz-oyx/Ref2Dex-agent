#!/usr/bin/env python3
"""Fixed conditional contact-safety decision; full controls remain visible."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from analyze_orientation_feedback_opportunity import clustered_interval
from analyze_structured_contact_opportunity import pd_targets
from contact_risk_intervention_statistics import contributions,duplicate_contrast


def arrays(record):
    def value(key):return record[key].numpy()
    n=len(record['state'])
    force_weight=value('mass_kg')[:,None]*record['gravity_magnitude']
    hand=np.linalg.norm(value('future_hand_force'),axis=-1).max(-1)/force_weight
    obj=np.linalg.norm(value('future_object_force'),axis=-1)/force_weight
    contact=(hand>.1)&(obj>.1)
    loss=(~contact).any(-1).astype(float)
    joint=contact[:,-3:].all(-1).astype(float)
    clearance=value('future_clearance')>=.002
    support=contact[:,-3:].all(-1)&clearance[:,-3:].all(-1)
    score=(np.maximum(value('future_state')[:,-3:,38].min(-1)-value('rest_z'),0)*support-
           np.maximum(value('state')[:,38]-value('rest_z'),0))*1000
    previous=np.concatenate(((value('initial_clearance')>=.002)[:,None],clearance[:,:-1]),-1)
    geometric=(np.maximum.accumulate(previous,axis=-1)&~clearance).any(-1).astype(float)
    q=np.concatenate((value('state')[:,None,:18],value('future_state')[:,:-1,:18]),1)
    bank=value('expert_bank');weights=value('candidate_weights')[:,3]
    alternative=bank[:,:,1].copy()
    for block,(start,stop) in enumerate(((0,3),(6,8),(8,10),(10,12),(12,14),(14,18))):
        alternative[...,start:stop]=np.einsum('ne,nted->ntd',weights[:,block],bank[...,start:stop])
    offset,scale=value('pd_offset'),value('pd_scale')
    alternative[...,3:6]=np.clip((value('hold_target')[:,None,3:6]-q[:,:,3:6]-offset[3:6])/scale[3:6],-1,1)
    alternative_pd=pd_targets(alternative,q,offset,scale)
    delta=np.max(np.abs(alternative_pd-value('actual_pd_targets')),axis=-1)
    return dict(outcomes=np.stack((score,loss,geometric,joint),-1).astype(float),
        mu=value('nuisance_forecasts').astype(float).mean(1)[:,:5],assignment=value('assignment'),
        propensity=value('propensity'),allocation=value('allocation'),bucket=value('split_group_bucket'),
        episode=np.array(record['episode_id']),group=np.array([f'{int(m)}/{int(s)}' for m,s in zip(record['motion_id'],record['start_frame'])]),
        initial=record['outcome']['initially_clear'].numpy(),actual_pd_difference=delta.max(-1)>1e-5,
        actual_pd_changed_steps=(delta>1e-5).sum(-1),trigger=value('trigger'))


def load(source):
    import torch
    torch.set_num_threads(2)
    manifest_path=source/'run_manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['run_status']!='COMPLETED' or manifest.get('child_exit_code')!=0 or manifest['smoke_only']:
        raise ValueError('complete independent scientific source required')
    if [p['seed'] for p in manifest['phases']]!=list(range(651,659)):
        raise ValueError('whole frozen eight-seed panel')
    if any(sha(Path(k))!=v for k,v in manifest['input_sha256'].items()):raise ValueError('source drift')
    hashes={str(manifest_path.resolve()):sha(manifest_path)};parts=[];detections=0
    for phase in manifest['phases']:
        directory=Path(phase['directory']);record_path=directory/'records.pt';plan=directory/'planning.pt'
        audit_path=Path(phase['audit']);audit=json.loads(audit_path.read_text())
        control=source/'phase-controls'/f"seed{phase['seed']}"/'run_manifest.json'
        if phase['run_status']!='COMPLETED' or phase.get('native_exit_code')!=0 or phase.get('audit_exit_code')!=0:
            raise ValueError('every owned native and independent audit must exit zero')
        if sha(record_path)!=phase['result']['record_sha256'] or sha(plan)!=phase['result']['planning_sha256']:
            raise ValueError('actual source hashes')
        if sha(audit_path)!=phase['audit_sha256'] or audit['run_status']!='COMPLETED' or audit['run_manifest_sha256']!=sha(control):
            raise ValueError('independent audit ownership')
        record=torch.load(record_path,map_location='cpu',weights_only=False)
        if record['schema']!='ref2dex.contact_risk_interventions_source.v1' or record['seed']!=phase['seed']:
            raise ValueError('truthful source identity')
        if record['future_done'].any() or len(record['state'])!=audit['phases'][0]['physical']['rows']:
            raise ValueError('accepted complete physical rows')
        detections+=record['shadow_detection_rows'];parts.append(arrays(record))
        for path in (record_path,plan,audit_path,control):hashes[str(path.resolve())]=sha(path)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]},detections,manifest,hashes


def support(data,arm,take):
    chosen=(data['assignment']==arm)&take
    return dict(rows=int(chosen.sum()),episodes=len(set(data['episode'][chosen])),groups=len(set(data['group'][chosen])))


def summarize(data,detections):
    n=len(data['assignment']);take=np.ones(n,dtype=bool)
    if not n:
        return dict(label='UNCLEAR',rows=0,shadow_detection_rows=detections,reason='no actual binding windows')
    aipw=contributions(data['assignment'],data['outcomes'],data['mu'])
    ht=contributions(data['assignment'],data['outcomes'],np.zeros_like(data['mu']))
    supports={str(arm):support(data,arm,take) for arm in range(5)}
    counts=dict(rows=n,episodes=len(set(data['episode'])),groups=len(set(data['group'])),arms=np.bincount(data['assignment'],minlength=5).tolist())
    adequate=n>=64 and counts['episodes']>=24 and counts['groups']>=8
    adequate &= all(supports[str(a)]['rows']>=24 and supports[str(a)]['episodes']>=12 and supports[str(a)]['groups']>=8 for a in (2,3))
    adequate &= supports['0']['rows']>=4 and supports['1']['rows']>=4 and supports['4']['rows']>=8
    comparisons={}
    for name,arm in [('always_base',0),('cup',1),('direct_score_and_state_only',3),('shuffled',4)]:
        difference=aipw[:,2]-aipw[:,arm]
        comparisons[name]=dict(mean=difference.mean(0).tolist(),
            group90=[clustered_interval(difference[:,i],data['group'],14753) for i in range(4)],
            episode90=[clustered_interval(difference[:,i],data['episode'],14754) for i in range(4)],
            raw_ht_difference=(ht[:,2]-ht[:,arm]).mean(0).tolist())
    nulls={}
    for name,arm,left,right in [('cm',2,(2,3,4),(5,6,7)),('direct',3,(8,9,10),(11,12,13))]:
        available=all(np.isin(data['allocation'],slots).any() for slots in (left,right))
        contrast=duplicate_contrast(data['allocation'],data['outcomes'],data['mu'],arm,left,right)
        nulls[name]=dict(available=bool(available),mean=contrast.mean(0).tolist(),slots=[list(left),list(right)])
    noise=max(.05,*(2*abs(v['mean'][1]) for v in nulls.values()))
    cm=data['assignment']==2;changed=cm&data['actual_pd_difference']
    coverage=dict(detection_rows=detections,binding_windows=n,binding_fraction=n/detections if detections else 0.,
        actual_cm_windows=int(cm.sum()),actual_cm_different_direct_windows=int(changed.sum()),
        actual_cm_different_direct_steps=int(data['actual_pd_changed_steps'][cm].sum()),
        scope='PD alternative on the same actual observations; not unexecuted physical outcomes')
    primary=comparisons['direct_score_and_state_only'];mean=primary['mean']
    gate=dict(support=bool(adequate),null_available=all(v['available'] for v in nulls.values()),
        contact_loss_five_pp=mean[1]<=-.05,contact_loss_exceeds_noise=-mean[1]>=noise,
        contact_loss_group90=primary['group90'][1] is not None and primary['group90'][1][1]<0,
        contact_loss_episode90=primary['episode90'][1] is not None and primary['episode90'][1][1]<0,
        height_point=mean[0]>=-2.,height_group90=primary['group90'][0] is not None and primary['group90'][0][0]>=-5.,
        height_episode90=primary['episode90'][0] is not None and primary['episode90'][0][0]>=-5.,
        geometric_loss_point=mean[2]<=.02,joint_presence_point=mean[3]>=-.02,
        actual_cm_changed12=int(changed.sum())>=12,binding_coverage_one_pct=coverage['binding_fraction']>=.01)
    gate['passed']=all(gate.values())
    label='PROMISING' if gate['passed'] else 'UNPROMISING' if adequate and gate['null_available'] else 'UNCLEAR'
    fit=data['bucket']<50;test=~fit
    fixed_values={str(a):float(aipw[fit,a,0].mean()) for a in (0,1)} if fit.any() else {}
    fixed=max((0,1),key=lambda a:fixed_values[str(a)]) if fit.any() and all(((data['assignment']==a)&fit).any() for a in (0,1)) else None
    fixed_result=None
    if fixed is not None and test.any():
        difference=aipw[test,2]-aipw[test,fixed]
        fixed_result=dict(reference=fixed,fit_values_mm=fixed_values,rows=int(test.sum()),mean=difference.mean(0).tolist(),
            group90=[clustered_interval(difference[:,i],data['group'][test],14753) for i in range(4)],
            scope='fixed arm selected on fit only; report outside fit only')
    ep,epcounts=np.unique(data['episode'],return_counts=True)
    return dict(label=label,counts=counts,policy_support=supports,coverage=coverage,reference_null=nulls,
        required_contact_loss_reduction=noise,primary_gate=gate,comparisons=comparisons,
        aipw_arm_values=aipw.mean(0).tolist(),raw_ht_arm_values=ht.mean(0).tolist(),best_fixed=fixed_result,
        outcome_columns=['supported_change_mm','any_contact_loss','geometric_loss','terminal_joint_presence'],
        initially_clear_support={str(a):support(data,a,data['initial']) for a in range(5)},
        replanning=dict(episodes_with_multiple_windows=int((epcounts>=2).sum()),subsequent_windows=int(np.maximum(epcounts-1,0).sum())),
        full_control_utility_proven=False,stable_grasp_or_learned_policy_utility_proven=False)


def run(args):
    begin=time.monotonic()
    if args.output.exists():raise ValueError('unique analysis output')
    data,detections,manifest,hashes=load(args.source)
    result=summarize(data,detections)
    for path in (Path(__file__),ROOT/'scripts/contact_risk_intervention_statistics.py',ROOT/'scripts/analyze_orientation_feedback_opportunity.py',ROOT/'scripts/analyze_structured_contact_opportunity.py'):
        hashes[str(path.resolve())]=sha(path)
    result.update(experiment_id='P-20261002-contact-risk-interventions',run_status='COMPLETED',input_sha256=hashes,
        source_cumulative_seconds=manifest['cumulative_seconds'],elapsed_seconds=time.monotonic()-begin,
        estimator='fixed pre-allocation frozen Cm AIPW; raw HT parallel',device='cpu',
        device_reason='raw force/geometry/PD arithmetic and known-propensity statistics; no neural computation',
        scope='randomized H10 contact-risk intervention effect conditional on current binding over a mixed behavior distribution; no exact-state counterfactual, full policy or final stable utility')
    if result['elapsed_seconds']+manifest['cumulative_seconds']>3600:raise TimeoutError('fixed whole slot bound')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
