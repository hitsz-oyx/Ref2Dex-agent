#!/usr/bin/env python3
"""Independent force labels and scalar propensity arithmetic on accepted data."""
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


def run(args):
    begin=time.monotonic()
    if args.output.exists():raise ValueError('unique independent audit')
    import torch
    torch.set_num_threads(2)
    expected=json.loads(args.results.read_text());manifest=json.loads((args.source/'run_manifest.json').read_text())
    if expected['run_status']!='COMPLETED' or manifest['run_status']!='COMPLETED':raise ValueError('terminal analysis/source')
    for k,v in expected['input_sha256'].items():
        if sha(Path(k))!=v:raise ValueError('analysis input hash')
    rows=[];groups=[];episodes=[];allocation=[];assignments=[];detections=0;label_error=0.
    for phase in manifest['phases']:
        record=torch.load(Path(phase['directory'])/'records.pt',map_location='cpu',weights_only=False)
        detections+=record['shadow_detection_rows']
        raw_hand=record['future_hand_force'].double().square().sum(-1).sqrt().amax(-1)
        raw_object=record['future_object_force'].double().square().sum(-1).sqrt()
        weight=record['mass_kg'].double()[:,None]*record['gravity_magnitude']
        contact=(raw_hand/weight>.1)&(raw_object/weight>.1)
        if not torch.equal(contact,record['future_contact'].all(-1)):raise ValueError('raw force presence labels')
        clear=record['future_clearance']>=.002
        support=contact[:,-3:].all(-1)&clear[:,-3:].all(-1)
        height=((record['future_state'][:,-3:,38].double().amin(-1)-record['rest_z'].double()).clamp_min(0)*support-
            (record['state'][:,38].double()-record['rest_z'].double()).clamp_min(0))*1000
        if height.numel():label_error=max(label_error,float((height-record['outcome']['supported_change_mm']).abs().max()))
        for index in range(len(record['state'])):
            assignment=int(record['assignment'][index]);slot=int(record['allocation'][index])
            mapped=0 if slot==0 else 1 if slot==1 else 2 if slot<8 else 3 if slot<14 else 4
            if assignment!=mapped:raise ValueError('actual random allocation')
            p=(1/16,1/16,6/16,6/16,2/16)[assignment]
            if abs(float(record['propensity'][index])-p)>1e-7:raise ValueError('merged propensity')
            seen=bool(record['initial_clearance'][index]>=.002);drop=False
            for tick in range(10):
                current=bool(clear[index,tick]);drop|=seen and not current;seen|=current
            truth=np.array([float(height[index]),float(not bool(contact[index].all())),float(drop),float(contact[index,-3:].all())])
            mu=record['nuisance_forecasts'][index].double().mean(0)[:5].numpy()
            corrected=mu.copy();corrected[assignment]+=(truth-mu[assignment])/p
            raw=np.zeros((5,4));raw[assignment]=truth/p
            rows.append((corrected,raw,truth,mu));allocation.append(slot);assignments.append(assignment)
            groups.append(f"{int(record['motion_id'][index])}/{int(record['start_frame'][index])}");episodes.append(record['episode_id'][index])
    if label_error>1e-4:raise ValueError('independent supported height label')
    max_error=0.;classification_exact=True
    if rows:
        augmented=np.stack([r[0] for r in rows]);raw=np.stack([r[1] for r in rows])
        for actual,wanted in ((augmented.mean(0),expected['aipw_arm_values']),(raw.mean(0),expected['raw_ht_arm_values'])):
            max_error=max(max_error,float(np.max(np.abs(actual-np.asarray(wanted)))))
        for name,arm in [('always_base',0),('cup',1),('direct_score_and_state_only',3),('shuffled',4)]:
            difference=augmented[:,2]-augmented[:,arm];saved=expected['comparisons'][name]
            max_error=max(max_error,float(np.max(np.abs(difference.mean(0)-saved['mean']))))
            for metric in range(4):
                for key,identities,seed in [('group90',groups,14753),('episode90',episodes,14754)]:
                    interval=clustered_interval(difference[:,metric],np.array(identities),seed)
                    if interval is None:
                        if saved[key][metric] is not None:raise ValueError('cluster interval availability')
                    else:max_error=max(max_error,float(np.max(np.abs(np.array(interval)-saved[key][metric]))))
        null_values={}
        for name,arm,left,right in [('cm',2,(2,3,4),(5,6,7)),('direct',3,(8,9,10),(11,12,13))]:
            accum=np.zeros(4)
            for slot,row in zip(allocation,rows):
                weight=(16/3 if slot in left else 0.)-(16/3 if slot in right else 0.)
                accum+=weight*(row[2]-row[3][arm])
            null_values[name]=accum/len(rows)
            max_error=max(max_error,float(np.max(np.abs(null_values[name]-expected['reference_null'][name]['mean']))))
        noise=max(.05,2*abs(null_values['cm'][1]),2*abs(null_values['direct'][1]))
        max_error=max(max_error,abs(noise-expected['required_contact_loss_reduction']))
        mean=augmented[:,2].mean(0)-augmented[:,3].mean(0)
        group_ci=[clustered_interval((augmented[:,2]-augmented[:,3])[:,i],np.array(groups),14753) for i in range(4)]
        episode_ci=[clustered_interval((augmented[:,2]-augmented[:,3])[:,i],np.array(episodes),14754) for i in range(4)]
        assignment=np.array(assignments);support=expected['policy_support']
        for arm in range(5):
            indices=np.flatnonzero(assignment==arm)
            if support[str(arm)]!=dict(rows=len(indices),episodes=len(set(episodes[i] for i in indices)),groups=len(set(groups[i] for i in indices))):
                raise ValueError('independent actual support')
        adequate=len(rows)>=64 and len(set(episodes))>=24 and len(set(groups))>=8
        adequate &= all(support[str(a)]['rows']>=24 and support[str(a)]['episodes']>=12 and support[str(a)]['groups']>=8 for a in (2,3))
        adequate &= support['0']['rows']>=4 and support['1']['rows']>=4 and support['4']['rows']>=8
        null_available=all(any(slot in left for slot in allocation) and any(slot in right for slot in allocation)
            for left,right in [((2,3,4),(5,6,7)),((8,9,10),(11,12,13))])
        gate=dict(support=bool(adequate),null_available=null_available,contact_loss_five_pp=bool(mean[1]<=-.05),
            contact_loss_exceeds_noise=bool(-mean[1]>=noise),contact_loss_group90=group_ci[1] is not None and group_ci[1][1]<0,
            contact_loss_episode90=episode_ci[1] is not None and episode_ci[1][1]<0,height_point=bool(mean[0]>=-2),
            height_group90=group_ci[0] is not None and group_ci[0][0]>=-5,height_episode90=episode_ci[0] is not None and episode_ci[0][0]>=-5,
            geometric_loss_point=bool(mean[2]<=.02),joint_presence_point=bool(mean[3]>=-.02),
            actual_cm_changed12=expected['coverage']['actual_cm_different_direct_windows']>=12,
            binding_coverage_one_pct=len(rows)/detections>=.01)
        gate['passed']=all(gate.values())
        label='PROMISING' if gate['passed'] else 'UNPROMISING' if adequate and null_available else 'UNCLEAR'
        classification_exact=gate==expected['primary_gate'] and label==expected['label']
    else:
        classification_exact=expected['label']=='UNCLEAR' and expected['rows']==0
    if max_error>1e-4 or not classification_exact:raise ValueError('independent statistics or original gates failed')
    result=dict(run_status='COMPLETED',audit_passed=True,rows=len(rows),shadow_detection_rows=detections,
        raw_supported_label_max_error_mm=label_error,statistics_max_error=max_error,classification_and_gates_exact=classification_exact,
        result_sha256=sha(args.results),auditor_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-begin,
        scope='independent raw force/clearance labels, scalar propensity corrections, cluster intervals, duplicate null and primary gates; physical PD/fullNN in accepted source audits')
    if manifest['cumulative_seconds']+expected['elapsed_seconds']+result['elapsed_seconds']>3600:raise TimeoutError('fixed complete slot')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
