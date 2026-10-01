#!/usr/bin/env python3
"""Frozen prospective local-policy contrasts; no fitting or threshold search."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from src.task.CmResidual.trajectory_selector import effective_commands,recommendation_probability
from src.task.CmResidual.contact_trajectory import trajectory_targets


def cluster_interval(values,groups,z=1.6448536269514722):
    values=np.asarray(values,dtype=np.float64);mean=float(values.mean()) if len(values) else None
    if not len(values):return dict(value=None,groups=0,interval90=None)
    unique,inverse=np.unique(np.asarray(groups),return_inverse=True);g=len(unique)
    if g<2:return dict(value=mean,groups=g,interval90=None)
    sums=np.bincount(inverse,weights=values-mean)
    se=float(np.sqrt(g/(g-1)*np.square(sums).sum())/len(values))
    return dict(value=mean,groups=g,se=se,interval90=[mean-z*se,mean+z*se])


def contrast_support(matches,commands,episodes,control):
    different=~(commands[:,0]==commands[:,control]).all(-1)
    result=dict(different_command_windows=int(different.sum()))
    for name,index in [('cm',0),('control',control)]:
        rows=different&matches[:,index]
        result[name]=dict(windows=int(rows.sum()),episodes=len(set(np.asarray(episodes)[rows.numpy()].tolist())))
    result['sufficient']=all(result[k]['windows']>=32 and result[k]['episodes']>=15 for k in ['cm','control'])
    return result


def run(args):
    torch.set_num_threads(2)
    manifest=json.loads((args.run/'run_manifest.json').read_text())
    if manifest['run_status']!='COMPLETED' or len(manifest['phases'])!=6:raise ValueError('six terminal phases required')
    records=[];hashes={}
    for phase in manifest['phases']:
        p=Path(phase['directory'])/'records.pt';hashes[str(p)]=sha(p)
        if hashes[str(p)]!=phase['record_sha256']:raise ValueError('record drift')
        records.append(torch.load(p,map_location='cpu',weights_only=False))
    keys=['candidate_actions','assignment','propensity','state','history','rest_z','future_state','future_contact','actual_action',
          'candidate_pd_targets','actual_pd_targets','effective_candidate_commands','motion_id','start_frame','env_id']
    data={k:torch.cat([r[k] for r in records]) for k in keys}
    trace={k:torch.cat([r['policy_trace'][k] for r in records]) for k in records[0]['policy_trace']}
    episodes=[e for r in records for e in r['episode_id']];frames=[f'{int(m)}/{int(s)}' for m,s in zip(data['motion_id'],data['start_frame'])]
    n=len(episodes);rows=torch.arange(n);pool=trace['policy_proposals'].long();assignment=data['assignment']
    if any(r['schema']!='ref2dex.selector_pool_contact_consequence.v1' or r['future_done'].any() for r in records):raise ValueError('invalid prospective window')
    if not torch.equal(assignment,pool[rows,trace['selected_policy'].long()]):raise ValueError('allocation mismatch')
    if not torch.equal(data['propensity'],recommendation_probability(data['candidate_actions'],pool,assignment)):raise ValueError('propensity mismatch')
    if not torch.equal(data['history'][:,-1,:49],data['state']):raise ValueError('pre-state mismatch')
    if not torch.equal(data['effective_candidate_commands'],effective_commands(data['candidate_actions'])):raise ValueError('effective command mismatch')
    selected=data['candidate_actions'][rows,assignment]
    if not torch.equal(data['actual_action'][:,:2],selected[:,None].expand(-1,2,-1)):raise ValueError('execution mismatch')
    if not torch.equal(data['actual_pd_targets'][:,0],data['candidate_pd_targets'][rows,assignment]):raise ValueError('native target mismatch')
    if not all(torch.isfinite(data[k]).all() for k in keys):raise ValueError('nonfinite actual record')
    commands=data['effective_candidate_commands'].gather(1,pool[:,:,None].expand(-1,-1,12))
    chosen=data['effective_candidate_commands'][rows,assignment]
    matches=(commands==chosen[:,None]).all(-1);p=data['propensity']
    labels=trajectory_targets(data['future_state'],data['future_contact'].all(-1),data['state'][:,38],data['rest_z'])
    def report(v,mask=None):
        mask=torch.ones(n,dtype=torch.bool) if mask is None else mask
        return dict(episode=cluster_interval(v[mask].numpy(),np.asarray(episodes)[mask.numpy()]),
                    frame=cluster_interval(v[mask].numpy(),np.asarray(frames)[mask.numpy()]))
    names=records[0]['policy_names'];contrasts={};policy_values={}
    for index,name in enumerate(names):
        policy_values[name]={k:report(matches[:,index].float()*labels[k]/p) for k in ['supported_change_mm','joint_contact','release']}
    for index,name in enumerate(names[1:],1):
        weight=(matches[:,0].float()-matches[:,index].float())/p
        contrasts[name]={k:report(weight*labels[k]) for k in ['supported_change_mm','joint_contact','release']}
        contrasts[name]['initially_lifted_release']=report(weight*labels['release'],labels['initially_lifted'])
        contrasts[name]['support']=contrast_support(matches,commands,episodes,index)
        different=~(commands[:,0]==commands[:,index]).all(-1)
        contrasts[name]['release_observed_support']={}
        for arm_name,arm_index in [('cm',0),('control',index)]:
            mask=different&matches[:,arm_index]
            lifted=mask&labels['initially_lifted']
            contrasts[name]['release_observed_support'][arm_name]=dict(windows=int(mask.sum()),events=int(labels['release'][mask].sum()),
                initially_lifted_windows=int(lifted.sum()),initially_lifted_events=int(labels['release'][lifted].sum()))
    active=~(commands[:,0]==commands[:,3]).all(-1)
    executed=~(chosen==commands[:,3]).all(-1)
    native_delta=data['candidate_pd_targets'][rows,pool[:,0]]-data['candidate_pd_targets'][:,4]
    latency=[v['milliseconds'] for r in records for v in r['inference_latency']]
    base=contrasts['always_base'];support=all(v['support']['sufficient'] for v in contrasts.values())
    def upper(report):return report['frame']['interval90'][1] if report['frame']['interval90'] else float('inf')
    gain=all(v['supported_change_mm']['frame']['value']>=.5 for v in contrasts.values())
    gate=dict(support=support,gain_over_four_controls=gain,
              base_gain_lower_positive=bool(base['supported_change_mm']['frame']['interval90'] and base['supported_change_mm']['frame']['interval90'][0]>0),
              all_release_upper_le_2pp=upper(base['release'])<=.02,
              initially_lifted_release_upper_le_5pp=upper(base['initially_lifted_release'])<=.05,
              joint_contact_loss_le_2pp=base['joint_contact']['frame']['value']>=-.02,
              physical_proposal_coverage_5_to_80pct=.05<=float(active.float().mean())<=.8)
    gate['passed']=all(gate.values());label='UNCLEAR' if not support else ('PROMISING' if gate['passed'] else 'UNPROMISING')
    output=dict(experiment_id=manifest['experiment_id'],run_status='COMPLETED',label=label,gate=gate,rows=n,episodes=len(set(episodes)),
                frame_groups=len(set(frames)),initially_lifted_windows=int(labels['initially_lifted'].sum()),release_events=int(labels['release'].sum()),
                physical_proposal_windows=int(active.sum()),physical_proposal_coverage=float(active.float().mean()),
                actual_nonbase_command_windows=int(executed.sum()),actual_nonbase_command_coverage=float(executed.float().mean()),
                executed_cm_nonbase_matches=int((active&matches[:,0]).sum()),policy_assignment_counts=torch.bincount(trace['selected_policy'].long(),minlength=5).tolist(),
                native_cm_vs_base_delta=dict(wrist_translation_l2_median_m=float(native_delta[active,:3].norm(dim=-1).median()) if active.any() else None,
                    wrist_rotation_l2_median_rad=float(native_delta[active,3:6].norm(dim=-1).median()) if active.any() else None,
                    finger_l2_median_rad=float(native_delta[active,6:].norm(dim=-1).median()) if active.any() else None),
                replanned_episodes=sum(episodes.count(e)>1 for e in set(episodes)),latency_ms=dict(median=float(np.median(latency)),p90=float(np.quantile(latency,.9)),max=max(latency)),
                policy_values=policy_values,contrasts=contrasts,checkpoint_sha256=records[0]['ranker_sha256'],record_sha256=hashes,
                scope='prospective mixed-history local randomized common-physical-action IPW; descriptive normal cluster90 intervals, zero observed release is not zero risk; no pure-policy/stable-grasp claim; state_only six slots retain expert identity',
                inputs_unchanged=all(sha(Path(k))==v for k,v in manifest['input_sha256'].items()),
                elapsed_seconds_including_setup=manifest['cumulative_seconds'],output_bytes_including_setup=manifest['output_bytes'])
    if not output['inputs_unchanged']:raise ValueError('collection source/input drift')
    if args.output.exists():raise ValueError('analysis exists')
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:output[k] for k in ['label','rows','gate','contrasts']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
