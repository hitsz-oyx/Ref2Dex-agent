#!/usr/bin/env python3
"""Frozen IPW actual local controller utility gates; no per-state oracle."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from analyze_executable_contact_opportunity import clustered_interval,support


def run(args):
    begin=time.monotonic();import torch
    torch.set_num_threads(2);m=json.loads((args.run/'run_manifest.json').read_text());audit=json.loads(args.audit.read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only'] or len(m['phases'])!=12 or audit['run_status']!='COMPLETED' or args.output.exists():raise ValueError('all fresh science/terminal replay required')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('source/input drift')
    records=[]
    for p in m['phases']:
        path=Path(p['directory'])/'records.pt'
        if sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
        records.append(torch.load(path,map_location='cpu',weights_only=False))
    keys=['assignment','propensity','motion_id','start_frame','program','recommendations','changed_base','changed_fixed','ood']
    b={key:torch.cat([r[key] for r in records]) for key in keys};out={key:torch.cat([r['outcome'][key] for r in records]).numpy() for key in ['score_mm','retained','lost_clearance','joint_last3']}
    ep=np.array(sum([r['episode_id'] for r in records],[]));groups=np.array([f'{int(i)}/{int(j)}' for i,j in zip(b['motion_id'],b['start_frame'])]);assignment=b['assignment'].numpy();p=b['propensity'].numpy();n=len(p)
    supports={name:support(index,assignment,ep,groups) for index,name in enumerate(records[0]['policy_names'])}
    adequate=all(s['matches']>=96 and s['episodes']>=24 and s['initial_groups']>=8 for s in supports.values())
    changed={}
    cm=assignment==0
    for ref in ['base','fixed']:
        mask=cm&b['changed_'+ref].any(-1).numpy();changed[ref]=dict(windows=int(mask.sum()),episodes=len(set(ep[mask])),initial_groups=len(set(groups[mask])),frames=int(b['changed_'+ref][torch.from_numpy(cm)].sum()),window_fraction=float(mask.sum()/cm.sum()) if cm.any() else None)
    physical_gate=all(s['windows']>=48 and s['episodes']>=12 and s['initial_groups']>=8 for s in changed.values())
    contrasts={}
    for index,name in enumerate(records[0]['policy_names'][1:],1):
        weight=(assignment==0)/p-(assignment==index)/p;score=weight*out['score_mm'];risk=weight*out['lost_clearance'];joint=weight*out['joint_last3']
        frame=clustered_interval(score,groups);episode=clustered_interval(score,ep);effect=float(score.mean())
        gate=dict(gain_at_least_2mm=effect>=2,initial_group90_positive=frame is not None and frame[0]>0,episode90_positive=episode is not None and episode[0]>0,geometric_loss_point_bound=float(risk.mean())<=.02,joint_force_point_bound=float(joint.mean())>=-.05)
        gate['passed']=all(gate.values());contrasts[name]=dict(score_uplift_mm=effect,initial_group90=frame,episode90=episode,geometric_loss_difference=float(risk.mean()),joint_force_difference=float(joint.mean()),gate=gate)
    summaries={}
    for index,name in enumerate(records[0]['policy_names']):
        mask=assignment==index;tp=1/p[mask];program=b['program'][torch.from_numpy(mask)];summaries[name]=dict(support=supports[name],ipw_score_mm=float(((assignment==index)/p*out['score_mm']).mean()),observed_score_mm=float(out['score_mm'][mask].mean()),observed_loss_rate=float(out['lost_clearance'][mask].mean()),observed_joint_last3_rate=float(out['joint_last3'][mask].mean()),observed_retained_rate=float(out['retained'][mask].mean()),program_frame_counts=torch.bincount(program.flatten(),minlength=8).tolist(),nonbase_frames=int((program!=4).sum()),physical_changed_base_frames=int(b['changed_base'][torch.from_numpy(mask)].sum()),physical_changed_fixed_frames=int(b['changed_fixed'][torch.from_numpy(mask)].sum()))
    all_utility=all(c['gate']['passed'] for c in contrasts.values());label='PROMISING' if adequate and physical_gate and all_utility else ('UNPROMISING' if adequate else 'UNCLEAR')
    elapsed=time.monotonic()-begin
    smoke_audit=json.loads(args.engineering_audit.read_text());total=m['cumulative_seconds']+audit['elapsed_seconds']+smoke_audit['elapsed_seconds']+elapsed
    result=dict(experiment_id=m['experiment_id'],run_status='COMPLETED',label=label,rows=n,episodes=len(set(ep)),initial_groups=len(set(groups)),support_adequate=adequate,physical_change_gate=physical_gate,actual_cm_changes=changed,contrasts=contrasts,policies=summaries,first_episode_frames=sum(r['first_episode_frames'] for r in records),sim_frames=sum(r['sim_frames'] for r in records),same_actual_state_shadow_agreement={name:float((b['recommendations'][:,:,0]==b['recommendations'][:,:,i]).float().mean()) for i,name in enumerate(records[0]['policy_names']) if i},cm_ood_decisions=int(b['ood'][:,:,0].sum()),cumulative_seconds_including_fit_engineering_audits_analysis=total,bytes_including_fit_engineering=m['output_bytes'],budget_passed=total<=3600 and m['output_bytes']<=8<<30,scope='actual randomized full H10 controller windows on held initial groups; motor-conditioned execute1/reobserve ten times; fixed catalogue physics vs local outcome scoring and controls; not individual regret or learned-policy/final stable grasp utility',claim='C3 OPEN')
    if not result['budget_passed']:raise ValueError('original slot2 budget exceeded')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--engineering-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
