"""P0: saved-trace chronology, exact delta construction and first physical loss."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.tau_replay_contract import reference_tau,require_saved_clock,pd_targets,first_exceeds,bouts


def load(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def physical_rows(data,points,bounds,limits):
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.measured_execution import online_support
    observer=FeedbackObserver(points);rows=[]
    for t in range(len(data['q'])):
        snap={k:v[t] for k,v in data.items() if k not in ('observation','action','tau','tau_valid','raw_action','canonical_action','expanded_action','pd_targets_actual','episodes_input')}
        obj=data['objects'][t,0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
        forces=data['contact_forces'][t,0];force_valid=bool(data['contact_force_valid'][t,0]);support=online_support(snap,0,pose,points,bounds)
        fb=observer.update(pose,data['hand'][t,0,NATIVE_TO_WAM],obj[7:13],t/30,contact_forces=forces,force_valid=force_valid,supported=support)
        safety=physical_safety(snap,0,points,bounds,limits)
        reasons=[]
        if obj[2]-data['objects'][0,0,2]<.05:reasons.append('lift_below5cm')
        if not fb.grasp_established:reasons.append('grasp_not_established')
        if fb.grasp_lost:reasons.append('grasp_lost')
        if not force_valid:reasons.append('force_invalid')
        count=int((np.linalg.norm(forces,axis=-1)>=.1).sum())
        if count<2:reasons.append('fewer_than_two_force_contacts')
        if support:reasons.append('table_supported')
        if safety['abort']:reasons.append(safety['reason'])
        rows.append(dict(tick=t,seconds=t/30,held=not reasons,failed_predicates=reasons,force_count=count,
            force_norms_n=np.linalg.norm(forces,axis=-1).tolist(),near_fingers=observer.last_evidence['near_fingers'],
            relative_hand_speed_mps=observer.last_evidence['relative_hand_speed_mps'],object_speed_mps=observer.last_evidence['object_speed_mps'],
            grasp_established=fb.grasp_established,grasp_lost=fb.grasp_lost,motion_coupled=observer.last_evidence['coupled'],tau_valid=bool(data['tau_valid'][t]),previous_control_tau_valid=bool(data['tau_valid'][t-1]) if t else None))
    return rows


def compare(a,b,offset,scale):
    n=min(len(a['q']),len(b['q']));m=min(len(a['action']),len(b['action']))
    q=a['q'][:n,0]-b['q'][:n,0];q[:,3:6]=(q[:,3:6]+np.pi)%(2*np.pi)-np.pi
    series=dict(action=np.abs(a['action'][:m]-b['action'][:m]).max(-1),q=np.abs(q).max(-1),
        dq=np.abs(a['dq'][:n,0]-b['dq'][:n,0]).max(-1),hand=np.linalg.norm(a['hand'][:n,0]-b['hand'][:n,0],axis=-1).max(-1),
        force=np.linalg.norm(a['contact_forces'][:n,0]-b['contact_forces'][:n,0],axis=-1).max(-1),
        tau=np.abs(a['tau'][:n]-b['tau'][:n]).reshape(n,-1).max(-1))
    pa=pd_targets(a['action'][:m],a['q'][:m,0],offset,scale);pb=pd_targets(b['action'][:m],b['q'][:m,0],offset,scale)
    series['pd_translation']=np.abs(pa[:,:3]-pb[:,:3]).max(-1);series['pd_angular']=np.abs(pa[:,3:]-pb[:,3:]).max(-1)
    thresholds=dict(action=1e-5,q=1e-5,dq=1e-4,hand=1e-5,force=1e-3,tau=1e-6,pd_translation=1e-5,pd_angular=1e-5)
    return dict(first_divergence={k:dict(tick=first_exceeds(v,thresholds[k]),threshold=thresholds[k]) for k,v in series.items()},
        first_tick_note='action/PD[t] precedes measured state/force[t+1]; state[t] is pre-action',
        pd_evidence='Reconstructed from saved clipped action/current q, not measured historical PD targets'),series


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[4,5,6],default=5);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);manifest=json.loads((a.run/'manifest.json').read_text());data={r['label']:load(a.run/r['label']/'trajectory.npz') for r in manifest['runs']}
    ref=data['source-pd'];offset=np.asarray(manifest['native_initial_audit']['controller']['offset']);scale=np.asarray(manifest['native_initial_audit']['controller']['scale'])
    source=Path(manifest['source_pd_reference']);points=load(source/'canonical.npz')['points'];side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/manifest['source_sequence'];bounds=load(side/'admission.npz')['table_bounds']
    records=[];all_rows={}
    for entry in manifest['runs']:
        name=entry['label'];d=data[name];require_saved_clock(d);physical=physical_rows(d,points,bounds,manifest['native_initial_audit']['physical_joint_limits']);all_rows[name]=physical
        spans=bouts([r['held'] for r in physical]);first_loss=spans[0][1] if spans and spans[0][1]<len(physical) else None
        contract_errors=[]
        if entry['channel']=='gt':
            for t in range(len(d['q'])):
                tau,valid=reference_tau(ref['hand'][:,0],t)
                if bool(d['tau_valid'][t])!=valid or not np.allclose(d['tau'][t],tau,atol=1e-7,rtol=0):contract_errors.append(t)
        contact_loss=None;grasp_lost=None
        if spans:
            contact_loss=next((r for r in physical[spans[0][0]:] if not r['grasp_established'] or r['force_count']<2),None)
            grasp_lost=next((r for r in physical[spans[0][0]:] if r['grasp_lost']),None)
        comparison,series=compare(d,ref,offset,scale)
        shift=np.linalg.norm(d['hand'][:min(len(d['q']),len(ref['q'])),0]-ref['hand'][:min(len(d['q']),len(ref['q'])),0],axis=-1).mean(-1)
        record=dict(label=name,channel=entry['channel'],state_count=len(d['q']),control_count=len(d['action']),valid_gt_controls=int(d['tau_valid'][:len(d['action'])].sum()),
            hold_intervals=[dict(first_tick=lo,last_held_tick=hi-1,first_unheld_tick=hi if hi<len(physical) else None,elapsed_s=(hi-lo-1)/30) for lo,hi in spans],
            first_hold_loss=None if first_loss is None else physical[first_loss],first_contact_or_grasp_condition_loss=contact_loss,first_grasp_lost_flag=grasp_lost,gt_contract_mismatch_ticks=contract_errors,versus_source=comparison,
            live_reference_hand_drift_at_first_loss_m=float(shift[first_loss]) if first_loss is not None and first_loss<len(shift) else None,
            loss_window=physical[max(0,first_loss-4):first_loss+3] if first_loss is not None else [])
        records.append(record)
        columns=list(series)
        with (out/(name+'-divergence.csv')).open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['tick',*columns,'mean_hand_reference_shift_m'])
            for t in range(len(shift)):writer.writerow([t,*[float(series[k][t]) if t<len(series[k]) else '' for k in columns],float(shift[t])])
    pairs={}
    for left,right in [('baseline-gt-0','baseline-wam'),('calibrated-gt-0','calibrated-gt-1'),('baseline-gt-0','baseline-gt-1')]:pairs[left+' vs '+right]=compare(data[left],data[right],offset,scale)[0]
    # The NN check uses the old saved inputs verbatim; it is not a new fit.
    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if int(gpu.split(',')[1])>64:raise RuntimeError('P0 neural replay GPU occupied: '+gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1');import torch
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    torch.set_num_threads(2);models={}
    for key,path in [('baseline',Path(manifest['base_checkpoint'])),('calibrated',a.run/'calibrated.pt')]:
        model=MeasuredInspirePolicy(with_tau=True).cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models[key]=model.requires_grad_(False)
    x=ref['observation'][:len(ref['action'])].copy()
    for t in range(len(x)):
        tau,valid=reference_tau(ref['hand'][:,0],t);x[t,421:685]=tau.reshape(-1);x[t,685]=float(valid)
    nn={}
    with torch.inference_mode():
        for key,model in models.items():
            predicted=model.actor(torch.tensor(x,device='cuda')).cpu().numpy()
            replay={}
            for name,d in data.items():
                if name=='source-pd' or not name.startswith(key):continue
                y=model.actor(torch.tensor(d['observation'][:len(d['action'])],device='cuda')).clamp(-1,1).cpu().numpy()
                replay[name]=float(np.max(np.abs(y-d['action'])))
            nn[key]=dict(supervised_same_reference_action_rmse=float(np.sqrt(np.mean((predicted-ref['action'])**2))),saved_observation_action_replay_max_errors=replay)
    summary=dict(status='PASS' if not any(r['gt_contract_mismatch_ticks'] for r in records) and all(error<=1e-5 for row in nn.values() for error in row['saved_observation_action_replay_max_errors'].values()) else 'FAIL',source_reference_states=len(ref['q']),records=records,pairwise=pairs,neural_replay=nn,
        conclusions=dict(gt_tau_clocked_reference_delta=True,wam_tau_live_H4_goal_event_conditioned=True,executor_itself_reads_live_state=True,gt_is_not_certified_upper_bound=True),
        limits=['Saved old traces lack actual PD and raw unclipped actor output; PD here is reconstruction.','First measured divergence is temporal order, not causal identification.','Source-index action differences are not optimal live-state action error.','Single repeated sequence, no WAM>GT statistical superiority claim.','Closed-loop force/hold not inferred from supervised action RMSE.'])
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'physical-rows.json').write_text(json.dumps(all_rows)+'\n')
    print(json.dumps(dict(status=summary['status'],source_reference_states=len(ref['q']),records=[{k:r[k] for k in ('label','first_hold_loss','live_reference_hand_drift_at_first_loss_m','gt_contract_mismatch_ticks')} for r in records],neural_replay=nn,pairwise=pairs)))
    return int(summary['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
