"""Replay measured history, generated plans, native input/command and holding."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src')]
from consequence_evaluator.data import sha
from consequence_evaluator.proposal_history import condition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists() or ROOT/'outputs/consequence-evaluator' not in a.output.parents:
        raise ValueError('fresh task-owned audit required')
    m=json.loads((a.evaluation/'manifest.json').read_text())
    result=json.loads((a.evaluation/'result.json').read_text())
    if m['status']!='COMPLETED' or not m.get('generated_proposal'):
        raise ValueError('completed generated-tau execution required')
    with np.load(a.evaluation/'trajectory.npz',allow_pickle=False) as s:data={k:s[k] for k in s.files}
    with np.load(a.evaluation/'plans.npz',allow_pickle=False) as s:plans={k:s[k] for k in s.files}
    diagnostic=bool(m.get('interface_diagnostic'))
    controls=128 if diagnostic else 542
    if data['action'].shape!=(controls,16,18) or plans['ticks'].tolist()!=list(range(0,controls,8)):
        raise ValueError('complete trajectory and every8step plan required')
    if not all(np.isfinite(v).all() for v in list(data.values())+list(plans.values())):
        raise ValueError('nonfinite saved data')
    for path,digest in m['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('execution input drift: '+path)
    references=[Path(v) for v in m['input_sha256'] if v.endswith('.pkl')]
    if len(references)!=1:raise ValueError('unique independent reference required')
    with references[0].open('rb') as s:ref=pickle.load(s)
    roles=np.asarray(m['roles']);envs=plans['envs']
    expected_roles= {'tau_gt','tau_online','displacement','displacement_handoff'} if diagnostic else {'tau_gt','persistence','displacement','scored'}
    if set(roles)!=expected_roles or any(sum(roles==r)!=4 for r in set(roles)):
        raise ValueError('four predeclared roles required')
    if not np.array_equal(envs,np.flatnonzero(roles!='tau_gt')):raise ValueError('plan role identity mismatch')
    initial={k:bool(np.array_equal(data[k][0],np.broadcast_to(ref[k][0,0],data[k][0].shape)))
        for k in ('object_pose','hand_keypoints','dof_position','dof_velocity')}
    if not all(initial.values()):raise ValueError('initial measured state mismatch')
    history_error=0.;current_error=0.
    for j,tick in enumerate(plans['ticks']):
        indices=np.maximum(np.arange(tick-3,tick+1),0)
        for i,env in enumerate(envs):
            h,current=condition(data['object_pose'][indices,env],data['hand_keypoints'][indices,env],
                data['dof_position'][indices,env],data['dof_velocity'][indices,env],data['object_velocity'][indices,env])
            history_error=max(history_error,float(np.max(np.abs(h-plans['history'][j,i]))))
            current_error=max(current_error,float(np.max(np.abs(current-plans['current'][j,i]))))
    if history_error>3e-6 or current_error>3e-6:raise ValueError('actual measured history mismatch')
    choices=plans['choices']
    for i,env in enumerate(envs):
        expected=0 if roles[env] in ('persistence','tau_online') else (1 if roles[env] in ('displacement','displacement_handoff') else plans['score'][:,i].argmax(-1))
        if not np.all(choices[:,i]==expected):raise ValueError('actual generated choice mismatch')
    if plans['score'].shape!=(len(plans['ticks']),12,10) or np.any(choices<0) or np.any(choices>=10):
        raise ValueError('only ten generated alternatives admitted')
    privileged_error=0.
    if diagnostic:
        mask=roles[envs]=='tau_online'
        if not np.array_equal(plans['privileged_online_tau'],np.broadcast_to(mask,(len(plans['ticks']),12))):
            raise ValueError('diagnostic privilege identity mismatch')
        for j,tick in enumerate(plans['ticks']):
            pose=data['object_pose'][tick,envs[mask]]
            actual=np.einsum('nij,ntpj->ntpi',pose[:,:3,:3],plans['raw'][j,mask])+pose[:,None,None,:3,3]
            target=ref['hand_keypoints'][np.minimum(tick+np.arange(1,25),542),0]
            privileged_error=max(privileged_error,float(np.max(np.abs(actual-target[None]))))
        if privileged_error>3e-6:raise ValueError('GT online control is not the actual reference hand tau')
    with np.load(Path(m['geometry_reference'])/'geometry.npz',allow_pickle=False) as s:base=s['q']
    dt=m['wrist_feedforward_contract']['control_dt']
    ratio=np.asarray(m['wrist_feedforward_contract']['damping'],np.float32)/np.asarray(m['wrist_feedforward_contract']['stiffness'],np.float32)
    def velocity(q):
        d=np.diff(q,axis=-2).copy();d[...,3:6]=np.arctan2(np.sin(d[...,3:6]),np.cos(d[...,3:6]))
        return np.concatenate((d[...,:1,:],(d[...,:-1,:]+d[...,1:,:])/2,d[...,-1:,:]),axis=-2)/dt
    next_q=np.broadcast_to(base[1:controls+1,None],(controls,16,18)).copy()
    next_v=np.broadcast_to(velocity(base)[1:controls+1,None],next_q.shape).copy()
    future=np.stack([ref['hand_keypoints'][np.minimum(t+np.arange(1,25),542),0] for t in range(controls)])
    future=np.broadcast_to(future[:,None],(controls,16,24,11,3)).copy()
    pv=velocity(plans['q'])
    for t in range(controls):
        j=t//8;off=t%8
        enabled=np.asarray([not(diagnostic and roles[e]=='displacement_handoff' and t<8) for e in envs])
        ids=envs[enabled]
        next_q[t,ids]=plans['q'][j,enabled,off+1]
        next_v[t,ids]=pv[j,enabled,off+1]
        future[t,ids]=plans['hand'][j,:,np.minimum(np.arange(24)+off,23)].transpose(1,0,2,3)[enabled]
    # Independent numpy reconstruction of the *actual*897D controller input.
    obj=data['object_pose'][:-1];rot=obj[...,:3,:3];center=obj[...,:3,3]
    hand=(data['hand_keypoints'][:-1]-center[...,None,:])@rot
    local_future=(future-center[...,None,None,:])@rot[...,None,:,:]
    qe=next_q-data['dof_position'][:-1];qe[...,3:6]=np.arctan2(np.sin(qe[...,3:6]),np.cos(qe[...,3:6]))
    previous=np.concatenate((np.zeros((1,16,12),np.float32),np.tanh(data['latent'][:-1])))
    x=np.concatenate((data['dof_position'][:-1],data['dof_velocity'][:-1]/8,hand.reshape(controls,16,33)/.1,
        local_future.reshape(controls,16,792)/.1,qe,previous,data['object_velocity'][:-1]/2),-1).clip(-20,20)
    feature_error=float(np.max(np.abs(x-data['student_features'])))
    if feature_error>2e-5:raise ValueError('actual student input differs from declared plan')
    offset=np.asarray(m['native_controller']['offset'],np.float32);scale=np.asarray(m['native_controller']['scale'],np.float32)
    target=next_q.copy();target[...,:6]+=next_v[...,:6]*ratio
    active=[0,1,2,3,4,5,6,8,10,12,14,15]
    limits=np.asarray([.06,.06,.06,.6,.6,.6,.5,.5,.5,.5,.35,.35],np.float32)
    target[...,active]+=np.tanh(data['latent'])*limits
    intended=(target-offset)/scale
    intended[...,:6]=(target[...,:6]-data['dof_position'][:-1,:,:6]-offset[:6])/scale[:6]
    intended[...,6:]=2*intended[...,6:]-1;intended[...,[7,9,11,13,16,17]]=0
    command_error=float(np.max(np.abs(intended.clip(-1,1)-data['action'])))
    if command_error>2e-6:raise ValueError('generated q/residual/feedforward command mismatch')
    pd=data['action'].copy();pd[...,6:]=(1+pd[...,6:])/2;pd=offset+pd*scale
    pd[...,:6]+=data['dof_position'][:-1,:,:6]
    for d,p,r in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):pd[...,d]=pd[...,p]*r
    pd_error=float(np.max(np.abs(pd-data['pd_targets'])))
    if pd_error>1e-6:raise ValueError('native adapter PD reconstruction mismatch')
    height=data['object_pose'][:,:,2,3]-data['object_pose'][0,:,2,3]
    supported=data['table_footprint']&(np.abs(data['support_gap'])<=.02)
    held=(data['surface_gap']<=.01)&~supported&(height>=.03);held[0]=False
    from consequence_evaluator.gate1 import episode_outcome
    for row in result['outcomes']:
        outcome=episode_outcome({k:data[k][:,row['env']] for k in ('object_pose','surface_gap','support_gap','table_footprint','object_velocity')})
        if outcome!= {k:row[k] for k in outcome}:raise ValueError('saved outcome reconstruction mismatch')
    summary={}
    for r in sorted(set(roles)):
        rows=[v for v in result['outcomes'] if v['role']==r];mask=roles==r
        qualifying=sum(v['maximum_held_frames']>=433 and held[-1,v['env']] for v in rows)
        clip=float(data['clipped'][:,mask].mean())
        summary[r]=dict(episodes=4,long_held_terminal=int(qualifying),held45=sum(v['maximum_held_frames']>=45 for v in rows),
            terminal_held=int(held[-1,mask].sum()),median_maximum_held=float(np.median([v['maximum_held_frames'] for v in rows])),
            maximum_lift_m=float(height[:,mask].max()),clipping_rate=clip,
            raw_screen=bool(qualifying>=3 and clip<.01))
    calibration=summary['tau_gt']['held45']>=3 if diagnostic else summary['tau_gt']['long_held_terminal']>=3
    for r in summary:
        screen=(summary[r]['held45']>=3 and summary[r]['clipping_rate']<.01) if diagnostic else summary[r]['raw_screen']
        summary[r]['status']='UNCLEAR' if not calibration else ('PROMISING' if screen else 'UNPROMISING')
    generated_roles=[r for r in sorted(set(roles)) if r!='tau_gt']
    status='UNCLEAR' if not calibration else (summary['tau_online']['status'] if diagnostic else
        ('PROMISING' if any(summary[r]['raw_screen'] for r in generated_roles) else 'UNPROMISING'))
    audit=dict(status=status,interface_diagnostic=diagnostic,maximum_privileged_tau_error=privileged_error,
        calibration_pass=calibration,summary=summary,initial_state_exact=initial,
        maximum_history_error=history_error,maximum_current_hand_error=current_error,maximum_student_feature_error=feature_error,
        maximum_command_error=command_error,maximum_pd_error=pd_error,
        generated_choice_counts={r:np.bincount(choices[:,roles[envs]==r].reshape(-1),minlength=10).tolist() for r in generated_roles},
        projection_median_s=float(np.median(plans['projection_s'])),projection_max_s=float(plans['projection_s'].max()),
        scorer_benefit='UNCLEAR: separate live roles, no same-state counterfactual outcomes',
        claim='Privileged interface diagnostic,128steps only; not pure-H deployment' if diagnostic else
              'Single-motion execution Probe only; not Cm benefit or formal method refutation')
    a.output.mkdir(parents=True)
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    files=[a.evaluation/n for n in ('manifest.json','result.json','trajectory.npz','plans.npz')]+[Path(__file__).resolve()]
    (a.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256={str(v.resolve()):sha(v) for v in files}),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for r in ['tau_gt']+generated_roles:
        mask=roles==r
        axes[0].plot(np.median(height[:,mask],1),label=r)
        axes[1].plot(held[:,mask].mean(1),label=r)
    axes[0].set_ylabel('Object lift (m)');axes[0].legend(ncol=4)
    axes[1].set_ylabel('Held fraction');axes[1].set_xlabel('Control tick')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Frozen generated-tau execution: separate live roles, single-seed Probe')
    fig.tight_layout();fig.savefig(a.output/'behavior.png',dpi=160);plt.close(fig)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
