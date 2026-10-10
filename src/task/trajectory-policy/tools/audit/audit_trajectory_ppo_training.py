"""Replay saved on-policy density, reward, history, credit and native execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.resolve().parents:
        raise ValueError('fresh task-owned output required')
    started = time.monotonic()
    manifest = json.loads((args.training/'manifest.json').read_text())
    result = json.loads((args.training/'result.json').read_text())
    if manifest['status'] != 'COMPLETED' or manifest['engineering_smoke'] or manifest['updates'] != 24 or manifest['matmul_precision'] != 'highest':
        raise ValueError('full completed training required')
    for path, digest in manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('source/input drift: '+path)
    if sha(args.training/'final.pt') != manifest['final_checkpoint_sha256']:
        raise ValueError('final checkpoint drift')
    with np.load(args.training/'low.npz') as f:
        low = {key:f[key] for key in f.files}
    with np.load(args.training/'high.npz') as f:
        high = {key:f[key] for key in f.files}
    if high['history'].shape != (384, 16, 328) or low['action'].shape[1:] != (16, 18):
        raise ValueError('full saved rollout required')
    if not all(np.isfinite(v).all() for v in list(low.values())+list(high.values())):
        raise ValueError('nonfinite trace')
    state = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    util, memory = map(int,state.split(','))
    if util > 10 or memory > 512:
        raise ValueError('GPU not idle: '+state)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from trajectory_policy.actor import load_actor
    from trajectory_policy.ppo import HistoryValue
    from consequence_evaluator.tau_tracking import TauTracker
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    errors = {}
    def error(key, a, b, limit):
        delta = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        errors[key] = max(errors.get(key,0.),delta)
        if delta > limit:
            raise ValueError(key+' mismatch: '+str(delta))
    def rebuild_history(indices):
        values = []
        for row in range(16):
            obj, hand, q, dq, vel = (low[key][indices,row] for key in ('obj','hand','q','dq','velocity'))
            rot, center = obj[-1,:3,:3],obj[-1,:3,3]
            local_hand=(hand-center)@rot
            fingers=local_hand[:,1:]-local_hand[:,:1]
            local_obj=np.linalg.inv(obj[-1])[None]@obj
            local_vel=np.concatenate((vel[:,:3]@rot,vel[:,3:]@rot),-1)
            wrist=np.concatenate((dq[:,:3]@rot,dq[:,3:6]),-1)
            values.append(np.concatenate((local_hand[:,0].reshape(-1),fingers.reshape(-1),local_obj[:,:3].reshape(-1),
                q[:,6:].reshape(-1),dq[:,6:].reshape(-1),local_vel.reshape(-1),wrist.reshape(-1),obj[-1,2:3,3],-rot[2])))
        return np.asarray(values,np.float32)
    first = None
    for u in range(24):
        packet=torch.load(args.training/'actors'/('u%04d.pt'%u),map_location='cuda',weights_only=False)
        actor=load_actor(packet,'cuda').eval()
        value=HistoryValue(actor.history_mean,actor.history_scale).to('cuda').eval()
        value.load_state_dict(packet['value_model'])
        if first is None:
            first={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        for j in range(u*16,(u+1)*16):
            index=int(high['low_index'][j]); start=int(low['episode_start'][index])
            h=rebuild_history(np.maximum(index+np.arange(-3,1),start))
            error('measured_history',h,high['history'][j],2e-5)
            with torch.no_grad():
                distribution=actor.distribution(torch.as_tensor(h,device='cuda'))
                error('behavior_mean',distribution.loc.cpu(),high['mean'][j],2e-5)
                error('behavior_std',distribution.scale.cpu(),high['std'][j],1e-7)
                c=torch.as_tensor(high['c'][j],device='cuda')
                error('behavior_logprob',distribution.log_prob(c).sum(-1).cpu(),high['logp'][j],.002)
                error('value',value(torch.as_tensor(h,device='cuda')).cpu(),high['value'][j],2e-5)
            duration=int(high['duration'][j,0])
            terminal=int(low['episode_tick'][index])+duration==542
            if not np.all(high['duration'][j]==duration) or duration != min(8,542-int(low['episode_tick'][index])) or not np.all(high['done'][j]==terminal):
                raise ValueError('executed prefix/terminal differs')
            error('discounted_reward',sum(.99**k*low['reward'][index+k] for k in range(duration)),high['reward'][j],2e-5)
            if int(high['update'][j]) != u or int(high['chunk'][j]) != j%16:
                raise ValueError('update/chunk differs')
        with torch.no_grad():
            hb=torch.as_tensor(high['bootstrap_history'][u*16],device='cuda')
            error('bootstrap_value',value(hb).cpu(),high['bootstrap'][u*16],2e-5)
        end=int(high['low_index'][(u+1)*16-1])+int(high['duration'][(u+1)*16-1,0])
        if end<len(low['q']):
            start=int(low['episode_start'][end])
            error('bootstrap_history',rebuild_history(np.maximum(end+np.arange(-3,1),start)),high['bootstrap_history'][u*16],2e-5)
        v=high['value'][u*16:(u+1)*16].astype(np.float64)
        r=high['reward'][u*16:(u+1)*16].astype(np.float64)
        done=high['done'][u*16:(u+1)*16]
        duration=high['duration'][u*16:(u+1)*16]
        a=np.zeros_like(v);carry=np.zeros(16)
        for k in range(15,-1,-1):
            nxt=high['bootstrap'][u*16] if k==15 else v[k+1]
            carry=r[k]+.99**duration[k]*(~done[k])*nxt-v[k]+(.99*.95)**duration[k]*(~done[k])*carry
            a[k]=carry
        error('advantage',a,high['advantage'][u*16:(u+1)*16],3e-5)
        error('returns',a+v,high['returns'][u*16:(u+1)*16],3e-5)
    final=load_actor(torch.load(args.training/'final.pt',map_location='cuda',weights_only=False),'cuda')
    change=np.sqrt(sum(float((v.detach().cpu()-first[k]).square().sum()) for k,v in final.state_dict().items() if k.startswith('mean_net.')))
    error('parameter_change',change,result['actor_parameter_change'],1e-7)
    if change<=0:
        raise ValueError('actor did not learn')
    for key in ('history_mean','history_scale','action_mean','action_scale'):
        if not torch.equal(final.state_dict()[key].cpu(),first[key]):
            raise ValueError('normalization drift')
    supported=low['table_footprint']&(abs(low['support_gap'])<=.02)
    lift=low['next_obj'][:,:,2,3]-low['initial_z']
    held=(low['surface_gap']<=.01)&~supported&(lift>=.03)
    prior=np.concatenate((np.zeros((1,16),bool),held[:-1]))
    reset=low['episode_tick']==0;prior[reset]=False
    action_prior=np.concatenate((np.zeros((1,16,18),np.float32),low['action'][:-1]));action_prior[reset]=0
    clipped=(abs(low['intended']-low['action']).max(-1)>1e-6)
    if not np.array_equal(held,low['held']) or not np.array_equal(clipped,low['clipped']):
        raise ValueError('held/clipping events differ')
    terms=dict(proximity=.2*np.exp(-2*np.maximum(low['surface_gap'],0)),lift=.5*np.clip(lift/.2,0,1)*(low['surface_gap']<=.01),
        held=held.astype(np.float32),loss=-.5*(prior&~held).astype(np.float32),clipping=-.1*clipped.astype(np.float32),
        action_rate=-.01*np.mean((low['action']-action_prior)**2,-1))
    for key,val in terms.items():
        error('reward_'+key,val,low['reward_'+key],2e-6)
    error('reward',sum(terms.values()),low['reward'],2e-6)
    control=manifest['native_controller'];offset=np.asarray(control['offset'],np.float32);scale=np.asarray(control['scale'],np.float32)
    policy=TauTracker().to('cuda').eval()
    path=next(Path(p) for p in manifest['input_sha256'] if p.endswith('/ref7_4-tau-finger-fit-20261010-r3/final.pt'))
    policy.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['state_dict'])
    previous=np.concatenate((np.zeros((1,16,12),np.float32),np.tanh(low['latent'][:-1])));previous[reset]=0
    for j in range(384):
        index=int(high['low_index'][j]);duration=int(high['duration'][j,0]);s=slice(index,index+duration)
        future=np.stack([high['hand'][j][:,np.minimum(np.arange(24)+slot,23)] for slot in range(duration)])
        nq=high['q'][j,:,1:duration+1].transpose(1,0,2)
        nv=high['velocity'][j,:,1:duration+1].transpose(1,0,2)
        rot=low['obj'][s,:,:3,:3];center=low['obj'][s,:,:3,3]
        lh=(low['hand'][s]-center[...,None,:])@rot
        lf=(future-center[...,None,None,:])@rot[...,None,:,:]
        qe=nq-low['q'][s];qe[...,3:6]=np.arctan2(np.sin(qe[...,3:6]),np.cos(qe[...,3:6]))
        x=np.concatenate((low['q'][s],low['dq'][s]/8,lh.reshape(duration,16,33)/.1,lf.reshape(duration,16,792)/.1,qe,previous[s],low['velocity'][s]/2),-1).clip(-20,20)
        error('executor_inputs',x,low['features'][s],2e-5)
        with torch.no_grad():
            predicted=policy.actor(torch.as_tensor(low['features'][s].reshape(-1,897),device='cuda')).reshape(duration,16,12).cpu().numpy()
        error('executor_latent',predicted,low['latent'][s],2e-5)
        target=nq.copy();target[...,:6]+=nv[...,:6]*np.asarray(control['gain_ratio'],np.float32)
        active=[0,1,2,3,4,5,6,8,10,12,14,15]
        target[...,active]+=np.tanh(low['latent'][s])*np.asarray([.06]*3+[.6]*3+[.5]*4+[.35]*2,np.float32)
        intended=(target-offset)/scale;intended[...,:6]=(target[...,:6]-low['q'][s,:,:6]-offset[:6])/scale[:6]
        intended[...,6:]=intended[...,6:]*2-1;intended[...,[7,9,11,13,16,17]]=0
        error('intended',intended,low['intended'][s],2e-5)
    if not np.array_equal(low['action'],low['applied']):
        raise ValueError('requested/applied differs')
    error('command',np.clip(low['intended'],-1,1),low['action'],1e-7)
    pd=low['action'].copy();pd[...,6:]=(pd[...,6:]+1)/2;pd=offset+scale*pd;pd[...,:6]+=low['q'][...,:6]
    for distal,parent,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        pd[...,distal]=pd[...,parent]*ratio
    error('pd',pd,low['pd_targets'],1e-6)
    metrics=[json.loads(line) for line in (args.training/'monitor.jsonl').read_text().splitlines()]
    if len(metrics)!=24 or any(m['joint_action_kl']>.020001 for m in metrics):
        raise ValueError('optimization trust budget differs')
    if int(held.sum())!=result['held_steps'] or len(low['q'])*16!=result['environment_interactions']:
        raise ValueError('interaction/event counts differ')
    report=dict(status='COMPLETED',maximum_errors=errors,actor_parameter_change=change,held_steps=int(held.sum()),
        environment_interactions=len(low['q'])*16,elapsed_s=time.monotonic()-started,
        audited='All measured H, pre-update Gaussian density/value, bootstrap inputs/value, variable-duration reward/GAE, task events, frozen executor inputs/output and requested/applied/native PD; no physics rerun')
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    files=[args.training/name for name in ('manifest.json','result.json','high.npz','low.npz','final.pt','monitor.jsonl')]+[Path(__file__)]+list((args.training/'actors').glob('*.pt'))
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',physical_gpu=args.gpu,input_sha256={str(p.resolve()):sha(p) for p in files}),indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    def deadline(signum,frame):
        raise TimeoutError('training replay exceeded60s')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(60)
    main()
