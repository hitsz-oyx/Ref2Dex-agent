"""Matched GT-tau R-only objective repair; no WAM fitting or native execution."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam.rl_robot import RLRobotWindows
from ptv3_shared_wam.robot import features, ACTIVE
from ptv3_shared_wam.checkpoint import atomic_json, git_commit
from trajectory_policy.chunk_retargeter import ChunkRetargeter
from trajectory_policy.chunk_objective import control_aware_loss


def pack(dataset, indices):
    result = dict(state=np.empty((len(indices),87),np.float32),
                  tau=np.empty((len(indices),24,33),np.float32),
                  action=np.empty((len(indices),8,12),np.float32),
                  startup=np.empty(len(indices),bool), near=np.empty(len(indices),bool))
    keys = np.asarray([dataset.keys[int(i)] for i in indices])
    for seq in np.unique(keys[:,1]):
        rows = np.flatnonzero(keys[:,1]==seq); ticks=keys[rows,0]
        d,c = dataset.sequence(int(seq))
        state,tau=features(*(d[k][ticks] for k in ('q','dq','hand','obj','velocity')),
                           d['hand'][ticks[:,None]+np.arange(1,25)])
        result['state'][rows],result['tau'][rows]=state,tau
        result['action'][rows]=d['applied'][ticks[:,None]+np.arange(8)][...,list(ACTIVE)]
        result['startup'][rows]=ticks<16
        local=state[:,36:69].reshape(-1,11,3)
        for begin in range(0,len(rows),32):
            delta=local[begin:begin+32,:,None]-c['points'][None,None]
            gap=np.sqrt((delta**2).sum(-1)).min((1,2))
            result['near'][rows[begin:begin+32]]=gap<=.02
    return result


def objective(prediction,target,scale,pd_scale,arm):
    if arm=='A_all8':
        return ((prediction-target)/scale).abs().mean()
    if arm=='B_control':
        return control_aware_loss(prediction,target,scale,pd_scale)[0]
    error=((prediction-target[:,:1])/scale).abs().mean()
    delta=(prediction[:,0]-target[:,0])*pd_scale
    physical=sum(delta[:,a:b].abs().mean()/unit for a,b,unit in
                 ((0,3,.01),(3,6,.1),(6,12,.1)))/3
    return error+physical


@torch.no_grad()
def evaluate(net,panel,pd_scale):
    net.eval();pred=[]
    for begin in range(0,len(panel['state']),128):
        pred.append(net(panel['state'][begin:begin+128],panel['tau'][begin:begin+128]).cpu().numpy())
    pred=np.concatenate(pred);target=panel['action'].cpu().numpy()
    delta=(pred[:,0].clip(-1,1)-target[:,0])*pd_scale.cpu().numpy()
    xyz=np.linalg.norm(delta[:,:3],axis=-1)*1000
    rotation=np.abs(delta[:,3:6]).max(-1);finger=np.abs(delta[:,6:]).max(-1)
    result=dict(xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(np.percentile(xyz,95)),
                rotation_p95_rad=float(np.percentile(rotation,95)),finger_p95_rad=float(np.percentile(finger,95)),
                clipping=float((np.abs(pred[:,0])>1).any(-1).mean()),
                raw_first_MAE=float(np.abs(pred[:,0]-target[:,0]).mean()))
    for name,mask in [('startup',panel['startup']),('later',~panel['startup']),
                      ('near_surface20mm',panel['near']),('far_surface20mm',~panel['near'])]:
        result[name]=dict(count=int(mask.sum()),xyz_mean_mm=float(xyz[mask].mean()) if mask.any() else None,
                         xyz_p95_mm=float(np.percentile(xyz[mask],95)) if mask.any() else None)
    if pred.shape[1]>1:
        result['tail_standardized_L1']=float((np.abs(pred[:,1:]-target[:,1:])/net.action_scale[1:].cpu().numpy()).mean())
    net.train();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--card',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=140);p.add_argument('--draw-seed',type=int,default=141)
    p.add_argument('--panel-seed',type=int,default=142);p.add_argument('--seconds',type=int,default=1200)
    a=p.parse_args();out=a.output.resolve();source=a.source.resolve();parent=a.parent.resolve()
    if out.exists() or out.parent!=source.parent or not 1<=a.updates<=3000 or not 1<=a.seconds<=1200:
        raise ValueError('fresh sibling Task output and bounded updates/deadline required')
    import yaml
    card=yaml.safe_load(a.card.read_text().split('---',2)[1])
    if card['task']!='ptv3-shared-wam' or any(s not in card['seeds'] for s in [a.seed,a.draw_seed,a.panel_seed]):
        raise ValueError('assigned Task-local experiment seeds required')
    if a.device=='cpu' and 'CPU' not in a.card.read_text():
        raise ValueError('record a concrete CPU reason in the card')
    if a.device=='cuda':
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        import subprocess
        if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    out.mkdir(parents=True);start=time.monotonic()
    def deadline(*_):raise TimeoutError('fixed R-only probe deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(a.seconds)
    manifest=dict(status='RUNNING',experiment_id=card['experiment_id'],run_id=out.name,git_commit=git_commit(),
                  parent=str(parent),source=str(source),device=a.device,updates=a.updates,batch=128,
                  seeds=[a.seed,a.draw_seed,a.panel_seed],threads=4,arm_order=['A_all8','B_control','C_first'],
                  CPU_reason='six project GPUs occupied; bounded small R diagnostic avoids interrupting ongoing runs')
    atomic_json(out/'manifest.json',manifest)
    try:
        packet=torch.load(parent,map_location='cpu',weights_only=False)
        base=ChunkRetargeter(128)
        base.load_state_dict({k[len('retargeter.'):]:v for k,v in packet['model'].items() if k.startswith('retargeter.')})
        pd_scale=torch.tensor(packet['config']['native_PD_scale'],device=a.device)
        manifest.update(parent_step=packet['step'],parent_git_commit=packet['provenance']['git_commit'])
        del packet
        train,dev=RLRobotWindows(source,'train'),RLRobotWindows(source,'val')
        pool_indices=train.sample_indices(np.random.default_rng(a.draw_seed),8192)
        train_arrays=pack(train,pool_indices)
        val_indices=np.load(a.validation);dev_arrays=pack(dev,val_indices)
        panel_ids=np.random.default_rng(a.panel_seed).choice(8192,512,replace=False)
        panels={'TRAIN':{k:v[panel_ids] for k,v in train_arrays.items()},'dev':dev_arrays}
        for panel in [train_arrays,*panels.values()]:
            for key in ('state','tau','action'):panel[key]=torch.tensor(panel[key],device=a.device)
        manifest.update(TRAIN_windows=len(train),dev_windows=len(dev),fit_pool=8192,
                        fit_unique=len(set(pool_indices)),report_TRAIN=512,report_dev=len(val_indices),
                        normalization='unchanged parent TRAIN-only normalizers',
                        grouping='near means hand-point to sampled CAD surface <=20mm, not contact force')
        atomic_json(out/'manifest.json',manifest)
        np.savez_compressed(out/'panels.npz',fit_indices=np.asarray(pool_indices),TRAIN_report_ids=panel_ids,dev_indices=val_indices)
        results={}
        for arm in manifest['arm_order']:
            net=copy.deepcopy(base)
            if arm=='C_first':
                net.action_queries=torch.nn.Parameter(net.action_queries[:1].detach().clone())
                net.action_mean=net.action_mean[:1].clone();net.action_scale=net.action_scale[:1].clone()
            net=net.to(a.device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
            rng=np.random.default_rng(a.draw_seed);curve=[];best=None;best_key=None;arm_start=time.monotonic()
            for step in range(a.updates+1):
                if step:
                    ids=torch.tensor(rng.integers(8192,size=128),device=a.device)
                    opt.zero_grad(set_to_none=True)
                    prediction=net(train_arrays['state'][ids],train_arrays['tau'][ids])
                    loss=objective(prediction,train_arrays['action'][ids],net.action_scale,pd_scale,arm)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite R loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    if not torch.isfinite(norm):raise FloatingPointError('nonfinite R gradients')
                    opt.step()
                if step%500==0 or step==a.updates:
                    row=dict(step=step,panels={k:evaluate(net,v,pd_scale) for k,v in panels.items()},
                             elapsed_s=time.monotonic()-arm_start)
                    curve.append(row);key=(row['panels']['dev']['xyz_p95_mm'],row['panels']['dev']['raw_first_MAE'])
                    if best_key is None or key<best_key:
                        best_key=key;best=copy.deepcopy(row)
                        torch.save(dict(schema='ref2dex.rl660-R-only.v1',arm=arm,width=128,model=net.cpu().state_dict(),
                                        step=step,parent=str(parent),native_PD_scale=pd_scale.cpu(),
                                        scope='offline GT-tau only; no autonomous grasp claim'),out/(arm+'-step%d.pt'%step))
                        alias=out/(arm+'_best.pt');temporary=out/(arm+'.alias-part')
                        temporary.symlink_to(arm+'-step%d.pt'%step);temporary.replace(alias)
                        net.to(a.device)
                    atomic_json(out/(arm+'-curve.json'),curve)
                    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>100_000_000:
                        raise RuntimeError('R-only artifact budget100MB exceeded')
                    print(json.dumps(dict(arm=arm,**row)),flush=True)
                if step and step%100==0:
                    elapsed=time.monotonic()-arm_start
                    atomic_json(out/'progress.json',dict(arm=arm,step=step,elapsed_s=time.monotonic()-start,
                                seconds_per_update=elapsed/step,loss=float(loss),
                                remaining_updates=(len(manifest['arm_order'])-manifest['arm_order'].index(arm))*a.updates-step))
            results[arm]=dict(initial=curve[0],selected=best,final=curve[-1],selected_by='dev first xyz p95, then raw first MAE')
            atomic_json(out/'partial_result.json',results)
        atomic_json(out/'result.json',dict(status='COMPLETED',arms=results,elapsed_s=time.monotonic()-start,
                    scope='single matched warm-start R-only Probe; objective and selection not a grasp result'))
        manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-start)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error));raise
    finally:
        signal.alarm(0);atomic_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
