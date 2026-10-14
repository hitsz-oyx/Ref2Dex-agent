"""Matched all8 R-only Probe: native commands versus query-relative targets."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from ptv3_shared_wam.action_targets import NATIVE,PD,GEOMETRY,REPRESENTATIONS,R_ONLY_SCHEMA,encode_targets,freeze_chunk,consume_target
from ptv3_shared_wam.rl_robot import RLRobotWindows
from ptv3_shared_wam.checkpoint import atomic_json,git_commit
from trajectory_policy.chunk_retargeter import ChunkRetargeter
from trajectory_policy.pd_retargeter import pd_from_action
from trajectory_policy.retargeter_data import ACTIVE
from r_only_data import pack


def physical_pack(dataset,indices):
    values=pack(dataset,indices);keys=np.asarray([dataset.keys[int(i)] for i in indices]);n=len(indices)
    values.update(pd=np.empty((n,8,18),np.float32),future_q=np.empty((n,8,18),np.float32))
    for seq in np.unique(keys[:,1]):
        rows=np.flatnonzero(keys[:,1]==seq);ticks=keys[rows,0];d,_=dataset.sequence(int(seq))
        for dest,source in [('pd','pd_targets'),('future_q','q')]:
            values[dest][rows]=d[source][ticks[:,None]+np.arange(8)]
    return values


@torch.no_grad()
def metrics(net,panel,representation,controller):
    net.eval();pred=torch.cat([net(panel['state'][i:i+128],panel['tau'][i:i+128]) for i in range(0,len(panel['state']),128)])
    frozen=freeze_chunk(pred,panel['state'][:,:18],representation)
    _,applied,clipped=consume_target(frozen,panel['future_q'],representation,controller)
    actual=pd_from_action(applied,panel['future_q'],pred.new_tensor(controller['offset']),pred.new_tensor(controller['scale']))
    difference=(actual-panel['pd'])[...,list(ACTIVE)].cpu().numpy()
    xyz=np.linalg.norm(difference[...,:3],axis=-1)*1000;tick=panel['tick']
    result=dict(xyz_mean_mm=float(xyz[:,0].mean()),xyz_p95_mm=float(np.percentile(xyz[:,0],95)),
                chunk_xyz_mean_mm=float(xyz.mean()),chunk_xyz_p95_mm=float(np.percentile(xyz,95)),
                rotation_p95_rad=float(np.percentile(np.abs(difference[:,0,3:6]).max(-1),95)),
                finger_p95_rad=float(np.percentile(np.abs(difference[:,0,6:]).max(-1),95)),
                clipping=float(clipped[:,0].float().mean()),chunk_clipping=float(clipped.float().mean()),
                standardized_L1=float(((pred-panel['target'])/net.action_scale).abs().mean()))
    for name,mask in [('startup',tick<16),('later',tick>=16)]:
        errors=xyz[mask,0];result[name]=dict(count=len(errors),mean_mm=float(errors.mean()) if len(errors) else None,
                                             p95_mm=float(np.percentile(errors,95)) if len(errors) else None)
    net.train();return result,pred.cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--card',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5);a=p.parse_args();out=a.output.resolve()
    if out.exists() or out.parent!=ROOT/'outputs/ptv3-shared-wam':raise ValueError('fresh Task run required')
    import yaml
    card=yaml.safe_load(a.card.read_text().split('---',2)[1])
    if card['task']!=TASK.name or any(s not in card['seeds'] for s in [144,145,146]):raise ValueError('assigned seeds required')
    from qualify_R_after_training import choose_idle_gpu
    memories={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    pids={g:list(map(int,subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split())) for g in memories}
    owners={pid:Path('/proc',str(pid)).stat().st_uid for consumers in pids.values() for pid in consumers}
    if choose_idle_gpu(memories,pids,owners,os.getuid())!=a.gpu:raise RuntimeError('idle GPU and conservative remote-four quota required')
    shared=json.loads((ROOT/'outputs/ptv3-shared-wam-v2/pretrain-main3-remote-80k-20261014-r1/aggregate_budget_status.json').read_text())
    if time.time()-shared['checked_at']>300 or shared['combined_bytes']+150_000_000>=393_000_000_000:raise RuntimeError('shared budget margin unavailable')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);torch.set_num_threads(4);torch.manual_seed(144)
    out.mkdir();start=time.monotonic()
    def timeout(*_):raise TimeoutError('1800-second matched representation budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    parent=torch.load(a.parent,map_location='cpu',weights_only=False);source=Path(parent['provenance']['source'])
    controller=parent['config']['native_controller'];dev_ids=np.load(a.parent.parent/'validation.npy')
    initial={k[len('retargeter.'):]:v for k,v in parent['model'].items() if k.startswith('retargeter.')};del parent
    manifest=dict(status='RUNNING',git_commit=git_commit(),source=str(source.resolve()),parent=str(a.parent.resolve()),
                  experiment_id=card['experiment_id'],run_id=out.name,seeds=[144,145,146],representations=list(REPRESENTATIONS),
                  normalization='TRAIN-only per-horizon-coordinate.v1',input_normalization='unchanged parent TRAIN state/tau',
                  device='cuda',physical_gpu=a.gpu,updates=2000,batch=128,optimizer='AdamW3e-4 wd.01 clip1',
                  objective='all8 standardized L1, coefficients one',initialization='identical random trainable parameters; representation-specific output statistics',
                  max_seconds=1800,artifact_budget_bytes=150_000_000,native_controller=controller,
                  scope='GT tau R-only Probe; no joint WAM update or physics conclusion')
    atomic_json(out/'manifest.json',manifest)
    try:
        train,dev=RLRobotWindows(source,'train'),RLRobotWindows(source,'val')
        full=physical_pack(train,np.arange(len(train)));dev_np=physical_pack(dev,dev_ids)
        train_ids=np.random.default_rng(146).choice(len(train),512,replace=False)
        panels_np=dict(TRAIN={k:v[train_ids] for k,v in full.items()},dev=dev_np)
        np.savez_compressed(out/'panels.npz',TRAIN_indices=train_ids,dev_indices=dev_ids)
        base=ChunkRetargeter(128)
        for name in ('state_mean','state_scale','tau_mean','tau_scale'):getattr(base,name).copy_(initial[name])
        del initial
        shared_data={k:torch.as_tensor(v,device='cuda') for k,v in full.items() if k in ('state','tau','future_q','pd')}
        results={};manifest['packing_seconds']=time.monotonic()-start
        gpu_rows=[];roundtrips={}
        for arm,representation in [('native',NATIVE),('PD',PD),('geometry',GEOMETRY)]:
            labels=encode_targets(full['pd'],full['state'][:,:18],representation,full['action'])
            labels=torch.as_tensor(labels,device='cuda');net=copy.deepcopy(base).cuda()
            net.action_mean.copy_(labels.mean(0));net.action_scale.copy_(labels.std(0,unbiased=False).clamp_min(1e-6))
            optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
            panels={}
            for name,v in panels_np.items():
                panels[name]={k:(value if k=='tick' else torch.as_tensor(value,device='cuda')) for k,value in v.items()}
                panels[name]['target']=torch.as_tensor(encode_targets(v['pd'],v['state'][:,:18],representation,v['action']),device='cuda')
            # Exact-label roundtrip separates an encoding bug/ambiguity from fit error.
            v=shared_data;frozen=freeze_chunk(labels,v['state'][:,:18],representation)
            _,applied,clipped=consume_target(frozen,v['future_q'],representation,controller)
            exact=pd_from_action(applied,v['future_q'],applied.new_tensor(controller['offset']),applied.new_tensor(controller['scale']))
            roundtrips[arm]=dict(max_PD_error=float((exact-v['pd']).abs().max()),clipping=float(clipped.float().mean()),scope='all admitted TRAIN windows, all8 targets')
            if arm!='geometry' and roundtrips[arm]['max_PD_error']>2e-6:raise ValueError('label PD roundtrip failed')
            rng=np.random.default_rng(145);curve=[];best=None;best_key=None;arm_start=time.monotonic()
            for step in range(2001):
                if step:
                    ids=torch.as_tensor(train.sample_indices(rng,128),device='cuda');optimizer.zero_grad(set_to_none=True)
                    prediction=net(shared_data['state'][ids],shared_data['tau'][ids])
                    loss=((prediction-labels[ids])/net.action_scale).abs().mean();loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite matched R update')
                    optimizer.step()
                if step%500==0:
                    row=dict(step=step,panels={},elapsed_s=time.monotonic()-arm_start);predictions={}
                    for name,v in panels.items():row['panels'][name],predictions[name]=metrics(net,v,representation,controller)
                    curve.append(row);key=(row['panels']['dev']['xyz_p95_mm'],row['panels']['dev']['standardized_L1'])
                    if best_key is None or key<best_key:
                        best_key=key;best=copy.deepcopy(row);filename=arm+'-step%d.pt'%step
                        torch.save(dict(schema=R_ONLY_SCHEMA,width=128,model={k:v.detach().cpu() for k,v in net.state_dict().items()},
                               arm=arm,step=step,action_representation=representation,native_controller=controller,
                               native_PD_scale=torch.tensor(controller['scale'])[list(ACTIVE)]*torch.tensor([1.]*6+[.5]*6)),out/filename)
                        alias=out/(arm+'.alias-part');alias.symlink_to(filename);alias.replace(out/(arm+'_best.pt'))
                        np.savez_compressed(out/(arm+'-selected-predictions.npz'),**predictions)
                    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                    gpu_rows.append(dict(arm=arm,step=step,gpu=gpu,elapsed_s=time.monotonic()-start))
                    atomic_json(out/'gpu.json',gpu_rows);atomic_json(out/(arm+'-curve.json'),curve)
                    remaining=(time.monotonic()-arm_start)/max(step,1)*(2000-step+(2-['native','PD','geometry'].index(arm))*2000) if step else None
                    atomic_json(out/'progress.json',dict(arm=arm,step=step,elapsed_s=time.monotonic()-start,eta_s=remaining,gpu=gpu))
                    print(json.dumps(dict(arm=arm,**row)),flush=True)
                    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>150_000_000:raise RuntimeError('150MB artifact budget')
                    consumers=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                    if consumers-{os.getpid()}:raise RuntimeError('foreign GPU process')
            results[arm]=dict(initial=curve[0],selected=best,final=curve[-1]);atomic_json(out/'partial_result.json',results)
            del net,optimizer,labels,panels
        atomic_json(out/'result.json',dict(status='COMPLETED',arms=results,roundtrips=roundtrips,elapsed_s=time.monotonic()-start))
        manifest.update(status='COMPLETED',roundtrips=roundtrips,geometry_native_roundtrip_pass=roundtrips['geometry']['max_PD_error']<2e-6 and roundtrips['geometry']['clipping']==0,
                        elapsed_s=time.monotonic()-start)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error));raise
    finally:
        signal.alarm(0);atomic_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
