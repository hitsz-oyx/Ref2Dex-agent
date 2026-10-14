"""Fixed compute, fixed initial R: small retained pool versus full TRAIN stream."""
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

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam.rl_robot import RLRobotWindows
from ptv3_shared_wam.checkpoint import atomic_json,git_commit
from trajectory_policy.chunk_retargeter import ChunkRetargeter
from trajectory_policy.chunk_objective import control_aware_loss
from r_only_data import pack,tensors,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--card',type=Path,required=True);a=p.parse_args()
    run=a.baseline.resolve();out=a.output.resolve()
    if out.exists() or out.parent!=run.parent:raise ValueError('fresh sibling Task output required')
    import yaml
    card=yaml.safe_load(a.card.read_text().split('---',2)[1])
    if card['task']!='ptv3-shared-wam' or any(s not in card['seeds'] for s in [140,141,142]):raise ValueError('assigned Task seeds required')
    torch.set_num_threads(4);torch.manual_seed(140);out.mkdir();start=time.monotonic()
    def timeout(*_):raise TimeoutError('600-second coverage comparison deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    original=json.loads((run/'manifest.json').read_text());source=Path(original['source'])
    manifest=dict(status='RUNNING',git_commit=git_commit(),source=str(source),parent=str((run/'B_control_best.pt').resolve()),
                  experiment_id=card['experiment_id'],run_id=out.name,updates=2000,batch=128,device='cpu',threads=4,
                  CPU_reason='six GPU budget occupied; bounded R-only comparison',seeds=[140,141,142],
                  variable='fixed8192 pool vs original full TRAIN sampler',normalization='unchanged TRAIN-only parent',
                  loss='4 first + tail + PD0',optimizer='fresh AdamW3e-4 wd.01 clip1')
    atomic_json(out/'manifest.json',manifest)
    try:
        train,dev=RLRobotWindows(source,'train'),RLRobotWindows(source,'val')
        full=pack(train,np.arange(len(train)))
        with np.load(run/'panels.npz') as f:
            fit_ids=f['fit_indices'].copy();report_ids=fit_ids[f['TRAIN_report_ids']];dev_ids=f['dev_indices'].copy()
        outside=np.setdiff1d(np.arange(len(train)),np.unique(fit_ids))
        outside_ids=np.random.default_rng(142).choice(outside,512,replace=False)
        pool=tensors({k:v[fit_ids] for k,v in full.items()})
        panels=dict(TRAIN_pool=tensors({k:v[report_ids] for k,v in full.items()}),
                    TRAIN_outside_pool=tensors({k:v[outside_ids] for k,v in full.items()}),dev=tensors(pack(dev,dev_ids)))
        full=tensors(full)
        np.savez_compressed(out/'panels.npz',fit_indices=fit_ids,TRAIN_report_indices=report_ids,TRAIN_outside_indices=outside_ids,dev_indices=dev_ids)
        packet=torch.load(run/'B_control_best.pt',map_location='cpu',weights_only=False)
        base=ChunkRetargeter(128);base.load_state_dict(packet['model']);pdscale=packet['native_PD_scale'].cpu()
        manifest.update(parent_step=packet['step'],full_TRAIN_windows=len(train),fixed_unique=len(np.unique(fit_ids)),packing_seconds=time.monotonic()-start)
        atomic_json(out/'manifest.json',manifest);results={}
        for arm in ['fixed_pool','full_stream']:
            net=copy.deepcopy(base);optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
            rng=np.random.default_rng(141);curve=[];best=None;best_key=None;arm_start=time.monotonic()
            for step in range(2001):
                if step:
                    if arm=='fixed_pool':ids=rng.integers(8192,size=128);data=pool
                    else:ids=train.sample_indices(rng,128);data=full
                    optimizer.zero_grad(set_to_none=True);pred=net(data['state'][ids],data['tau'][ids])
                    loss,_=control_aware_loss(pred,data['action'][ids],net.action_scale,pdscale)
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite coverage fit')
                    optimizer.step()
                if step%500==0:
                    row=dict(step=step,panels={k:evaluate(net,v,pdscale) for k,v in panels.items()},elapsed_s=time.monotonic()-arm_start)
                    curve.append(row);metric=row['panels']['dev'];key=(metric['xyz_p95_mm'],metric['raw_first_MAE'])
                    if best_key is None or key<best_key:
                        best_key=key;best=copy.deepcopy(row);name=arm+'-step%d.pt'%step
                        if (out/name).exists():raise FileExistsError(name)
                        torch.save(dict(schema='ref2dex.rl660-R-only.v1',arm=arm,width=128,step=step,model=net.state_dict(),
                                        native_PD_scale=pdscale,parent=manifest['parent'],scope='offline coverage Probe; physics untested'),out/name)
                        alias=out/(arm+'.alias-part');alias.symlink_to(name);alias.replace(out/(arm+'_best.pt'))
                    atomic_json(out/(arm+'-curve.json'),curve);print(json.dumps(dict(arm=arm,**row)),flush=True)
                    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>50_000_000:raise RuntimeError('50MB artifact budget')
                if step and step%100==0:
                    atomic_json(out/'progress.json',dict(arm=arm,step=step,elapsed_s=time.monotonic()-start,
                                seconds_per_update=(time.monotonic()-arm_start)/step,loss=float(loss.detach())))
            results[arm]=dict(initial=curve[0],selected=best,final=curve[-1]);atomic_json(out/'partial_result.json',results)
        atomic_json(out/'result.json',dict(status='COMPLETED',arms=results,elapsed_s=time.monotonic()-start,
                    scope='single matched warm start; fixed dev reused, no formal generalization or physics claim'))
        manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-start)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error));raise
    finally:
        signal.alarm(0);atomic_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
