"""Fixed 256-window first-control fit; preserves the parent's full query state."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_rl660_R import pack, evaluate, ChunkRetargeter, RLRobotWindows, atomic_json, git_commit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();baseline=a.baseline.resolve();out=a.output.resolve()
    if out.exists() or out.parent!=baseline.parent:raise ValueError('fresh sibling output required')
    torch.set_num_threads(4);torch.manual_seed(140)
    def timeout(*_):raise TimeoutError('fixed300-second memorization budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    out.mkdir();start=time.monotonic();manifest=dict(status='RUNNING',git_commit=git_commit(),device='cpu',
        baseline=str(baseline),updates=2000,windows=256,batch=64,threads=4,execution_steps=1,
        scope='local fit diagnostic; not a pure comparison to one-query C or a grasp result')
    atomic_json(out/'manifest.json',manifest)
    try:
        original=json.loads((baseline/'manifest.json').read_text())
        source=Path(original['source']);train=RLRobotWindows(source,'train');dev=RLRobotWindows(source,'val')
        with np.load(baseline/'panels.npz') as f:
            fit_indices=f['fit_indices'][f['TRAIN_report_ids'][:256]];dev_indices=f['dev_indices'].copy()
        panels=dict(TRAIN=pack(train,fit_indices),dev=pack(dev,dev_indices))
        np.savez(out/'panels.npz',TRAIN_indices=fit_indices,dev_indices=dev_indices)
        for panel in panels.values():
            for key in ('state','tau','action'):panel[key]=torch.tensor(panel[key])
        initial=torch.load(baseline/'A_all8-step0.pt',map_location='cpu',weights_only=False)
        net=ChunkRetargeter(128);net.load_state_dict(initial['model'])
        pd_scale=initial['native_PD_scale'].cpu();optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
        rng=np.random.default_rng(141);curve=[]
        for step in range(2001):
            if step:
                ids=rng.integers(256,size=64);b=panels['TRAIN'];optimizer.zero_grad(set_to_none=True)
                pred=net(b['state'][ids],b['tau'][ids])[:,0];target=b['action'][ids,0]
                first=((pred-target)/net.action_scale[0]).abs().mean();delta=(pred-target)*pd_scale
                pd0=sum(delta[:,l:r].abs().mean()/unit for l,r,unit in ((0,3,.01),(3,6,.1),(6,12,.1)))/3
                loss=first+pd0;loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite fit')
                optimizer.step()
            if step%500==0:
                row=dict(step=step,panels={k:evaluate(net,v,pd_scale) for k,v in panels.items()},elapsed_s=time.monotonic()-start)
                curve.append(row);atomic_json(out/'curve.json',curve);print(json.dumps(row),flush=True)
        torch.save(dict(schema='ref2dex.rl660-R-only.v1',arm='first-only-memorization256',model=net.state_dict(),
                        width=128,step=2000,native_PD_scale=pd_scale,scope=manifest['scope']),out/'step2000.pt')
        atomic_json(out/'result.json',dict(status='COMPLETED',initial=curve[0],final=curve[-1],elapsed_s=time.monotonic()-start))
        manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-start)
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e));raise
    finally:
        signal.alarm(0);atomic_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
