"""Freeze the exact wrist baseline; fit only a linear shared-conditioned correction."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--mode',choices=['none','aligned','shuffled'],required=True)
    a=p.parse_args();a.output=a.output.resolve()
    if a.output.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in a.output.parents:raise ValueError('Fresh Task output required')
    if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('Disk reserve')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2')
    import torch
    import numpy as np
    from ptv3_shared_wam_v2.wrist_shared_probe import WristSharedMLP,condition_rows,phase_draw,write_json
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('900s residual budget')));signal.alarm(900)
    a.output.mkdir();started=time.monotonic()
    status=dict(status='RUNNING',pid=os.getpid(),gpu=a.gpu,training=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(a.output/'process.json',status)
    try:
        packet=torch.load(a.baseline,map_location='cpu',weights_only=False)
        assert packet['mode']=='none' and packet['step']==6000
        stats=packet['stats'];base=WristSharedMLP().cuda();base.load_state_dict(packet['model']);base.eval().requires_grad_(False)
        feature_meta=json.loads((a.cache/'manifest.json').read_text());raw_cache=Path(feature_meta['raw_cache'])
        panels=json.loads((raw_cache/'panels.json').read_text());data={}
        for split in ('train','val'):
            d=dict(np.load(raw_cache/(split+'.npz')));z=np.load(a.cache/(split+'-z.npy'))
            z=condition_rows((z-stats['zmean'])/stats['zstd'],a.mode,199+(split=='val'))
            x=torch.tensor((d['x']-stats['xmean'])/stats['xstd'],device='cuda')
            with torch.no_grad():
                original=torch.cat([base.net(x[i:i+1024]) for i in range(0,len(x),1024)])
            data[split]=dict(z=torch.tensor(z,device='cuda'),base=original,
                target=torch.tensor((d['y']-stats['ymean'])/stats['ystd'],device='cuda'),tick=d['tick'])
            if split=='train':draw=phase_draw(d['tick'],d['sequence'],512,np.random.default_rng(197))
        torch.manual_seed(196);head=torch.nn.Linear(2304,6,bias=False).cuda();torch.nn.init.zeros_(head.weight)
        optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,weight_decay=1e-4)
        factor=torch.tensor(stats['ystd']*stats['controller_scale'],device='cuda')
        frozen={k:v.clone() for k,v in base.state_dict().items()}
        write_json(a.output/'manifest.json',dict(schema='ref2dex.frozen-wrist-shared-linear-residual.v1',mode=a.mode,
            baseline=str(a.baseline.resolve()),feature_cache=str(a.cache.resolve()),lr=5e-4,weight_decay=1e-4,
            updates=6000,draw_seed=197,global_batch=512,baseline_frozen=True,shared_frozen=True,bias=False,
            selection='fixed6000; no DEV model selection',git_commit=status['git_commit']))
        @torch.no_grad()
        def evaluate(full=False):
            sets=dict(panels)
            if full:sets.update({s+'_all':dict(split=s,indices=list(range(len(d['base'])))) for s,d in data.items()})
            result={}
            for name,entry in sets.items():
                d=data[entry['split']];ix=np.asarray(entry['indices']);err=((d['base'][ix]+head(d['z'][ix])-d['target'][ix])*factor).cpu().numpy()
                xyz=np.linalg.norm(err[:,:3],axis=-1)*1000;ticks=d['tick'][ix];result[name]={}
                for phase,mask in [('all',np.ones(len(ix),bool)),('startup',ticks<16),('later',ticks>=16)]:
                    if mask.any():result[name][phase]=dict(count=int(mask.sum()),xyz_mean_mm=float(xyz[mask].mean()),xyz_p95_mm=float(np.percentile(xyz[mask],95)))
            return result
        write_json(a.output/'initial-metrics.json',evaluate())
        for step in range(1,6001):
            ix=draw();d=data['train'];loss=(d['base'][ix]+head(d['z'][ix])-d['target'][ix]).abs().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step==1:np.save(a.output/'first-draw.npy',ix)
            if step==1 or step%1000==0:
                row=dict(step=step,loss=float(loss),elapsed_s=time.monotonic()-started,panels=evaluate(step==6000),
                    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
                write_json(a.output/'progress.json',row)
                with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(step=step,DEV=row['panels']['DEV_fixed256']['all'])),flush=True)
        assert all(torch.equal(v,frozen[k]) and v.grad is None for k,v in base.named_parameters())
        assert torch.equal(head(torch.zeros(4,2304,device='cuda')),torch.zeros(4,6,device='cuda'))
        row.update(status='COMPLETED',baseline_parameters_unchanged=True,zero_condition_exactly_restores_baseline=True)
        write_json(a.output/'result.json',row)
        torch.save(dict(schema='ref2dex.frozen-wrist-shared-linear-residual.v1',head=head.state_dict(),stats=stats,baseline=str(a.baseline.resolve()),mode=a.mode),a.output/'final.pt')
        status.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    except BaseException as error:status.update(status='FAILED',error=repr(error));raise
    finally:write_json(a.output/'process.json',status);signal.alarm(0)


if __name__=='__main__':main()
