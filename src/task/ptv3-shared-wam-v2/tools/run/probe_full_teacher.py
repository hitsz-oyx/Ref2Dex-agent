"""Scratch MLP fits full teacher observations to single-control PD labels; no tau."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[5]
ACTIVE=[0,1,2,3,4,5,6,8,10,12,14,15]
ARMS=['full_absolute','full_relative','current_relative','memory_full_relative']


def write(path,data):
    temp=path.with_suffix('.json.part');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--card',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--updates',type=int,default=12000)
    a=p.parse_args();out=a.output.resolve();cache=a.cache.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:raise ValueError('fresh task output required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('assigned GPU occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(2);torch.manual_seed(180);torch.backends.cuda.matmul.allow_tf32=True
    card=yaml.safe_load(a.card.read_text().split('---',2)[1])
    if card.get('experiment_id')!='P-20261015-full-teacher-input' or card.get('seeds')!=[180,181,162]:raise ValueError('assigned experiment card required')
    meta=json.loads((cache/'manifest.json').read_text())
    if meta.get('status')!='COMPLETED' or meta.get('schema')!='ref2dex.full-teacher-input-cache.v1':raise ValueError('audited cache required')
    import shutil
    if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('20GiB disk reserve')
    out.mkdir(parents=True);started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1200s supervised fitting budget')));signal.alarm(1200)
    relative=a.arm!='full_absolute';current=a.arm=='current_relative';memory=a.arm=='memory_full_relative'
    manifest=dict(status='RUNNING',arm=a.arm,experiment_id=card['experiment_id'],cache=str(cache),physical_gpu=a.gpu,
        pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        updates=a.updates,batch=512,learning_rate=3e-4,weight_decay=.0001,widths=[1442,1024,1024,1024,512,12],
        input='delta0 721 padded to1442' if current else 'teacher delta1+delta16 1442',tau_input=False,
        output='PD minus live q for first6; absolute active fingers' if relative else 'absolute q_PD ACTIVE12',
        q_used_by_network=False,q_used_only_for_label_encoding_and_fixed_decoder=relative,
        seeds=[180,181,162],selection='dev XYZ p95 then mean; memory uses fixed training subset',
        limits=dict(seconds=1200,bytes=768_000_000,min_free_bytes=20*2**30))
    write(out/'manifest.json',manifest)
    try:
        data={}
        for split in ['train','val']:
            x=np.load(cache/(split+('-current.npy' if current else '-full.npy')),mmap_mode='r')
            x=np.array(x,copy=True)
            if current:x=np.pad(x,((0,0),(0,721)))
            q=np.load(cache/(split+'-q.npy'));pd=np.load(cache/(split+'-pd.npy'))[:,ACTIVE]
            target=pd.copy()
            if relative:target[:,:6]-=q[:,:6]
            data[split]={k:torch.as_tensor(v,device='cuda') for k,v in dict(x=x,q=q,pd=pd,target=target).items()}
        xmean=data['train']['x'].mean(0);xscale=data['train']['x'].std(0,unbiased=False).clamp(min=1e-3)
        ymean=data['train']['target'].mean(0);yscale=data['train']['target'].std(0,unbiased=False).clamp(min=1e-3)
        for split,d in data.items():d['x']=(d['x']-xmean)/xscale;d['target']=(d['target']-ymean)/yscale
        layers=[]
        for n,m in zip([1442,1024,1024,1024],[1024,1024,1024,512]):layers.extend([torch.nn.Linear(n,m),torch.nn.ReLU()])
        head=torch.nn.Linear(512,12);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);layers.append(head)
        model=torch.nn.Sequential(*layers).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.0001)
        manifest['parameters']=sum(p.numel() for p in model.parameters());write(out/'manifest.json',manifest)
        panels={s:torch.as_tensor(np.load(cache/(s+'-panel.npy')),device='cuda') for s in data}
        memids=panels['train'].unique();rng=np.random.default_rng(181)
        # Identical vectorized sequence-uniform/half-startup sampling across arms.
        phase=[]
        for pools in meta['metadata']['train']['phase_pools']:
            if not pools:raise ValueError('startup and later training pools required')
            lens=np.array([len(z) for z in pools]);phase.append((np.concatenate(pools),np.r_[0,np.cumsum(lens)[:-1]],lens))
        def draw():
            if memory:return memids[torch.randint(len(memids),(512,),device='cuda')]
            batches=[]
            for flat,offset,lens in phase:
                seq=rng.integers(len(lens),size=256);local=(rng.random(256)*lens[seq]).astype(np.int64)
                batches.append(flat[offset[seq]+local])
            return torch.as_tensor(np.concatenate(batches),device='cuda')
        @torch.no_grad()
        def evaluate(split,ids):
            model.eval();d=data[split];prediction=torch.cat([model(d['x'][ix]) for ix in ids.split(512)])*yscale+ymean
            if relative:prediction[:,:6]+=d['q'][ids,:6]
            delta=(prediction-d['pd'][ids]).cpu().numpy();xyz=np.linalg.norm(delta[:,:3],axis=-1)*1000
            # Native inverse, clamp and forward PD for a separate executable metric.
            q=d['q'][ids].cpu().numpy();offset=np.asarray(meta['controller']['offset'],np.float32)[ACTIVE];scale=np.asarray(meta['controller']['scale'],np.float32)[ACTIVE]
            raw=(prediction.cpu().numpy()-offset)/scale;raw[:,:6]=(prediction[:,:6].cpu().numpy()-q[:,:6]-offset[:6])/scale[:6];raw[:,6:]=raw[:,6:]*2-1
            clipped=raw.clip(-1,1);executed=offset+scale*np.concatenate((clipped[:,:6],(clipped[:,6:]+1)/2),-1);executed[:,:6]+=q[:,:6]
            executable_xyz=np.linalg.norm(executed[:,:3]-d['pd'][ids,:3].cpu().numpy(),axis=-1)*1000
            result=dict(xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(np.percentile(xyz,95)),
                xyz_executable_mean_mm=float(executable_xyz.mean()),xyz_executable_p95_mm=float(np.percentile(executable_xyz,95)),
                rotation_p95_rad=float(np.percentile(abs(delta[:,3:6]).max(-1),95)),finger_p95_rad=float(np.percentile(abs(delta[:,6:]).max(-1),95)),
                clipping=float((abs(raw-clipped).max(-1)>1e-6).mean()),normalized_L1=float((torch.as_tensor(delta,device='cuda')/yscale).abs().mean()))
            model.train();return result,prediction.cpu().numpy()
        curve=[];best=None;selected=None
        for step in range(a.updates+1):
            if step:
                ids=draw();optimizer.zero_grad(set_to_none=True);loss=(model(data['train']['x'][ids])-data['train']['target'][ids]).abs().mean();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite supervised fit')
                optimizer.step()
            if step%500==0 or step==a.updates:
                row=dict(step=step,elapsed_s=time.monotonic()-started,panels={});preds={}
                for split in data:row['panels'][split],preds[split]=evaluate(split,memids if memory and split=='train' else panels[split])
                measure=row['panels']['train' if memory else 'val'];key=(measure['xyz_p95_mm'],measure['xyz_mean_mm'])
                if best is None or key<best:
                    best=key;selected=row
                    torch.save(dict(schema='ref2dex.full-teacher-MLP.v1',step=step,arm=a.arm,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        xmean=xmean.cpu(),xscale=xscale.cpu(),ymean=ymean.cpu(),yscale=yscale.cpu(),manifest=manifest),out/('step%d.pt'%step))
                    np.savez_compressed(out/('predictions-step%d.npz'%step),**preds,train_indices=(memids if memory else panels['train']).cpu().numpy(),val_indices=panels['val'].cpu().numpy())
                curve.append(row);write(out/'curve.json',curve)
                consumers=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if consumers-{os.getpid()}:raise RuntimeError('foreign consumer on assigned GPU')
                gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
                write(out/'progress.json',dict(step=step,elapsed_s=row['elapsed_s'],eta_s=row['elapsed_s']/max(step,1)*(a.updates-step),gpu=gpu,dev=row['panels']['val'],train=row['panels']['train']))
                print(json.dumps(dict(step=step,elapsed_s=row['elapsed_s'],train=row['panels']['train']['xyz_mean_mm'],dev=row['panels']['val']['xyz_mean_mm'])),flush=True)
                if shutil.disk_usage(ROOT).free<20*2**30 or sum(f.stat().st_size for f in out.iterdir() if f.is_file())>768_000_000:raise RuntimeError('disk budget')
        write(out/'result.json',dict(status='COMPLETED',arm=a.arm,selected=selected,final=curve[-1],elapsed_s=time.monotonic()-started,
            scope='one-step offline supervised fit; no physical grasp claim'))
        manifest.update(status='COMPLETED')
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error));raise
    finally:
        write(out/'manifest.json',manifest);signal.alarm(0)


if __name__=='__main__':main()
