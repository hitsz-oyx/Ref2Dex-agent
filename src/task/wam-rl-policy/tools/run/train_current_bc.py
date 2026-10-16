"""Prepare full current-step Shadow labels, then train a resident-GPU BC MLP."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','train'))
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5)
    p.add_argument('--seed',type=int,default=241)
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--seconds',type=int,default=7200)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output required')
    if shutil.disk_usage(ROOT).free<24*2**30:raise RuntimeError('Disk reserve')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if a.stage=='train':
        occupied=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('Assigned GPU occupied')
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import numpy as np
    out.mkdir(parents=True);started=time.monotonic()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()
    def write(name,value):
        part=out/(name+'.part');part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');part.replace(out/name)
    process=dict(status='RUNNING',pid=os.getpid(),stage=a.stage,gpu=a.gpu if a.stage=='train' else None,
        git_commit=commit,max_seconds=a.seconds)
    write('process.json',process)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Bounded current BC')))
    signal.alarm(a.seconds)
    try:
        if a.stage=='prepare':
            meta=json.loads((a.data/'manifest.json').read_text());assert meta['status']=='COMPLETED' and meta['training_allowed']
            manifest=dict(schema='ref2dex.shadow-current-bc-cache.v1',status='RUNNING',source=str(a.data.resolve()),
                git_commit=commit,inputs='same-tick canonical observation300 clamped[-5,5]',labels='same-tick canonical applied action24',
                preparation_device='CPU: file extraction/statistics only; no model compute',splits={})
            for split in ('train','dev'):
                records=[r for r in meta['records'] if r['split']==split];size=sum(len(r['ticks']) for r in records)
                x=np.lib.format.open_memmap(out/(split+'-x.npy'),mode='w+',dtype=np.float32,shape=(size,300))
                y=np.lib.format.open_memmap(out/(split+'-y.npy'),mode='w+',dtype=np.float32,shape=(size,24))
                seq=np.lib.format.open_memmap(out/(split+'-seq.npy'),mode='w+',dtype=np.int32,shape=(size,))
                ticks=np.lib.format.open_memmap(out/(split+'-tick.npy'),mode='w+',dtype=np.int32,shape=(size,))
                objects=[];object_index={};cursor=0;index=[]
                for i,r in enumerate(records):
                    ids=np.asarray(r['ticks']);count=len(ids)
                    with np.load(a.data/r['path'],allow_pickle=False) as z:
                        if z['observation'].shape!=(200,300) or z['action'].shape!=(199,24) or not np.allclose(np.diff(z['time']),1/60,atol=1e-7):
                            raise ValueError('Native clock/shape changed')
                        observations=np.clip(z['observation'][ids],-5,5);actions=z['action'][ids]
                    if not np.isfinite(observations).all() or not np.isfinite(actions).all() or abs(actions).max()>1.00001:
                        raise ValueError('Finite canonical labels required')
                    x[cursor:cursor+count]=observations;y[cursor:cursor+count]=actions;seq[cursor:cursor+count]=i;ticks[cursor:cursor+count]=ids
                    code=r['object_code']
                    if code not in object_index:object_index[code]=len(objects);objects.append(code)
                    index.append(dict(sequence=i,lo=cursor,hi=cursor+count,object=object_index[code],source=r['path']))
                    cursor+=count
                    if i%1000==0:write('progress.json',dict(split=split,records=i,rows=cursor,total=size,elapsed_s=time.monotonic()-started))
                x.flush();y.flush();seq.flush();ticks.flush();assert cursor==size
                write(split+'-index.json',dict(records=index,objects=objects))
                manifest['splits'][split]=dict(windows=size,objects=len(objects),sequences=len(records))
                if split=='train':
                    stats=dict(x_mean=x.mean(0,dtype=np.float64).astype(np.float32),x_scale=x.std(0,dtype=np.float64).clip(.05).astype(np.float32),
                        y_mean=y.mean(0,dtype=np.float64).astype(np.float32),y_scale=y.std(0,dtype=np.float64).clip(.05).astype(np.float32))
                    np.savez(out/'stats.npz',**stats)
                else:
                    # Object-balanced fixed offline panel: no behavioral outcome selection.
                    rng=np.random.default_rng(244);pool=[]
                    for obj in range(len(objects)):
                        groups=[r for r in index if r['object']==obj]
                        for _ in range(16):
                            r=groups[int(rng.integers(len(groups)))];pool.append(int(rng.integers(r['lo'],r['hi'])))
                    np.save(out/'dev-panel.npy',np.asarray(pool,np.int64))
            manifest['status']='COMPLETED';write('manifest.json',manifest)
        else:
            import torch
            from wam_rl_policy.bc_mlp import SCHEMA,CurrentMLP
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
            cache=json.loads((a.data/'manifest.json').read_text());assert cache['status']=='COMPLETED'
            stats=dict(np.load(a.data/'stats.npz'));data={}
            for split in ('train','dev'):
                data[split]={k:torch.as_tensor(np.load(a.data/(split+'-'+k+'.npy')),device='cuda') for k in ('x','y')}
            torch.manual_seed(a.seed);model=CurrentMLP().cuda()
            with torch.no_grad():
                for key,value in stats.items():getattr(model,key).copy_(torch.as_tensor(value,device='cuda'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
            rng=np.random.default_rng(a.seed+1);panel=np.load(a.data/'dev-panel.npy')
            # Match previous startup emphasis while using the entire strict source pool.
            ticks=np.load(a.data/'train-tick.npy');early=np.flatnonzero(ticks<8)
            manifest=dict(schema=SCHEMA,status='RUNNING',git_commit=commit,seed=a.seed,steps=a.steps,batch=a.batch,
                source_cache=str(a.data.resolve()),data=cache['splits'],stats='TRAIN-only',
                initialization='fresh random, no teacher checkpoint weights',lr=3e-4,weight_decay=1e-5,
                sampler='half startup tick<8; half full TRAIN windows, uniform within pools',
                loss='TRAIN-standardized current24D L1',params=sum(p.numel() for p in model.parameters()),
                selection='offline object-balanced DEV current command MAE; endpoints also preserved')
            write('manifest.json',manifest);curve=[];best=float('inf')
            @torch.no_grad()
            def evaluate():
                model.eval();errors=[]
                for off in range(0,len(panel),2048):
                    ids=panel[off:off+2048];errors.append((model(data['dev']['x'][ids])-data['dev']['y'][ids]).abs())
                error=torch.cat(errors);model.train()
                return dict(command_mae=float(error.mean()),root_mae=float(error[:,:6].mean()),finger_mae=float(error[:,6:].mean()))
            for step in range(1,a.steps+1):
                ids=np.concatenate((rng.choice(early,a.batch//2),rng.integers(len(ticks),size=a.batch-a.batch//2)))
                x=data['train']['x'][ids];y=data['train']['y'][ids]
                pred=model(x);loss=((pred-y)/model.y_scale).abs().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite BC loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if step==1 or step%1000==0 or step==a.steps:
                    torch.cuda.synchronize();elapsed=time.monotonic()-started
                    metrics=evaluate();row=dict(step=step,loss=float(loss),dev=metrics,elapsed_s=elapsed,eta_s=elapsed/step*(a.steps-step),
                        gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip(),
                        peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    curve.append(row);write('curve.json',curve);write('progress.json',row);print(json.dumps(row),flush=True)
                    packet=dict(schema=SCHEMA,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),config=manifest,
                        source_cache=str(a.data.resolve()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state)
                    if metrics['command_mae']<best:
                        best=metrics['command_mae'];torch.save(packet,out/'best.pt.part');(out/'best.pt.part').replace(out/'best.pt')
                    if step in (5000,10000,20000,50000) or step==a.steps:torch.save(packet,out/('step%d.pt'%step))
                    if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('Disk reserve')
            manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started);write('manifest.json',manifest)
        process.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    except BaseException as error:
        process.update(status='FAILED',error=repr(error));raise
    finally:write('process.json',process);signal.alarm(0)


if __name__=='__main__':main()
