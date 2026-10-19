"""Full DExplore BC using the shared Shadow model, sampler and optimizer recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def guard(out):
    if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB storage reserve')
    if (out/'STOP').exists():raise InterruptedError('owned STOP requested')


def prepare(a):
    from wam_rl_policy_v2.inspire_shadow import CURRENT_DIM,INPUT_DIM,CONTRACT,ACTIVE,current_state,measured_from_legacy,object_state
    root=a.output;root.mkdir(parents=True)
    meta=json.loads((a.source/'manifest.json').read_text())
    if meta.get('status')!='COMPLETED' or not meta.get('training_allowed') or len(meta['records'])!=660:
        raise ValueError('complete admitted660 same-execution corpus required')
    records=meta['records'];sizes={s:sum(r['frames']-1 for r in records if ('dev' if r['split']=='val' else r['split'])==s) for s in ('train','dev')}
    expected=sum(sizes.values())*(CURRENT_DIM+INPUT_DIM+12)*4
    if expected>1024**3:raise RuntimeError('unexpected cache allocation above1GiB')
    arrays={};counts={s:0 for s in sizes};indices={s:[] for s in sizes}
    for s,n in sizes.items():
        for name,shape,dtype in [('x',(n,CURRENT_DIM),np.float32),('y',(n,12),np.float32),('scheduled',(n,INPUT_DIM),np.float32),('include',(n,),bool),('tick',(n,),np.int32),('seq',(n,),np.int32)]:
            arrays[s,name]=np.lib.format.open_memmap(root/f'{s}-{name}.npy',mode='w+',dtype=dtype,shape=shape)
        arrays[s,'include'][:]=False
    geometries={};prepared=[];geometry_rows=[];dense_rows=[];(root/'geometries').mkdir()
    for i,row in enumerate(records):
        guard(root)
        split='dev' if row['split']=='val' else row['split'];folder=a.source/row['path']
        with np.load(folder/'trajectory.npz') as z:
            n=len(z['action']);action=z['action'].copy()
            if (n!=row['frames']-1 or z['native_current_obs'].shape!=(n+1,721)
                    or not np.array_equal(z['action'],z['applied']) or not np.array_equal(z['frame_tick'],np.arange(n+1))):
                raise ValueError('same-tick canonical capture mismatch: '+row['sequence'])
            y=action[:,ACTIVE];previous=np.zeros_like(y);previous[1:]=y[:-1]
            goal=np.broadcast_to(z['reference_obj'][-1,:3,3],(n,3))
            x=current_state(z['q'][:-1],z['dq'][:-1],measured_from_legacy(z['native_current_obs'][:-1]),
                object_state(z['obj'][:-1],z['velocity'][:-1]),goal,previous)
            if not np.isfinite(y).all() or np.abs(y).max()>1.00001:raise ValueError('invalid command labels')
        code='dexplore/'+row['object']
        with np.load(folder/'canonical.npz') as z:
            surface=z['points'].astype(np.float32);normals=z['normals'].astype(np.float32)
        normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
        query=surface[np.linspace(0,len(surface)-1,32,dtype=int)]
        signature=hashlib.sha256(surface.tobytes()+normals.tobytes()+query.tobytes()).hexdigest()
        if code in geometries and geometries[code][1]!=signature:raise ValueError('inconsistent per-object canonical geometry')
        if code not in geometries:
            j=len(geometries);geometries[code]=(j,signature);path=root/'geometries'/f'{j:03d}.npz'
            np.savez_compressed(path,surface=surface,normals=normals,query=query)
            geometry_rows.append(dict(object_code=code,scale=1,geometry=str(path)))
            dense_rows.append(dict(key=code+'@1',path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        lo=counts[split];hi=lo+n
        arrays[split,'x'][lo:hi]=x;arrays[split,'y'][lo:hi]=y
        arrays[split,'scheduled'][lo:hi,:CURRENT_DIM]=x;arrays[split,'scheduled'][lo:hi,CURRENT_DIM:]=0
        arrays[split,'tick'][lo:hi]=np.arange(n);arrays[split,'seq'][lo:hi]=i
        record=dict(**row,cache_split=split,lo=lo,hi=hi,geometry=geometries[code][0],source=str(folder))
        prepared.append(record);indices[split].append(record);counts[split]=hi
        if i%50==0:print(json.dumps(dict(stage='prepare',sequences=i+1,total=660,commands=sum(counts.values()))),flush=True)
    for arr in arrays.values():arr.flush()
    x,y=arrays['train','x'],arrays['train','y']
    stats=dict(x_mean=x.mean(0,dtype=np.float64).astype(np.float32),x_scale=x.std(0,dtype=np.float64).clip(.05).astype(np.float32),
        y_mean=y.mean(0,dtype=np.float64).astype(np.float32),y_scale=y.std(0,dtype=np.float64).clip(.05).astype(np.float32))
    np.savez(root/'stats.npz',**stats)
    rng=np.random.default_rng(244);panel=[]
    for obj in sorted({r['object'] for r in indices['dev']}):
        groups=[r for r in indices['dev'] if r['object']==obj]
        for _ in range(16):
            r=groups[int(rng.integers(len(groups)))];panel.append(int(rng.integers(r['lo'],r['hi'])))
    np.save(root/'dev-panel.npy',np.array(panel,np.int64))
    for rank in (0,1):np.save(root/f'done-{rank}.npy',np.zeros(660,bool))
    write(root/'surface-manifest.json',dict(schema='ref2dex.wam-v4-dense-surface.v2',status='COMPLETED',objects=dense_rows))
    meta=dict(schema='ref2dex.inspire-shadow-full-cache.v1',status='CURRENT_COMPLETED',scheduled_status='PENDING',
        source=str(a.source),records=geometry_rows,sequences=prepared,splits={s:dict(commands=n,sequences=len(indices[s])) for s,n in sizes.items()},
        conditioning=CONTRACT,current_dim=CURRENT_DIM,input_dim=INPUT_DIM,objects=len(geometries),full_source_sequences=660,
        current_admission='all finite same-tick commands, no future-label requirement',
        scheduled_admission='all startup rows and in-workspace measured H4; invalid rows excluded, never fabricated tau',
        geometry_replacement='explicit new Inspire canonical manifest',git_commit=a.commit,wam=str(a.wam.resolve()))
    write(root/'manifest.json',meta)
    from wam_rl_policy_v2.inspire_task_cache import build_task_fields
    build_task_fields(root)


class RecordSamples:
    def __init__(self,records,geometry,batch):self.records,self.geometry,self.batch=records,geometry,batch
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        from wam_rl_policy_v2.goal_scheduler import OrderedGoalScheduler,update_schedulers
        from wam_rl_policy_v2.wam_v4_runtime import v4_online_samples,collate_v4
        from wam_rl_policy_v2.inspire_shadow import NATIVE_TO_WAM
        from scipy.spatial.transform import Rotation
        row=self.records[index];geo=self.geometry[row['geometry']]
        with np.load(Path(row['source'])/'trajectory.npz') as z:
            poses=z['obj'][:-1].copy();hands=z['hand'][:-1][:,NATIVE_TO_WAM].copy();reference=z['reference_obj'].copy()
        scheduler=OrderedGoalScheduler(reference,np.arange(len(reference))/30,geo[2],interaction_points=geo[0],goal_type='se3_pose')
        goals=[]
        for tick,pose in enumerate(poses):
            decision=update_schedulers([scheduler],pose[None],np.array([tick/30]),hands[tick:tick+1],fast_geometry=True)[0]
            if tick>=3:goals.append(decision['goal_delta'])
        objects=np.c_[poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()]
        selected=np.arange(3,len(poses));ix=selected[:,None]+np.array([-3,-2,-1,0])
        batches=[]
        for lo in range(0,len(selected),self.batch):
            hi=min(lo+self.batch,len(selected))
            samples=v4_online_samples(objects[ix[lo:hi]],hands[ix[lo:hi]],[geo]*(hi-lo),goal_delta=np.asarray(goals[lo:hi]))
            batches.append(collate_v4(samples))
        return dict(index=index,batches=batches,rotations=poses[selected,:3,:3])


def one(value):return value


def conditions(a):
    import torch
    from wam_rl_policy_v2.wam_v4_runtime import V4Runtime
    from wam_rl_policy_v2.inspire_shadow import CURRENT_DIM,tau_world,TAU_SLICE,VALID_INDEX
    from wam_rl_policy_v2.batched_geometry import workspace_valid
    root=a.output;meta=json.loads((root/'manifest.json').read_text())
    runtime=V4Runtime(a.wam,meta['records'],root,dense_manifest=root/'surface-manifest.json',allow_geometry_replacement=True)
    identity=dict(wam=runtime.conditioning_identity,geometry=runtime.geometry_bindings,inference_precision='fp32')
    prior=root/f'wam-identity-{a.rank}.json'
    if prior.exists() and json.loads(prior.read_text())!=identity:raise ValueError('WAM/cache resume identity changed')
    write(prior,identity)
    assigned=[(i,r) for i,r in enumerate(meta['sequences']) if i%2==a.rank]
    done=np.load(root/f'done-{a.rank}.npy',mmap_mode='r+')
    assigned=[(i,r) for i,r in assigned if not done[i]]
    if not assigned:return
    rows=[r for _,r in assigned]
    loader=torch.utils.data.DataLoader(RecordSamples(rows,runtime.geometry,a.wam_batch),batch_size=None,num_workers=a.workers,
        collate_fn=one,pin_memory=True,**(dict(multiprocessing_context='spawn',prefetch_factor=2) if a.workers else {}))
    arrays={s:np.load(root/f'{s}-scheduled.npy',mmap_mode='r+') for s in ('train','dev')}
    include={s:np.load(root/f'{s}-include.npy',mmap_mode='r+') for s in arrays}
    started=time.monotonic();rows_done=0;invalid_count=0;checked=False
    for count,item in enumerate(loader,1):
        guard(root);global_index,row=assigned[item['index']];split=row['cache_split'];n=row['hi']-row['lo'];block=arrays[split][row['lo']:row['hi']]
        selected=include[split][row['lo']:row['hi']];selected[:min(3,n)]=True
        for j,cpu in enumerate(item['batches']):
            lo=3+j*a.wam_batch;size=len(cpu['xyz']);batch={k:v.to(runtime.device,non_blocking=True) for k,v in cpu.items()}
            keep=workspace_valid(batch,runtime.config);host=keep.cpu().numpy();ids=np.arange(lo,lo+size)[host]
            invalid_count+=int((~host).sum());selected[lo:lo+size]=host
            if len(ids):
                batch={k:v[keep] for k,v in batch.items()}
                with torch.inference_mode(),torch.autocast('cuda',enabled=False):trajectory=runtime.predict(batch)['trajectory'].float().cpu().numpy()
                if not checked:
                    first={k:v[:1] for k,v in batch.items()}
                    with torch.inference_mode(),torch.autocast('cuda',enabled=False):single=runtime.predict(first)['trajectory'].float().cpu().numpy()
                    error=float(np.abs(single[0]-trajectory[0]).max())
                    write(root/f'batch-check-{a.rank}.json',dict(max_hand_displacement_difference_m=error,tolerance_m=.001,batch=a.wam_batch))
                    if error>.001:raise ValueError('WAM batch/single discrepancy exceeds1mm')
                    checked=True
                rotations=item['rotations'][lo-3:lo-3+size][host]
                block[ids,TAU_SLICE]=tau_world(trajectory,rotations).reshape(len(ids),264);block[ids,VALID_INDEX]=1
        if not np.isfinite(block).all():raise ValueError('nonfinite scheduled conditions')
        arrays[split].flush();include[split].flush();done[global_index]=True;done.flush();rows_done+=n
        if count==1 or count%10==0 or count==len(rows):
            elapsed=time.monotonic()-started;report=dict(status='RUNNING',rank=a.rank,records=count,total=len(rows),commands=rows_done,
                invalid_workspace_frames=invalid_count,elapsed_s=elapsed,eta_s=elapsed/count*(len(rows)-count),gpu=a.gpu,
                gpu_state=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
            write(root/f'shard-{a.rank}-progress.json',report);print(json.dumps(report),flush=True)
    report['status']='COMPLETED';write(root/f'shard-{a.rank}-progress.json',report)


def train(a):
    import torch
    from wam_rl_policy_v2.inspire_shadow import CURRENT_DIM,CURRENT_INPUT_DIM,INPUT_DIM,CONTRACT,CURRENT_SCHEMA,SCHEMA,BC_ADMISSION,expanded_model,initialize_scheduled,bc_inputs,bc_stats,bc_admission
    root=a.output;root.mkdir();meta=json.loads((a.cache/'manifest.json').read_text());scheduled=a.stage=='scheduled'
    if scheduled and meta['scheduled_status']!='COMPLETED':raise ValueError('complete WAM conditions required')
    torch.manual_seed(a.seed);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    if meta.get('task_status')!='COMPLETED' or meta.get('conditioning')!=CONTRACT:raise ValueError('complete explicit-task cache required')
    admission={s:bc_admission(a.cache,s,scheduled) for s in ('train','dev')}
    data={s:dict(x=torch.as_tensor(bc_inputs(a.cache,s,scheduled),device='cuda'),y=torch.as_tensor(np.load(a.cache/f'{s}-y.npy'),device='cuda')) for s in ('train','dev')}
    stats=bc_stats(a.cache,scheduled)
    model=expanded_model(scheduled).cuda()
    for key,value in stats.items():getattr(model,key).copy_(torch.as_tensor(value,device='cuda'))
    if scheduled:
        if not a.parent:raise ValueError('current BC parent required')
        parent=torch.load(a.parent,map_location='cpu',weights_only=False)
        if parent['config'].get('training_admission')!=BC_ADMISSION:raise ValueError('current BC parent has a different task admission contract')
        initialize_scheduled(model,parent)
        for k,v in stats.items():
            if not torch.equal(getattr(model,k).cpu(),torch.as_tensor(v)):raise ValueError('parent normalization mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    include=admission['train']
    all_rows=np.flatnonzero(include);ticks=np.load(a.cache/'train-tick.npy');early=all_rows[ticks[all_rows]<4]
    dev_include=admission['dev']
    panel=np.load(a.cache/'dev-panel.npy');panel=panel[dev_include[panel]]
    if not len(early) or not len(panel):raise ValueError('startup and DEV coverage required')
    rng=np.random.default_rng(a.seed+1);best=float('inf');started=time.monotonic();curve=[]
    manifest=dict(schema=SCHEMA if scheduled else CURRENT_SCHEMA,status='RUNNING',git_commit=a.commit,
        input_dim=INPUT_DIM if scheduled else CURRENT_INPUT_DIM,action_dim=12,conditioning=CONTRACT,steps=a.steps,batch=2048,
        source_cache=str(a.cache),train_rows=len(all_rows),source_train_rows=len(include),dev_panel_rows=len(panel),training_admission=BC_ADMISSION,
        source_sequences=meta['splits'],params=sum(p.numel() for p in model.parameters()),hidden_dims=[1024,1024,512,512],activation='elu',
        optimizer='AdamW',lr=3e-4,weight_decay=1e-5,loss='TRAIN-standardized independent-command L1',
        sampler='half startup <4 native30Hz ticks (=8/60s), half full admitted TRAIN',normalization='TRAIN-only std floor .05; tau fixed .05m',
        selection='fixed object-balanced DEV command MAE',parent=str(a.parent) if a.parent else None,seed=a.seed,
        wam_identity=json.loads((a.cache/'wam-identity-0.json').read_text()) if scheduled else None)
    write(root/'manifest.json',manifest)
    @torch.no_grad()
    def evaluate():
        error=(model(data['dev']['x'][panel])-data['dev']['y'][panel]).abs()
        return dict(command_mae=float(error.mean()),root_mae=float(error[:,:6].mean()),finger_mae=float(error[:,6:].mean()))
    if scheduled:
        initial=evaluate();best=initial['command_mae'];write(root/'initial-dev.json',initial)
        initial_packet=dict(schema=manifest['schema'],step=0,model=model.state_dict(),optimizer=optimizer.state_dict(),config=manifest,source_cache=str(a.cache),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state)
        torch.save(initial_packet,root/'best.pt')
    for step in range(1,a.steps+1):
        ids=np.r_[rng.choice(early,1024),rng.choice(all_rows,1024)]
        prediction=model(data['train']['x'][ids]);loss=((prediction-data['train']['y'][ids])/model.y_scale).abs().mean()
        if not torch.isfinite(loss):raise FloatingPointError('BC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%1000==0 or step==a.steps:
            guard(root);metrics=evaluate();elapsed=time.monotonic()-started
            row=dict(step=step,loss=float(loss),dev=metrics,elapsed_s=elapsed,eta_s=elapsed/step*(a.steps-step),
                gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
            curve.append(row);write(root/'curve.json',curve);write(root/'progress.json',row);print(json.dumps(row),flush=True)
            packet=dict(schema=manifest['schema'],step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),config=manifest,source_cache=str(a.cache),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state)
            torch.save(packet,root/'latest.pt.part');(root/'latest.pt.part').replace(root/'latest.pt')
            if metrics['command_mae']<best:
                best=metrics['command_mae'];torch.save(packet,root/'best.pt.part');(root/'best.pt.part').replace(root/'best.pt')
            if step==a.steps:torch.save(packet,root/f'step{step}.pt')
    manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started,best_dev_command_mae=best);write(root/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','conditions','current','scheduled'])
    p.add_argument('--source',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--parent',type=Path)
    p.add_argument('--wam',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt')
    p.add_argument('--gpu',type=int,default=6);p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--wam-batch',type=int,default=64);p.add_argument('--steps',type=int,default=50000);p.add_argument('--seed',type=int,default=246);p.add_argument('--seconds',type=int,default=28800)
    a=p.parse_args();a.output=a.output.resolve();a.source=a.source.resolve()
    if ROOT/'outputs/wam-rl-policy-v2' not in a.output.parents:raise ValueError('Task output required')
    if a.stage!='conditions' and a.output.exists():raise ValueError('fresh stage output required')
    if not 0<a.seconds<=28800 or a.steps!=50000 or a.rank not in (0,1):raise ValueError('bounded Shadow recipe required')
    if a.cache:a.cache=a.cache.resolve()
    scratch=ROOT/'tmp'/a.output.name;scratch.mkdir(exist_ok=True,parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        REF2DEX_DEEP_OPT='all',PYTHONDONTWRITEBYTECODE='1')
    a.commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded full Inspire BC stage')));signal.alarm(a.seconds)
    if a.stage=='prepare':prepare(a)
    elif a.stage=='conditions':conditions(a)
    else:train(a)


if __name__=='__main__':main()
