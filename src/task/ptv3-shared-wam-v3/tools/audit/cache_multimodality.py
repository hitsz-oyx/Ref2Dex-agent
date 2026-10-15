"""GPU-bounded existing-weight H,G embeddings and held-out hand futures; no fitting."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,get_worker_info
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.data import NineWindows,collate
from ptv3_shared_wam_v3.model import SharedWAMV3
from ptv3_shared_wam_v3.multimodality import reconstruct_history
from ptv3_shared_wam_v3.checkpoint import atomic_json
from ptv3_shared_wam.loading import to_device,project_ipc_directory
from oakink_wm.data import local_objects


def identity(data,index):
    if index<len(data.main):
        sid=int(np.searchsorted(data.main.base.offsets,index,side='right')-1);source=data.main.base.sources[sid]
        row,anchor,tick,category=map(int,source.rows[index-data.main.base.offsets[sid]])
        seq=source.sequences[row];capture=str(source.root/'processed/sequences'/seq)
        return sid,row,tick,capture,anchor
    row,tick=data.native.index[index-len(data.main)];r=data.native.meta['sequences'][row]
    return r['source_id'],row,tick,str((data.native.root/r['path']).resolve()),r['target_slot']


def select_indices(data,sid,count,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(sid),0 if data.split=='train' else 1]))
    selected=[];seen=set();captures=defaultdict(list)
    for index in data.draw_source(sid,max(count*12,256),rng):
        index=int(index)
        if index in seen:continue
        seen.add(index);_,row,tick,capture,_=identity(data,index)
        if len(captures[capture])>=8 or any(abs(tick-t)<28 for t in captures[capture]):continue
        captures[capture].append(tick);selected.append(index)
        if len(selected)>=count*2:break
    return selected


class Selected(Dataset):
    def __init__(self,base,indices):self.base,self.indices=base,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        index=self.indices[i];sample,provenance,native=self.base[index]
        complete=(sample['trajectory_valid']&sample['start_hand_valid'][None]).all(axis=(0,2))
        sides=np.flatnonzero(complete)
        if not len(sides) or len(sample['scene_points'])>8 or not sample['object_goal_valid'].all():return None
        side=int(sides[0]);sid,row,tick,capture,anchor=identity(self.base,index)
        if native is None:
            source=self.base.main.base.sources[sid];raw=source.sequence(source.sequences[row])
            slots=local_objects(raw['poses'][tick],raw['pose_valid'][tick],anchor,raw['centers'])
            keys=[str(raw['objects'][j]) for j in slots];geometry_key=str(raw['objects'][anchor])
        else:
            keys=list(provenance['geometry_keys']);geometry_key=self.base.native.meta['sequences'][row]['objects'][anchor]
        assert len(keys)==len(sample['scene_points'])
        order=np.argsort(keys);scene_history=np.zeros((8,4,8,3),np.float32)
        for dest,slot in enumerate(order):
            mask=sample['scene_object']==slot;xyz=sample['xyz'][mask];features=sample['features'][mask]
            ids=np.linspace(0,len(xyz)-1,8,dtype=int)
            scene_history[dest]=reconstruct_history(xyz[ids],features[ids,3:12])
        pattern=sum(int(v)<<j for j,v in enumerate(sample['start_hand_valid'].reshape(-1)))
        meta=dict(index=index,source_id=sid,row=row,tick=tick,split=self.base.split,capture=capture,
            geometry_key=geometry_key,scene_signature=json.dumps(sorted(keys)),n_objects=len(keys),side=side,hand_pattern=pattern)
        record=dict(meta=meta,history=reconstruct_history(sample['start_hand'],sample['start_hand_history']).astype(np.float32),
            goal=(sample['goal_object_points']-sample['start_object_points']).astype(np.float32),scene_history=scene_history,
            future=(sample['start_hand'][side][None]+sample['trajectory_target'][:,side]).astype(np.float32),
            delta=sample['trajectory_target'][:,side].copy(),start=sample['start_hand'][side].copy())
        target=int(sample['goal_object_slot']);record['target_surface']=sample['xyz'][sample['scene_object']==target].copy()
        record['target_queries']=sample['scene_points'][target].copy()
        record['target_future_queries']=sample['scene_points'][target][None]+sample['scene_target'][target]
        return sample,record


def pack(items):
    items=[x for x in items if x is not None]
    if not items:return None
    return collate([(s,{},None) for s,r in items])['model_inputs'],[r for s,r in items]


def worker_init(_):
    torch.set_num_threads(1);project_ipc_directory()
    for source in get_worker_info().dataset.base.main.base.sources:source.sequence=lru_cache(maxsize=64)(source.sequence)


def tip_proximity(record,prediction):
    result=[];source=record['target_queries'];cloud=record['target_surface']
    for t in (7,23):
        target=record['target_future_queries'][t];a=source.mean(0);b=target.mean(0)
        u,_,vt=np.linalg.svd((source-a).T@(target-b));r=vt.T@u.T
        if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
        surface=(cloud-a)@r.T+b
        result.append([np.linalg.norm(points[6:,None]-surface[None],axis=-1).min(-1) for points in (record['future'][t],prediction[t])])
    return np.asarray(result,np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trainer-progress',type=Path,required=True)
    p.add_argument('--train-per-source',type=int,default=2048);p.add_argument('--dev-per-source',type=int,default=128)
    p.add_argument('--seed',type=int,default=214);p.add_argument('--gpu',type=int,default=0);p.add_argument('--seconds',type=int,default=3600)
    a=p.parse_args();out=a.output.resolve();assert not out.exists() and 1<=a.train_per_source<=4096 and 1<=a.dev_per_source<=256 and 60<=a.seconds<=3600
    if shutil.disk_usage(out.parent).free<20*2**30:raise RuntimeError('20GiB disk floor')
    torch.set_num_threads(2);torch.cuda.set_device(a.gpu);free,total=torch.cuda.mem_get_info()
    if free<8*2**30:raise RuntimeError('8GiB GPU headroom required before co-located diagnostic')
    torch.cuda.set_per_process_memory_fraction(3*2**30/total,a.gpu)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.manual_seed(a.seed)
    packet=torch.load(a.checkpoint,map_location='cpu',weights_only=False);recipe=json.loads((a.recipe/'recipe.json').read_text())
    if packet['stats']['recipe']!=recipe:raise ValueError('checkpoint and audit corpus binding mismatch')
    model=SharedWAMV3(packet['stats'],packet['config']).cuda().eval();model.load_state_dict(packet['model'],strict=True)
    for name in ('hand','scene'):
        for field in ('mean','std'):assert torch.equal(getattr(model.normalizer,name+'_'+field).cpu(),torch.tensor(packet['stats'][name][field],dtype=torch.float32))
    step=packet['step'];parent=packet.get('audit_parent_checkpoint');del packet
    out.mkdir(parents=True);start=time.monotonic();project_ipc_directory();completed=[];counts={};stored={};last_parent=0;slow_count=0
    model.geometry.register_forward_hook(lambda module,args,result:stored.update(geometry=result))
    model.interaction.register_forward_hook(lambda module,args,result:stored.update(shared=result.detach()))
    progress=dict(status='RUNNING',pid=os.getpid(),checkpoint=str(a.checkpoint),checkpoint_step=step,parent_checkpoint=parent,
        recipe=str(a.recipe),seed=a.seed,gpu=a.gpu,allocator_limit_bytes=3*2**30,batch_size=1,max_seconds=a.seconds,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    try:
        dev=NineWindows(recipe['main3_root'],recipe['native_root'],'dev',recipe);sources=sorted(dev.counts,key=int)
        for split,target in [('dev',a.dev_per_source),('train',a.train_per_source)]:
            data=dev if split=='dev' else NineWindows(recipe['main3_root'],recipe['native_root'],'train',recipe)
            for sid in sources:
                ids=select_indices(data,sid,target,a.seed);rows=[];meta=[];started=time.monotonic();skipped=0
                loader=DataLoader(Selected(data,ids),batch_size=1,collate_fn=pack,num_workers=2,multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=2,
                    generator=torch.Generator().manual_seed(a.seed+int(sid)),timeout=180)
                for bound in loader:
                    if bound is None:skipped+=1;continue
                    batch,records=bound;record=records[0]
                    if (out/'STOP').exists() or time.monotonic()-start>a.seconds or shutil.disk_usage(out).free<20*2**30:raise RuntimeError('STOP/deadline/free-disk boundary')
                    free,_=torch.cuda.mem_get_info()
                    if free<4*2**30:raise RuntimeError('co-located GPU free headroom below4GiB')
                    batch=to_device(batch,torch.device('cuda',a.gpu))
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(batch)
                    if not torch.isfinite(pred['trajectory']).all():raise ValueError('nonfinite forecast')
                    if 'future_label_invariance' not in progress:
                        altered=dict(batch,trajectory_target=torch.zeros_like(batch['trajectory_target']),scene_target=torch.zeros_like(batch['scene_target']))
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):control=model(altered)
                        for field in ('trajectory','scene'):assert torch.allclose(pred[field],control[field],atol=1e-6,rtol=1e-5),field
                        progress['future_label_invariance']='PASS';del altered,control
                    z=stored['shared'].float();geo=stored['geometry'];offset=geo['tokens'].shape[1]
                    def pooled(values,valid):return (values*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
                    embedding=torch.cat((pooled(z[:,:offset],geo['mask']),pooled(z[:,offset:offset+22],batch['start_hand_valid'].flatten(1)),pooled(z[:,offset+22:],batch['object_goal_valid'])),dim=-1)[0].cpu().numpy()
                    side=record['meta']['side'];delta=pred['trajectory'][0,:,side].float().cpu().numpy();absolute=record['start'][None]+delta
                    contact=tip_proximity(record,absolute)
                    target_slot=int(batch['goal_object_slot'][0]);scene_error=torch.linalg.vector_norm(pred['scene'][0,target_slot]-batch['scene_target'][0,target_slot],dim=-1).mean(-1).float().cpu().numpy()
                    rows.append(dict(z=embedding,future=record['future'],delta=record['delta'],prediction=absolute,pred_delta=delta,start=record['start'],
                        history=record['history'],goal=record['goal'],scene_history=record['scene_history'],tip_proximity=contact,scene_error=scene_error))
                    meta.append(record['meta']);stored.clear();del pred,batch
                    if len(rows)%32==0 or len(rows)==target:
                        elapsed=time.monotonic()-start;rate=len(rows)/(time.monotonic()-started)
                        progress.update(split=split,source=sid,source_processed=len(rows),source_target=target,skipped=skipped,elapsed_s=elapsed,
                            samples_per_s=rate,own_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                            own_reserved_mib=torch.cuda.memory_reserved()/2**20,gpu_free_mib=free/2**20,completed=completed)
                        if elapsed-last_parent>60 or last_parent==0:
                            parent_progress=json.loads(a.trainer_progress.read_text());progress['parent_training']=parent_progress
                            slow_count=slow_count+1 if parent_progress.get('updates_per_s',1) <.9 else 0;last_parent=elapsed
                            if slow_count>=3:raise RuntimeError('three consecutive parent-throughput checks below0.9update/s; release diagnostic')
                        atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                    if len(rows)>=target:break
                del loader
                if not rows:raise ValueError('no complete eligible hand futures for source'+sid)
                np.savez_compressed(out/f'{split}-{sid}.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]},metadata=np.asarray([json.dumps(m) for m in meta]))
                completed.append(split+'-'+sid);counts[split+'-'+sid]=len(rows)
                if sum(p.stat().st_size for p in out.glob('*.npz'))>2_000_000_000:raise RuntimeError('2GB diagnostic artifacts boundary')
        progress.update(status='COMPLETED',counts=counts,completed=completed,elapsed_s=time.monotonic()-start)
        atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    except BaseException as e:
        progress.update(status='FAILED',error=repr(e),completed=completed,elapsed_s=time.monotonic()-start);atomic_json(out/'progress.json',progress);raise


if __name__=='__main__':main()
