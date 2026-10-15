"""Bounded full-v2 native wiring check: explicit engineering use, at most four steps."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

TASK=Path(__file__).resolve().parents[2]
REPO=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap-stats',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,default=REPO)
    p.add_argument('--config',type=Path,default=TASK.parent/'ptv3-shared-wam-v2/configs/pretrain_remote_80k.json')
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--engineering-only',action='store_true',required=True)
    a=p.parse_args()
    out,artifacts=a.output.resolve(),a.artifact_root.resolve()
    if (artifacts/'outputs'/TASK.name not in out.parents or out.exists()
            or not 1<=a.steps<=4 or not 1<=a.batch_size<=32):
        raise ValueError('fresh Task output and bounded engineering load required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('assigned GPU already occupied')
    def check_slot():
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        active={int(row.split(',')[0]) for row in rows.splitlines() if float(row.split(',')[1])>256}
        if len(active|{a.gpu})>6:
            raise RuntimeError('no GPU budget slot for this engineering run')
    check_slot()
    if shutil.disk_usage(artifacts).free<20*2**30:
        raise RuntimeError('less than 20GiB free')
    scratch=artifacts/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),
        TRITON_CACHE_DIR=str(scratch/'triton'),TORCH_EXTENSIONS_DIR=str(scratch/'extensions'),
        PYTHONDONTWRITEBYTECODE='1')
    for path in (TASK/'src',TASK.parent/'ptv3-shared-wam-v2/src',TASK.parent/'ptv3-shared-wam/src',TASK/'tools/run'):
        sys.path.insert(0,str(path))
    import numpy as np
    import torch
    from pose_mesh_data.clean_data import CleanPoseMeshWindows,clean_collate
    from pose_mesh_data.native_recipe import NativeSourceSampler,source_pools,validate_stats_binding
    from pose_mesh_data.native_normalization import build_native_model
    from ptv3_shared_wam_v2.training import gradient_audit
    from ptv3_shared_wam_v2.objectives import forecast_loss,forecast_metrics
    from ptv3_shared_wam.loading import to_device
    from clean_native_corpus import write_json,footprint
    recipe=json.loads((a.recipe/'recipe.json').read_text())
    stats=json.loads((a.recipe/'horizon-stats.json').read_text())
    validate_stats_binding(recipe,stats)
    root=Path(recipe['data_manifest']).parent
    config=json.loads(a.config.read_text())
    config.update(data_profile='native-pose-mesh-clean.v1',seed=17,updates=a.steps,batch_size=a.batch_size)
    workspace=recipe['encoder_workspace']
    if config['voxel_origin_m']!=workspace['origin_m'] or config['voxel_m']!=workspace['voxel_m']:
        raise ValueError('encoder config differs from explicit input admission')
    if config['hand_weight']!=1 or config['scene_weight']!=1:
        raise ValueError('this engineering recipe retains equal existing loss coefficients')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    progress=dict(status='RUNNING',engineering_only=True,training_allowed=False,run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        code_root=str(REPO),physical_gpu=a.gpu,pid=os.getpid(),max_seconds=600,artifact_budget_bytes=100_000_000)
    stop=threading.Event();resource_conflict=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                state=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                active={int(row.split(',')[0]) for row in state.splitlines() if float(row.split(',')[2])>256}
                if len(active|{a.gpu})>6:
                    resource_conflict.set()
                with (out/'gpu.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),elapsed_s=time.monotonic()-started,gpus=state))+'\n')
            except Exception as error:
                resource_conflict.set()
                with (out/'gpu.jsonl').open('a') as f:f.write(json.dumps({'error':repr(error)})+'\n')
            stop.wait(1)
    def checkpoint(phase,step=0):
        progress.update(phase=phase,step=step,elapsed_s=time.monotonic()-started)
        write_json(out/'progress.json',progress)
        if (progress['elapsed_s']>600 or shutil.disk_usage(artifacts).free<20*2**30
                or footprint(out)>100_000_000 or (out/'STOP').exists() or resource_conflict.is_set()):
            raise RuntimeError('engineering deadline/disk/artifact/STOP/GPU-budget boundary')
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        checkpoint('readers')
        train=CleanPoseMeshWindows(root,'train',engineering=True,cache_size=8)
        sampler=NativeSourceSampler(train,recipe)
        dev=CleanPoseMeshWindows(root,'dev',engineering=True,cache_size=8)
        check_slot()
        torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        model=build_native_model(stats,config,json.loads(a.bootstrap_stats.read_text())).cuda()
        optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
            weight_decay=config['weight_decay'],fused=True)
        write_json(out/'recipe.json',recipe);write_json(out/'horizon-stats.json',stats);write_json(out/'config.json',config)
        coverage=[train[int(pool[0])] for pool in sampler.pools.values()]
        bound=clean_collate(coverage);batch=to_device(bound['model_inputs'],'cuda')
        checkpoint('gradient-audit')
        audit=gradient_audit(model,batch,config)
        write_json(out/'gradient-audit.json',audit)
        write_json(out/'coverage-provenance.json',[asdict(w.provenance) for w in coverage])
        with torch.no_grad():
            model.eval();prediction=model(batch);reference,_=forecast_loss(prediction,batch,config,model.normalizer)
            altered=dict(batch)
            for key,mask in (('trajectory',batch['trajectory_valid'] & batch['start_hand_valid'][:,None]),
                             ('scene',batch['scene_valid'] & batch['object_valid'][:,:,None,None])):
                altered[key+'_target']=torch.where(mask[...,None],batch[key+'_target'],1e6)
            masked,_=forecast_loss(prediction,altered,config,model.normalizer)
            if not torch.equal(reference,masked):raise ValueError('invalid targets affect actual-model loss')
        del prediction,batch,bound
        model.train()
        curve=[];drawn=Counter()
        for step in range(1,a.steps+1):
            checkpoint('optimizer',step)
            indices=sampler.draw(a.batch_size,17,step=step)
            windows=[train[int(i)] for i in indices]
            bound=clean_collate(windows)
            batch=to_device(bound['model_inputs'],'cuda')
            drawn.update(str(w.provenance.source_id) for w in windows)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                prediction=model(batch);loss,terms=forecast_loss(prediction,batch,config,model.normalizer)
            if not torch.isfinite(loss):raise ValueError('nonfinite actual native loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
            if not torch.isfinite(grad) or grad<=0:raise ValueError('nonfinite/zero native gradient')
            tracked={group['name']:next(p for p in group['params'] if p.grad is not None) for group in optimizer.param_groups}
            previous={name:parameter.detach().clone() for name,parameter in tracked.items()}
            optimizer.step();torch.cuda.synchronize()
            changes={name:float((parameter-previous[name]).abs().max()) for name,parameter in tracked.items()}
            if any(not np.isfinite(v) or v<=0 for v in changes.values()):
                raise ValueError('native optimizer did not update both parameter groups')
            record=dict(step=step,loss=float(loss),terms={k:float(v) for k,v in terms.items()},grad_norm=float(grad),
                parameter_change_max=changes,elapsed_s=time.monotonic()-started,source_ids=batch['source_id'].tolist(),
                allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20)
            curve.append(record);write_json(out/'curve.json',curve)
            print(json.dumps(record),flush=True)
            if step==1:write_json(out/'first-update-provenance.json',[asdict(w.provenance) for w in windows])
        checkpoint('dev')
        panels=source_pools(dev);metrics={}
        model.eval()
        with torch.no_grad():
            for sid,pool in panels.items():
                example=dev[int(pool[0])];bound=clean_collate([example]);batch=to_device(bound['model_inputs'],'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):prediction=model(batch)
                metrics[sid]=forecast_metrics(prediction,batch,config)
        result=dict(status='ENGINEERING_PASS',engineering_only=True,training_allowed=False,
            git_commit=progress['git_commit'],code_root=str(REPO),physical_gpu=a.gpu,
            parameters=model.parameter_counts(),completed_updates=a.steps,coverage_sources=[w.provenance.source_id for w in coverage],
            sampled_source_counts=dict(drawn),dev_sources=list(panels),train_sources_without_dev=sorted(set(sampler.sources)-set(panels)),
            invalid_target_mask_invariance=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,elapsed_s=time.monotonic()-started,
            initialization='Full v2 architecture from scratch; real main3 bootstrap statistics immediately replaced with native common TRAIN moments before any forward; no parent weights loaded.',
            bootstrap_stats=str(a.bootstrap_stats.resolve()),metrics=metrics,
            limitations=['Four-step engineering check only, not model efficacy or large-training admission.',
                         'Six native TRAIN sources; HO-Cap excluded, and no DexYCB DEV fabricated.'])
        checkpoint('complete',a.steps)
        write_json(out/'result.json',result)
        progress['status']='COMPLETED';write_json(out/'progress.json',progress)
        print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
    except BaseException as error:
        progress.update(status='FAILED',reason=repr(error));write_json(out/'progress.json',progress)
        raise
    finally:
        stop.set();thread.join(timeout=5)


if __name__=='__main__':main()
