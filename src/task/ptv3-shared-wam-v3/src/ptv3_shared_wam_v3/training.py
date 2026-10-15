"""Sustained nine-source GPU pretraining with immutable recovery checkpoints."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from .paths import ROOT, TASK, SCHEMA
from .model import SharedWAMV3
from .data import NineWindows, collate, loader
from .checkpoint import (atomic_json, save_checkpoint, load_checkpoint, restore_optimizer,
    checkpoint_alias, prune_recovery_checkpoints, directory_bytes, git_commit)
from .objectives import forecast_loss, forecast_metrics
from ptv3_shared_wam_v2.training import gradient_audit
from ptv3_shared_wam.loading import to_device


@torch.no_grad()
def evaluate(model, dataset, panel, config, device):
    model.eval()
    reports = {}
    for sid, indices in panel.items():
        totals, objective, count = {}, 0., 0
        for first in range(0,len(indices),8):
            samples = indices[first:first+8]
            batch = to_device(collate([dataset[int(i)] for i in samples])['model_inputs'],device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                pred = model(batch)
                loss,_ = forecast_loss(pred,batch,config,model.normalizer)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite DEV objective')
            objective += float(loss)*len(samples)
            count += len(samples)
            for key,value in forecast_metrics(pred,batch,config).items():
                slot = totals.setdefault(key,dict(sum_mm=0.,count=0))
                slot['sum_mm'] += value['sum_mm']; slot['count'] += value['count']
        reports[sid] = dict(objective=objective/count,metrics={k:dict(
            mean_mm=v['sum_mm']/v['count'] if v['count'] else None,count=v['count']) for k,v in totals.items()})
    model.train()
    return dict(sources=reports,objective=float(np.mean([r['objective'] for r in reports.values()])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--resume-reviewed-subset',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--loader-workers',type=int,default=2)
    p.add_argument('--experiment-card',type=Path,default=TASK/'docs/experiments/probes/P-20261015-nine-source-pretrain.md')
    a=p.parse_args()
    out,artifacts=a.output.resolve(),a.artifact_root.resolve()
    if artifacts/'outputs'/TASK.name not in out.parents or out.exists():
        raise ValueError('fresh v3 output required, including resume')
    config=json.loads(a.config.read_text())
    recipe=json.loads((a.recipe/'recipe.json').read_text())
    stats=json.loads((a.recipe/'horizon-stats.json').read_text())
    import yaml
    experiment=yaml.safe_load(a.experiment_card.read_text().split('---',2)[1])
    if experiment.get('task')!=TASK.name or config['seed'] not in experiment.get('seeds',[]):
        raise ValueError('Task-local experiment and training seed required')
    if config['schema']!=SCHEMA or stats.get('recipe')!=recipe or recipe.get('training_allowed') is not True:
        raise ValueError('v3 admitted recipe and matching common statistics required')
    if config['hand_weight']!=1 or config['scene_weight']!=1:
        raise ValueError('initial nine-source recipe retains hand/scene coefficients 1:1')
    world,rank,local=[int(os.environ.get(k,v)) for k,v in [('WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')]]
    if not 1<=world<=4 or config.get('expected_world_size',world)!=world or not 0<=a.loader_workers<=4:
        raise ValueError('bounded world size/workers and matching execution profile required')
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=config['tf32']
    torch.backends.cudnn.allow_tf32=config['tf32']
    if world>1:
        dist.init_process_group('nccl')
    device=torch.device('cuda',local)
    torch.manual_seed(config['seed']+rank)
    train=NineWindows(recipe['main3_root'],recipe['native_root'],'train',recipe)
    dev=NineWindows(recipe['main3_root'],recipe['native_root'],'dev',recipe)
    panel=dev.panel(config['validation_samples']//len(dev.counts),config['validation_seed'])
    model=SharedWAMV3(stats,config).to(device)
    optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
        weight_decay=config['weight_decay'],fused=True)
    first=0; previous_elapsed=0.; parent=None
    provenance=dict(task=TASK.name,git_commit=git_commit(),code_root=str(ROOT),
        experiment_id=experiment['experiment_id'],run_id=out.name,branch=experiment['branch'],
        world_size=world,recipe=str(a.recipe.resolve()),initialization='all weights from scratch',
        scope='offline nine-source pretraining Probe; robot benefit untested')
    if a.resume:
        parent=load_checkpoint(a.resume)
        if a.resume_reviewed_subset:
            from .data_transition import validate_subset_resume
            provenance['data_transition']=validate_subset_resume(parent,config,stats,world)
        elif parent['config']!=config or parent['stats']!=stats or parent['provenance']['world_size']!=world:
            raise ValueError('resume requires identical config/statistics/world size')
        model.load_state_dict(parent['model'],strict=True)
        restore_optimizer(optimizer,parent['optimizer'])
        first=parent['step']; previous_elapsed=parent['provenance'].get('elapsed_s',0.)
        provenance.update(resume_parent=str(a.resume.resolve()),parent_step=first,
            parent_git_commit=parent['provenance']['git_commit'],
            initialization='full v3 optimizer/model/per-rank RNG resume')
    if a.resume_reviewed_subset and not a.resume:raise ValueError('reviewed subset transition requires parent checkpoint')
    target=a.stop_after or config['updates']
    if not 0<=first<target<=config['updates']:
        raise ValueError('bounded increasing optimizer steps required')
    if rank==0:
        out.mkdir(parents=True)
        for name,value in [('config.json',config),('recipe.json',recipe),('horizon-stats.json',stats),('validation-panel.json',panel)]:
            atomic_json(out/name,value)
        atomic_json(out/'manifest.json',dict(status='RUNNING',config=config,provenance=provenance,
            parameters=model.parameter_counts(),train_windows=train.counts,dev_windows=dev.counts))
    if world>1:dist.barrier()
    started=time.monotonic(); curve=[]; best=float('inf'); files=[]; drawn=Counter()
    try:
        # Mixed nine-source engineering audit before sustained updates, per rank.
        coverage=train.panel(1,17)
        inputs=collate([train[ids[0]] for ids in coverage.values()])
        audit=gradient_audit(model,to_device(inputs['model_inputs'],device),config)
        if rank==0:
            atomic_json(out/'gradient-audit.json',audit)
            atomic_json(out/'coverage-provenance.json',inputs['provenance'])
        del inputs
        if parent:
            torch.set_rng_state(parent['rank_rng'][rank]['torch'])
            torch.cuda.set_rng_state(parent['rank_rng'][rank]['cuda'])
        wrapped=nn.parallel.DistributedDataParallel(model,device_ids=[local],gradient_as_bucket_view=True) if world>1 else model
        if rank==0:
            curve.append(dict(step=first,**evaluate(model,dev,panel,config,device)))
            atomic_json(out/'curve.json',curve)
        if world>1:dist.barrier()
        batches=loader(train,config,rank,first+1,target,a.loader_workers)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        logged=time.monotonic();logged_step=first
        for step,bound in enumerate(batches,first+1):
            batch=to_device(bound['model_inputs'],device,non_blocking=True)
            drawn.update(str(p['source_id']) for p in bound['provenance'])
            # Linear warmup avoids a large normalized-target jump at startup.
            scale=min(1.,step/max(1,config.get('warmup_updates',1)))
            for group in optimizer.param_groups:
                group['lr']=config['geometry_lr' if group['name']=='joint_geometry' else 'shared_lr']*scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                pred=wrapped(batch);loss,terms=forecast_loss(pred,batch,config,model.normalizer,distributed=True)
            if not torch.isfinite(loss):raise ValueError('nonfinite TRAIN loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
            if not torch.isfinite(norm) or norm<=0:raise ValueError('nonfinite/zero TRAIN gradient')
            optimizer.step()
            checkpoint_due=step==1 or step%config['checkpoint_interval']==0 or step==target
            new_best=False
            if step%config['validation_interval']==0 or step==target:
                if rank==0:
                    report=evaluate(model,dev,panel,config,device)
                    curve.append(dict(step=step,**report));atomic_json(out/'curve.json',curve)
                    new_best=report['objective']<best
                    if new_best:best=report['objective']
                decision=[new_best]
                if world>1:dist.broadcast_object_list(decision,src=0)
                new_best=decision[0]
            if checkpoint_due or new_best:
                states=[None]*world
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
                if world>1:dist.all_gather_object(states,rng)
                else:states=[rng]
                if rank==0:
                    provenance['elapsed_s']=previous_elapsed+time.monotonic()-started
                    path=out/('step%d.pt'%step)
                    save_checkpoint(path,model,optimizer,config,stats,step,provenance,states)
                    checkpoint_alias(out,'latest.pt',path.name)
                    if new_best:checkpoint_alias(out,'best.pt',path.name)
                    files.append(path.name);files=prune_recovery_checkpoints(out,files,config)
                    atomic_json(out/'checkpoint_files.json',files)
                if world>1:dist.barrier()
            if step==1 or step%config['log_interval']==0 or step==target:
                values=torch.stack([loss.detach(),terms['hand'].detach(),terms['scene'].detach()])
                if world>1:
                    dist.all_reduce(values);values/=world
                torch.cuda.synchronize()
                elapsed=time.monotonic()-started
                rate=(step-logged_step)/(time.monotonic()-logged)
                if rank==0:
                    record=dict(status='RUNNING',step=step,loss=float(values[0]),terms={k:float(values[i+1]) for i,k in enumerate(('hand','scene'))},
                        grad_norm=float(norm),elapsed_s=elapsed,updates_per_s=rate,
                        eta_seconds=(config['updates']-step)/rate,local_samples_per_s=rate*config['batch_size'],
                        global_batch=world*config['batch_size'],world_size=world,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        rank0_source_draws=dict(drawn))
                    atomic_json(out/'progress.json',record)
                    with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
                    if step==1:atomic_json(out/'first-update-provenance.json',bound['provenance'])
                logged=time.monotonic();logged_step=step
            if time.monotonic()-started+previous_elapsed>config['max_seconds']:
                raise TimeoutError('bounded total training deadline')
            if (out/'STOP').exists() or shutil.disk_usage(artifacts).free<20*2**30:
                raise RuntimeError('STOP or free disk floor')
        if rank==0:
            progress=json.loads((out/'progress.json').read_text())
            progress['status']='COMPLETED' if target==config['updates'] else 'PARTIAL'
            atomic_json(out/'progress.json',progress)
            atomic_json(out/'result.json',dict(status='COMPLETED' if target==config['updates'] else 'PARTIAL',
                final_step=target,elapsed_s=time.monotonic()-started,provenance=provenance))
            atomic_json(out/'manifest.json',dict(status='COMPLETED' if target==config['updates'] else 'PARTIAL',
                config=config,provenance=provenance,parameters=model.parameter_counts()))
    except BaseException as error:
        if rank==0:atomic_json(out/'failure.json',dict(status='FAILED',error=repr(error)))
        raise
    finally:
        if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
