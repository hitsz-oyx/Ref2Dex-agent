"""Independent v6 event-conditioned pretraining and bounded continuation checks."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import multiprocessing
import shutil
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.contracts import SCHEMA,INPUT_SEMANTICS,LABEL_SEMANTICS
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.local_subset import LOCAL_SCOPES, EXPANDED_SCOPE, validate_local_execution, initialize_weights, initialize_geometry_weights
from ptv3_shared_wam_v5.storage import read_json
from ptv3_shared_wam_v5.checkpoints import checkpoint_due,retained_steps
from ptv3_shared_wam_v5.metrics import forecast_metrics,merge_metrics,summarize_metrics
from ptv3_shared_wam_v6.objectives import forecast_loss,loss_components,objective_semantics,combine_components,active_geometry_weight
from ptv3_shared_wam_v6.geometry_consistency import validate_geometry_config
from ptv3_shared_wam_v6.precision import ActivationMonitor
from ptv3_shared_wam_v5.training_contract import execution_contract,validate_resume,loader_options,restore_optimizer_state
from ptv3_shared_wam_v5.selection import trajectory_score,bootstrap_trajectory,TAU_SELECTION
from ptv3_shared_wam_v4.objectives import normalized_components


def write(path,value):
    tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    path.chmod(0o664)


class Steps:
    def __init__(self,dataset,batch,seed,rank,start,last):
        self.dataset,self.batch,self.seed,self.rank,self.start,self.last=dataset,batch,seed,rank,start,last
    def __iter__(self):
        for step in range(self.start,self.last+1):yield self.dataset.draw(self.batch,self.seed,step,self.rank)
    def __len__(self):return self.last-self.start+1


def evaluate_condition(model,dataset,config,device,drop_event=False):
    if not len(dataset):return dict(status='NO_DEV_WINDOWS')
    was_training=model.training;model.eval();rng=np.random.default_rng(config.get('validation_seed',171));sources={}
    with torch.inference_mode():
        for sid,pool in dataset.pools.items():
            chosen=rng.choice(pool,min(len(pool),config.get('validation_samples',32)),replace=False)
            components=None;hand_sum=hand_count=0;diagnostics={}
            for start in range(0,len(chosen),config.get('validation_microbatch_size',8)):
                batch=collate([dataset[int(i)] for i in chosen[start:start+config.get('validation_microbatch_size',8)]])
                batch={k:v.to(device) for k,v in batch.items()}
                if drop_event:batch['event_valid']=torch.zeros_like(batch['event_valid'])
                with torch.autocast('cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda') and config.get('amp',True)):
                    prediction=model(batch)
                current=loss_components(prediction,batch,config,model.normalizer,reduction_dtype=torch.float64)
                if components is None:components=current
                else:components={k:(components[k][0]+v[0],components[k][1]+v[1]) for k,v in current.items()}
                mask=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
                error=(prediction['trajectory']-torch.where(mask[...,None],batch['trajectory_target'],0.)).norm(dim=-1)
                hand_sum+=float(torch.where(mask,error,0.).sum())*1000;hand_count+=int(mask.sum())
                merge_metrics(diagnostics,forecast_metrics(prediction,batch))
                if current.geometry_diagnostics:
                    merge_metrics(diagnostics,current.geometry_diagnostics)
            loss,terms=combine_components(normalized_components(components,distributed=False),config)
            sources[sid]=dict(objective=float(loss),hand_epe_mm=hand_sum/max(hand_count,1),
                metrics=summarize_metrics(diagnostics),
                components={k:float(v) for k,v in terms.items()})
    model.train(was_training)
    return dict(status='COMPLETED',sources=sources,source_macro_objective=float(np.mean([v['objective'] for v in sources.values()])))


def evaluate(model,dataset,config,device):
    conditioned=evaluate_condition(model,dataset,config,device)
    unknown=evaluate_condition(model,dataset,config,device,drop_event=True)
    conditioned['unknown_event_control']=unknown
    if conditioned.get('sources'):
        conditioned['conditioned_minus_unknown_hand_macro_mm']=float(np.mean([
            v['hand_epe_mm']-unknown['sources'][sid]['hand_epe_mm']
            for sid,v in conditioned['sources'].items()]))
    return conditioned


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','manifest','stats','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int)
    p.add_argument('--loader-context',choices=('spawn','forkserver'),default='spawn')
    p.add_argument('--loader-timeout',type=int,default=120)
    p.add_argument('--recovery-probe-updates',type=int)
    p.add_argument('--continuation-updates',type=int)
    p.add_argument('--capacity-probe-updates',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--initialize-from',type=Path)
    p.add_argument('--initialize-geometry-from',type=Path)
    p.add_argument('--preserve-all-checkpoints',action='store_true')
    p.add_argument('--checkpoint-budget-gib',type=int,default=32)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    if sum(bool(x) for x in (a.resume,a.initialize_from,a.initialize_geometry_from))>1:
        raise ValueError('choose one resume/initialization source')
    config=read_json(a.config);stats=read_json(a.stats)
    if not 8<=a.checkpoint_budget_gib<=32:raise ValueError('checkpoint preservation budget must be8..32GiB')
    validate_geometry_config(config)
    recipe=validate_recipe(a.manifest,a.stats)
    if a.capacity_probe_updates and (not (a.resume or a.initialize_geometry_from) or a.smoke or a.recovery_probe_updates
            or a.continuation_updates or not 3<=a.capacity_probe_updates<=16
            or recipe.get('local_subset_scope')!=EXPANDED_SCOPE):
        raise ValueError('capacity probe requires expanded-scope resume and3..16 updates')
    if config.get('event_dropout')!=recipe['event_dropout']:
        raise ValueError('config/dropout differs from admitted v6 recipe')
    reader_options=dict(geometry_gt_object=config.get('geometry_object_source')=='gt_dense2048')
    dataset=V6Windows(a.manifest,'train',**reader_options)
    last=config['updates'] if a.steps is None else a.steps
    batch_size=config['batch_size'] if a.batch_size is None else a.batch_size
    if a.recovery_probe_updates is not None and (not a.resume or a.smoke or not 1<=a.recovery_probe_updates<=16):
        raise ValueError('recovery probe requires a checkpoint and1..16 actual updates')
    if a.continuation_updates is not None and (not a.resume or a.smoke or a.recovery_probe_updates or not 17<=a.continuation_updates<=5000):
        raise ValueError('bounded continuation requires v6 resume and17..5000 updates')
    if a.smoke:
        if not 1<=last<=16:raise ValueError('engineering smoke requires1..16 bounded steps')
    elif recipe.get('local_subset_scope') not in LOCAL_SCOPES and (dataset.meta.get('training_allowed') is not True or dataset.meta.get('engineering_subset')
          or dataset.meta.get('pending_sequences',0)):
        raise ValueError('full training requires a complete, explicitly admitted frozen v6 recipe')
    if last<1 or batch_size<1:raise ValueError('positive bounded steps/batch required')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    validate_local_execution(config,recipe,world,last,batch_size)
    if recipe.get('local_subset_scope') in LOCAL_SCOPES and not (a.resume or a.initialize_from or a.initialize_geometry_from):
        raise ValueError('local subset requires admitted initialization or its own strict resume')
    if recipe.get('local_subset_scope') == EXPANDED_SCOPE and world==1 and not a.capacity_probe_updates:
        raise ValueError('expanded production training requires three GPUs; world1 only for capacity')
    device=f'cuda:{local}' if world>1 else a.device
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('requested CUDA unavailable')
        torch.cuda.set_device(torch.device(device));torch.backends.cuda.matmul.allow_tf32=config.get('tf32',True)
    if world>1:dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo')
    if not a.smoke and a.seed is None and 'seed' not in config:raise ValueError('full training requires an explicitly allocated --seed')
    seed=0 if a.smoke else int(a.seed if a.seed is not None else config['seed'])
    config=dict(config,seed=seed,batch_size=batch_size)
    contract=execution_contract(config,world)
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'result.json').exists() or any(a.output.glob('step*.pt*')):
        raise ValueError('fresh output required; resume into a new directory without overwriting or rotating previous checkpoints')
    identity=hashlib.sha256(a.manifest.read_bytes()+a.stats.read_bytes()).hexdigest()
    model=SharedWAMV6(stats,config).to(device)
    if config.get('strict_fp32') and any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('strict FP32 requires all model parameters float32')
    activation_monitor=ActivationMonitor(model) if config.get('monitor_activation_rms') and rank==0 else None
    optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
        weight_decay=config['weight_decay'],fused=device.startswith('cuda'))
    first=1;previous_elapsed=0.;best_score=None;best_step=None;best_path=None;tau_best=None;resume_binding=None;migration_binding=None;topology_transition=None;initialization_binding=None
    if a.initialize_from:
        saved=torch.load(a.initialize_from,map_location='cpu',weights_only=False)
        initialization_binding=initialize_weights(model,saved,a.initialize_from,recipe)
        del saved
    if a.initialize_geometry_from:
        saved=torch.load(a.initialize_geometry_from,map_location='cpu',weights_only=False)
        initialization_binding=initialize_geometry_weights(model,saved,a.initialize_geometry_from,config,identity)
        if a.capacity_probe_updates is not None:last=min(last,a.capacity_probe_updates)
        del saved
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved.get('schema')!=SCHEMA:raise ValueError('independent v6 checkpoint required')
        legacy_path=a.resume.parent/'run-manifest.json'
        legacy=read_json(legacy_path) if saved.get('execution_contract') is None and legacy_path.is_file() else None
        resume_binding=validate_resume(saved,config,identity,world,legacy)
        resume_binding.update(checkpoint=str(a.resume.resolve()),legacy_manifest=str(legacy_path) if legacy else None)
        model.load_state_dict(saved['model'],strict=True)
        initialization_binding=saved.get('initialization_binding')
        migration_binding=saved.get('migration_binding') or saved.get('migration')
        topology_transition=saved.get('topology_transition')
        resume_binding['optimizer_restore']=restore_optimizer_state(optimizer,saved['optimizer'])
        first=int(saved['step'])+1
        previous_elapsed=float(saved.get('elapsed_s',0.));best=saved.get('selection',{})
        best_score,best_step,best_path=best.get('score'),best.get('step'),best.get('checkpoint')
        tau_best=best.get('trajectory') or bootstrap_trajectory(a.resume.parent)
        if tau_best and tau_best.get('protocol')!=TAU_SELECTION:raise ValueError('trajectory selection protocol mismatch')
        random.setstate(saved['random_state'][rank]['python']);np.random.set_state(saved['random_state'][rank]['numpy'])
        torch.set_rng_state(saved['random_state'][rank]['torch'])
        if device.startswith('cuda'):torch.cuda.set_rng_state(saved['random_state'][rank]['cuda'])
        if a.recovery_probe_updates is not None:last=min(last,first+a.recovery_probe_updates-1)
        if a.continuation_updates is not None:last=min(last,first+a.continuation_updates-1)
        if a.capacity_probe_updates is not None:last=min(last,first+a.capacity_probe_updates-1)
        if last<first:raise ValueError('resume target must exceed saved update')
        del saved
    wrapped=DDP(model,device_ids=[local] if device.startswith('cuda') else None) if world>1 else model
    kwargs=dict(dataset=dataset,batch_sampler=Steps(dataset,batch_size,seed,rank,first,last),collate_fn=collate,
                num_workers=a.workers,pin_memory=device.startswith('cuda'),
                generator=torch.Generator().manual_seed(seed*1009+rank))
    if a.workers:
        ipc=Path(os.environ.get('REF2DEX_RUN_ROOT',paths.ROOT))/'tmp'/('v6i-'+str(os.getpid()))
        ipc.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.environ['TMPDIR']=str(ipc);tempfile.tempdir=str(ipc)
        multiprocessing.current_process()._config['tempdir']=str(ipc)
    kwargs.update(loader_options(a.workers,a.loader_timeout,a.loader_context))
    loader=DataLoader(**kwargs)
    dev=V6Windows(a.manifest,'dev',**reader_options) if rank==0 and not (a.smoke or a.capacity_probe_updates) else None
    if rank==0:
        write(a.output/'config.json',config)
        write(a.output/'run-manifest.json',dict(schema=SCHEMA,work_version=SCHEMA,git_commit=a.git_commit,
            run_id=a.output.name,data_identity=identity,input_semantics=INPUT_SEMANTICS,label_semantics=LABEL_SEMANTICS,
            objective_semantics=objective_semantics(config),engineering_smoke=a.smoke,device=device,world_size=world,
            precision_runtime=model.precision_runtime,
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,steps=last,batch_size=batch_size,
            execution_contract=contract,resume_binding=resume_binding,migration_binding=migration_binding,topology_transition=topology_transition,initialization_binding=initialization_binding,
            loader=dict(context=a.loader_context,workers=a.workers,timeout_s=a.loader_timeout),
            trajectory_selection_protocol=TAU_SELECTION,recovery_probe=bool(a.recovery_probe_updates),continuation_updates=a.continuation_updates,
            event_recipe=recipe,capacity_probe_updates=a.capacity_probe_updates,
            checkpoint_preservation=dict(all=a.preserve_all_checkpoints,budget_gib=a.checkpoint_budget_gib),
            source_ids=dataset.meta['source_ids'],max_seconds=config['max_seconds'],min_free_gib=20,config=config))
    start=time.monotonic();times=[];events=[];seen=set();draws=Counter();loss_value=None;step=first-1;completed_step=first-1;checkpoints_written=0;last_saved=None

    def save(step):
        nonlocal checkpoints_written,last_saved,best_path,tau_best
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None)
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng)
        else:states[0]=rng
        if rank==0:
            target=a.output/f'step{step:06d}.pt';pending=target.with_suffix('.pt.part')
            if target.exists() or pending.exists():raise ValueError('never overwrite existing v6 checkpoint or partial checkpoint')
            if best_step==step:best_path=str(target.resolve())
            if tau_best and tau_best['step']==step:tau_best=dict(tau_best,checkpoint=str(target.resolve()))
            torch.save(dict(schema=SCHEMA,data_identity=identity,config=config,step=step,world_size=world,
                objective_semantics=objective_semantics(config),
                execution_contract=contract,resume_binding=resume_binding,migration_binding=migration_binding,topology_transition=topology_transition,initialization_binding=initialization_binding,
                elapsed_s=previous_elapsed+time.monotonic()-start,
                selection=dict(score=best_score,step=best_step,checkpoint=best_path,trajectory=tau_best),
                model=model.state_dict(),optimizer=optimizer.state_dict(),random_state=states),pending)
            pending.replace(target);target.chmod(0o664);checkpoints_written+=1
            for name,path in [('best',best_path),('trajectory-best',tau_best.get('checkpoint') if tau_best else None)]:
                if path:
                    link=a.output/(name+'.pt');temporary=a.output/(name+'.pt.link')
                    temporary.symlink_to(path);temporary.replace(link)
            recent=sorted(a.output.glob('step*.pt'))
            protected=retained_steps([int(f.stem[4:]) for f in recent],config)|({best_step} if best_step else set())
            if tau_best:protected.add(tau_best['step'])
            for old in recent:
                if not a.preserve_all_checkpoints and int(old.stem[4:]) not in protected:old.unlink()
            artifact_budget=a.checkpoint_budget_gib*2**30 if a.preserve_all_checkpoints else config['artifact_budget_bytes']
            if sum(f.stat().st_size for f in a.output.iterdir() if f.is_file() and not f.is_symlink())>artifact_budget:
                raise RuntimeError('owned v6 artifact budget exceeded')
        last_saved=step
        if world>1:dist.barrier()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    if first==1 and config.get('validation_before_training') and not (a.smoke or a.capacity_probe_updates):
        if world>1:dist.barrier()
        if rank==0:
            evaluation=evaluate(model,dev,config,device)
            write(a.output/'dev-step000000.json',evaluation)
        if world>1:dist.barrier()
    wrapped.train()
    iterator=iter(loader);loader_failure=None
    for step in range(first,last+1):
        stop=(a.output/'STOP').exists() or previous_elapsed+time.monotonic()-start>config['max_seconds'] or shutil.disk_usage(a.output).free<20*2**30
        flag=torch.tensor(int(stop),device=device)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item():break
        batch=None
        try:batch=next(iterator)
        except Exception as exc:loader_failure=repr(exc)
        flag=torch.tensor(int(loader_failure is not None),device=device)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item():
            loader_failure=loader_failure or 'another rank failed to load its batch'
            break
        seen.update(map(int,batch['source_id'].tolist()));draws.update(map(int,batch['source_id'].tolist()))
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        multiplier=min(1.,step/max(1,config.get('warmup_updates',0))) if not a.smoke else 1.
        optimizer.param_groups[0]['lr']=config['geometry_lr']*multiplier
        optimizer.param_groups[1]['lr']=config['shared_lr']*multiplier
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if device.startswith('cuda') else None
        if ev:ev[0].record()
        if activation_monitor:
            activation_monitor.enabled=step<=first+2 or step%config['log_interval']==0
        with torch.autocast('cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda') and config.get('amp',True)):
            prediction=wrapped(batch);loss,terms=forecast_loss(prediction,batch,config,model.normalizer,world>1,step=step)
        if ev:ev[1].record()
        loss.backward()
        if ev:ev[2].record()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'],error_if_nonfinite=True)
        optimizer.step()
        completed_step=step
        if ev:ev[3].record();events.append(ev)
        if len(events)>64:events.pop(0)
        if not torch.isfinite(loss.detach()):raise ValueError('nonfinite loss')
        loss_value=float(loss.detach());times.append(time.monotonic()-start)
        if len(times)>256:times.pop(0)
        if step<=first+2 or step%config['log_interval']==0:
            term_names=('hand','scene','contact','part')+(('geometry_contact','geometry_clearance') if config.get('geometry_consistency') else ())
            values=torch.stack([terms[k].detach() for k in term_names])
            sources=torch.tensor([draws[s] for s in range(11)],device=device,dtype=torch.long)
            if world>1:dist.all_reduce(values);values/=world;dist.all_reduce(sources)
            if rank==0:
                means=dict(zip(term_names,map(float,values)))
                objective=config['hand_weight']*means['hand']+config['scene_weight']*means['scene']+config.get('interaction_weight',1.)*(means['contact']+means['part'])
                if config.get('geometry_consistency'):
                    objective+=active_geometry_weight(config,step)*(means['geometry_contact']+config['geometry_clearance_weight']*means['geometry_clearance'])
                rate=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None
                event_valid=batch['event_valid'];event_type=batch['event_type']
                event_counts=torch.stack([((event_type==k)&event_valid).sum() for k in range(3)]+[(~event_valid).sum()])
                # Logging is rank0-local; DDP gradient is already synchronized.
                event_grad=float(torch.stack([p.grad.detach().float().norm().square() for p in model.event_goal.parameters() if p.grad is not None]).sum().sqrt())
                record=dict(event_condition_counts_local=dict(zip(('NONE','CONTACT','RELEASE','UNKNOWN'),event_counts.tolist())),
                    event_goal_gradient_norm_after_clip=event_grad,
                    event_goal_output_weight_norm=float(model.event_goal.encoder[-1].weight.detach().float().norm()),
                    status='RUNNING',step=step,loss=objective,terms=means,grad_norm=float(norm),
                    activation_rms=dict(activation_monitor.values) if activation_monitor else None,
                    geometry_weight=active_geometry_weight(config,step),
                    elapsed_s=previous_elapsed+times[-1],updates_per_s=rate,
                    eta_seconds=(last-step)/rate if rate else None,global_batch=batch_size*world,world_size=world,
                    source_ids=[s for s,c in enumerate(sources.tolist()) if c],
                    source_draws={str(s):c for s,c in enumerate(sources.tolist()) if c},
                    learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None)
                write(a.output/'progress.json',record)
                with (a.output/'train.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
        new_best=False
        if not (a.smoke or a.capacity_probe_updates) and (step%config['validation_interval']==0 or step==config.get('validation_initial_step',20)
                            or ((a.recovery_probe_updates or a.continuation_updates) and step==first)):
            if world>1:dist.barrier()
            if rank==0:
                evaluation=evaluate(model,dev,config,device)
                candidate=trajectory_score(evaluation);evaluation['trajectory_selection']=candidate
                write(a.output/f'dev-step{step:06d}.json',evaluation)
                score=evaluation.get('source_macro_objective')
                if score is not None and np.isfinite(score) and (best_score is None or score<best_score):
                    best_score,best_step,new_best=float(score),step,True
                if candidate and (tau_best is None or candidate['score']<tau_best['score']):
                    tau_best=dict(candidate,step=step,checkpoint=None,origin='fixed-DEV');new_best=True
            if world>1:dist.barrier()
            signal=torch.tensor(int(new_best),device=device)
            if world>1:dist.broadcast(signal,src=0)
            new_best=bool(signal.item())
        if not (a.smoke or a.capacity_probe_updates) and (checkpoint_due(step,last,config) or new_best):save(step)
    if not (a.smoke or a.capacity_probe_updates) and completed_step>=first and last_saved!=completed_step:save(completed_step)
    if device.startswith('cuda'):torch.cuda.synchronize()
    signatures=None
    if (a.smoke or a.recovery_probe_updates or a.continuation_updates
            or (config.get('geometry_consistency') and a.steps is not None)) and world>1:
        signature=torch.stack([torch.stack([p.detach().double().sum(),p.detach().double().square().sum()])
                               for p in model.parameters()]).sum(0)
        all_signatures=[torch.zeros_like(signature) for _ in range(world)]
        dist.all_gather(all_signatures,signature)
        if any(not torch.equal(all_signatures[0],x) for x in all_signatures[1:]):raise ValueError('DDP model parameters diverged across ranks')
        signatures=[x.cpu().tolist() for x in all_signatures]
    if rank==0:
        measured=events[min(2,len(events)-1):]
        timing={name:float(np.mean([ev[i].elapsed_time(ev[i+1]) for ev in measured])) for i,name in enumerate(('forward_loss_ms','backward_ms','optimizer_ms'))} if measured else {}
        result=dict(status='FAILED_DATA_LOADER' if loader_failure else ('COMPLETED' if completed_step==last else 'STOPPED'),
            loader_failure=loader_failure,engineering_smoke=a.smoke,recovery_probe=bool(a.recovery_probe_updates),continuation_updates=a.continuation_updates,optimizer_updates=completed_step-first+1,
            last_step=completed_step,last_loss=loss_value,elapsed_s=time.monotonic()-start,source_ids_seen=sorted(seen),
            capacity_probe=bool(a.capacity_probe_updates),
            steady_updates_per_s=(len(times)-2)/(times[-1]-times[1]) if len(times)>2 else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device.startswith('cuda') else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
            timing=timing,checkpoint_written=bool(checkpoints_written),ddp_parameter_signatures=signatures,
            best_selection=dict(score=best_score,step=best_step,checkpoint=best_path,trajectory=tau_best))
        write(a.output/'result.json',result);print(json.dumps(result),flush=True)
        if (a.output/'progress.json').exists():
            progress=read_json(a.output/'progress.json');progress.update(status=result['status'],step=completed_step)
            write(a.output/'progress.json',progress)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
