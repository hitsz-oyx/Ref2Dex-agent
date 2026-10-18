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
from .model import SharedWAMV4
from .data import NineWindows, DevInputCache, collate, loader
from .performance import StepTimings
from .checkpoint import (atomic_json, save_checkpoint, load_checkpoint, restore_optimizer,
    initialize_from_v3,
    checkpoint_alias, prune_recovery_checkpoints, directory_bytes, git_commit,
    data_identity, validate_resume_contract)
from .objectives import (forecast_loss, forecast_metrics, loss_components, combine_components,
    OBJECTIVE_SEMANTICS, EVALUATION_SEMANTICS)
from ptv3_shared_wam.loading import to_device
from .lr_transition import validate_lr_resume, apply_learning_rates
from .selection import consider_selection, inherit_checkpoint


def gradient_audit(model, batch, config):
    """Run a bounded real-data audit for the v4 interaction path.

    A representative parameter per path is sufficient to prove routing; an
    all-parameter reverse pass here is prohibitively expensive before every
    distributed run and does not add a stronger contract check.
    """
    model.eval()
    prediction = model(batch)
    total, terms = forecast_loss(prediction, batch, config, model.normalizer)
    if not torch.isfinite(total):
        raise ValueError("nonfinite actual-data objective")
    report = dict(parameters=model.parameter_counts(),
                  interaction_loss=float(terms["interaction"].detach()))
    groups = (
        ("interaction_head", next(model.interaction_head.parameters())),
        ("interaction_query", next(model.interaction_query.parameters())),
        ("shared_interaction", next(model.interaction.parameters())),
        ("geometry", next(model.geometry.parameters())),
        ("hand_encoder", next(model.hand_encoder.parameters())),
        ("goal", next(model.goal.parameters())),
    )
    flat_grads = torch.autograd.grad(terms["interaction"],
                                     [parameter for _, parameter in groups],
                                     allow_unused=True)
    for (name, _), grad in zip(groups, flat_grads):
        norm = float(grad.detach().float().square().sum()) ** .5 if grad is not None else 0.
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("interaction loss does not reach " + name)
        report["interaction_" + name + "_gradient"] = norm
    model.train()
    return report


@torch.no_grad()
def evaluate(model, dataset, panel, config, device, microbatch_size=None, input_cache=None):
    was_training=model.training
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        try:
            return _evaluate(model,dataset,panel,config,device,microbatch_size,input_cache)
        finally:
            model.train(was_training)


def _evaluate(model, dataset, panel, config, device, microbatch_size=None, input_cache=None):
    model.eval()
    reports = {}
    microbatch_size = microbatch_size or config.get('validation_microbatch_size', 8)
    if microbatch_size < 1:
        raise ValueError('positive validation microbatch size required')
    for sid, indices in panel.items():
        totals, components = {}, {}
        if not len(indices):
            raise ValueError('nonempty source DEV panel required')
        batches = (input_cache.source_batches(sid,indices,microbatch_size) if input_cache else
            (collate([dataset[int(i)] for i in indices[first:first+microbatch_size]])['model_inputs']
             for first in range(0,len(indices),microbatch_size)))
        for cpu_batch in batches:
            batch = to_device(cpu_batch,device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['amp']):
                pred = model(batch)
                parts = loss_components(pred,batch,config,model.normalizer,reduction_dtype=torch.float64)
            for key,(numerator, denominator) in parts.items():
                if not torch.isfinite(numerator) or not torch.isfinite(denominator):
                    raise ValueError('nonfinite DEV sufficient statistics')
                slot = components.setdefault(key,dict(numerator=0.,denominator=0.))
                slot['numerator'] += float(numerator); slot['denominator'] += float(denominator)
            for key,value in forecast_metrics(pred,batch,config).items():
                slot = totals.setdefault(key,dict(sum=0.,count=0,unit=value['unit']))
                slot['sum'] += value['sum']; slot['count'] += value['count']
        means = {k:v['numerator']/max(v['denominator'],1e-8) for k,v in components.items()}
        objective, terms = combine_components(means,config)
        reports[sid] = dict(objective=objective,terms=terms,components=components,metrics={k:dict(
            mean=v['sum']/v['count'] if v['count'] else None,count=v['count'],unit=v['unit']) for k,v in totals.items()})
    model.train()
    return dict(sources=reports,objective=float(np.mean([r['objective'] for r in reports.values()])),
                semantics=EVALUATION_SEMANTICS)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--init-v3',type=Path,
        help='model-only v3 checkpoint warm-start; new structured C modules are initialized fresh')
    p.add_argument('--resume-reviewed-subset',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--resume-learning-rate-drop',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--loader-workers',type=int,default=2)
    p.add_argument('--profile-steps',type=int,default=0)
    p.add_argument('--no-dev-cache',action='store_true')
    p.add_argument('--experiment-card',type=Path,default=TASK/'docs/experiments/probes/P-20261017-wam-v4-interaction-head.md')
    a=p.parse_args()
    if a.resume_reviewed_subset:
        p.error('--resume-reviewed-subset is not supported by WAM-v4; use a separately reviewed new run')
    if a.resume and a.init_v3:
        raise ValueError('choose either a full v4 resume or a model-only v3 warm-start')
    if a.resume_learning_rate_drop and (not a.resume or a.resume_reviewed_subset):
        raise ValueError('LR-only transition requires resume and cannot change the dataset')
    out,artifacts=a.output.resolve(),a.artifact_root.resolve()
    if artifacts/'outputs'/TASK.name not in out.parents or out.exists():
        raise ValueError('fresh v3 output required, including resume')
    config=json.loads(a.config.read_text())
    if (config.get('objective_semantics') != OBJECTIVE_SEMANTICS
            or config.get('evaluation_semantics') != EVALUATION_SEMANTICS):
        raise ValueError('explicit audited objective/evaluation semantics required')
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
    if not 0 <= a.profile_steps <= 2000:
        p.error('--profile-steps must be between0 and2000')
    parent=load_checkpoint(a.resume) if a.resume else None
    if parent:
        validate_resume_contract(parent,config,stats,world)
        if a.resume_learning_rate_drop:
            validate_lr_resume(parent,config,stats,world)
        elif parent['config']!=config:
            raise ValueError('resume requires identical configuration')
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
    filters={split:dataset.window_filter_sha256 for split,dataset in [('train',train),('dev',dev)]}
    if parent and parent['provenance'].get('window_filter_sha256',filters) != filters:
        raise ValueError('resume window filter SHA-256 mismatch')
    model=SharedWAMV4(stats,config)
    warm_start=None
    if a.init_v3:
        warm_start=initialize_from_v3(a.init_v3,model,config,stats)
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
        weight_decay=config['weight_decay'],fused=True)
    first=0; previous_elapsed=0.
    provenance=dict(task=TASK.name,git_commit=git_commit(),code_root=str(ROOT),
        experiment_id=experiment['experiment_id'],run_id=out.name,branch=experiment['branch'],
        world_size=world,recipe=str(a.recipe.resolve()),data_identity=data_identity(stats),
        window_filter_sha256=filters, loader_workers=a.loader_workers,
        input_implementation='exact-int64-voxel-lookup.v1',
        historical_filter_binding=bool(parent and 'window_filter_sha256' in parent['provenance']),
        objective_semantics=OBJECTIVE_SEMANTICS,evaluation_semantics=EVALUATION_SEMANTICS,initialization=(
            'common v3 model weights; structured C head fresh'
            if warm_start else 'all weights from scratch'),
        scope='offline nine-source pretraining Probe; robot benefit untested')
    if warm_start:
        provenance['v3_warm_start']=warm_start
    if a.resume:
        if a.resume_learning_rate_drop:
            provenance['lr_transition']=validate_lr_resume(parent,config,stats,world)
        elif parent['config']!=config or parent['stats']!=stats or parent['provenance']['world_size']!=world:
            raise ValueError('resume requires identical config/statistics/world size')
        model.load_state_dict(parent['model'],strict=True)
        restore_optimizer(optimizer,parent['optimizer'])
        first=parent['step']; previous_elapsed=parent['provenance'].get('elapsed_s',0.)
        apply_learning_rates(optimizer,config,first+1)
        provenance.update(resume_parent=str(a.resume.resolve()),parent_step=first,
            parent_git_commit=parent['provenance']['git_commit'],
            initialization='full v4 optimizer/model/per-rank RNG resume')
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
    started=time.monotonic(); curve=[]; selection=None; files=[]; drawn=Counter()
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
            restored=torch.equal(torch.get_rng_state(),parent['rank_rng'][rank]['torch']) and torch.equal(
                torch.cuda.get_rng_state(),parent['rank_rng'][rank]['cuda'])
            atomic_json(out/f'resume-rank{rank}.json',dict(rank=rank,parent_step=first,
                rng_exact=restored,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                optimizer_steps=sorted({int(v['step'].item()) for v in optimizer.state.values() if 'step' in v})))
            if not restored:raise ValueError('per-rank RNG restore failed')
        wrapped=nn.parallel.DistributedDataParallel(model,device_ids=[local],gradient_as_bucket_view=True) if world>1 else model
        dev_cache=None
        if rank==0:
            if not a.no_dev_cache:
                dev_cache=DevInputCache(dev,panel,config.get('validation_microbatch_size',8))
                atomic_json(out/'dev-cache.json',dict(bytes=dev_cache.bytes,
                    build_seconds=dev_cache.build_seconds,microbatch_size=dev_cache.microbatch_size,
                    panel=dev_cache.panel,window_filter_sha256=filters))
            baseline_started=time.monotonic()
            baseline=evaluate(model,dev,panel,config,device,input_cache=dev_cache)
            atomic_json(out/'baseline-timing.json',dict(evaluation_seconds=time.monotonic()-baseline_started))
            curve.append(dict(step=first,**baseline))
            atomic_json(out/'curve.json',curve)
            selection,baseline_best=consider_selection(parent.get('selection') if parent else None,
                baseline['objective'],first,out/('step%d.pt'%first),out.name)
        else:baseline_best=False
        decisions=[baseline_best]
        if world>1:dist.broadcast_object_list(decisions,src=0)
        if parent:
            # Startup audits/evaluation must not consume the resumed RNG stream.
            torch.set_rng_state(parent['rank_rng'][rank]['torch'])
            torch.cuda.set_rng_state(parent['rank_rng'][rank]['cuda'])
        if decisions[0]:
            states=[None]*world
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
            if world>1:dist.all_gather_object(states,rng)
            else:states=[rng]
            if rank==0:
                path=out/('step%d.pt'%first)
                provenance['elapsed_s']=previous_elapsed+time.monotonic()-started
                save_checkpoint(path,model,optimizer,config,stats,first,provenance,states,selection)
                files.append(path.name)
                checkpoint_alias(out,'best.pt',path.name);checkpoint_alias(out,'latest.pt',path.name)
        elif rank==0:
            chosen=inherit_checkpoint(out,selection['checkpoint'])
            selection['checkpoint']=str(chosen)
            atomic_json(chosen.with_suffix('.selection.json'),dict(selection=selection))
            files.append(chosen.name);checkpoint_alias(out,'best.pt',chosen.name)
            current=chosen if chosen.name==a.resume.resolve().name else inherit_checkpoint(out,a.resume)
            if current.name not in files:
                files.append(current.name)
                atomic_json(current.with_suffix('.selection.json'),dict(selection=selection))
            checkpoint_alias(out,'latest.pt',current.name)
        if rank==0:
            atomic_json(out/'selection.json',selection)
            atomic_json(out/'checkpoint_files.json',files)
        if world>1:dist.barrier()
        batches=loader(train,config,rank,first+1,target,a.loader_workers)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        logged=time.monotonic();logged_step=first
        timing=StepTimings(a.profile_steps)
        batches=iter(batches)
        for step in range(first+1,target+1):
            timing.before_batch()
            bound=next(batches)
            timing.after_batch()
            batch=to_device(bound['model_inputs'],device,non_blocking=True)
            timing.mark(1)
            drawn.update(str(p['source_id']) for p in bound['provenance'])
            # Linear warmup avoids a large normalized-target jump at startup.
            rates=apply_learning_rates(optimizer,config,step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                pred=wrapped(batch);loss,terms=forecast_loss(pred,batch,config,model.normalizer,distributed=True)
            timing.mark(2)
            if not torch.isfinite(loss):raise ValueError('nonfinite TRAIN loss')
            loss.backward()
            timing.mark(3)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
            if not torch.isfinite(norm) or norm<=0:raise ValueError('nonfinite/zero TRAIN gradient')
            optimizer.step()
            timing.mark(4)
            validation_seconds=checkpoint_seconds=0.
            checkpoint_due=step==1 or step%config['checkpoint_interval']==0 or step==target
            new_best=False
            if step%config['validation_interval']==0 or step==target:
                if rank==0:
                    validation_started=time.monotonic()
                    report=evaluate(model,dev,panel,config,device,input_cache=dev_cache)
                    validation_seconds=time.monotonic()-validation_started
                    curve.append(dict(step=step,**report));atomic_json(out/'curve.json',curve)
                    selection,new_best=consider_selection(selection,report['objective'],step,
                        out/('step%d.pt'%step),out.name)
                    atomic_json(out/'selection.json',selection)
                decision=[new_best]
                if world>1:dist.broadcast_object_list(decision,src=0)
                new_best=decision[0]
            if checkpoint_due or new_best:
                checkpoint_started=time.monotonic()
                states=[None]*world
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
                if world>1:dist.all_gather_object(states,rng)
                else:states=[rng]
                if rank==0:
                    provenance['elapsed_s']=previous_elapsed+time.monotonic()-started
                    path=out/('step%d.pt'%step)
                    save_checkpoint(path,model,optimizer,config,stats,step,provenance,states,selection)
                    checkpoint_alias(out,'latest.pt',path.name)
                    if new_best:checkpoint_alias(out,'best.pt',path.name)
                    files.append(path.name);files=prune_recovery_checkpoints(out,files,config)
                    atomic_json(out/'checkpoint_files.json',files)
                if world>1:dist.barrier()
                checkpoint_seconds=time.monotonic()-checkpoint_started
            if step==1 or step%config['log_interval']==0 or step==target:
                term_names=('hand','scene','interaction','contact','part')
                values=torch.stack([loss.detach()]+[terms[k].detach() for k in term_names])
                local_counts={k:float(v[1]) for k,v in terms['components'].items()}
                all_counts=[None]*world
                if world>1:dist.all_gather_object(all_counts,local_counts)
                else:all_counts=[local_counts]
                if world>1:
                    dist.all_reduce(values);values/=world
                torch.cuda.synchronize()
                local_timing=timing.summary()
                timing_rows=[None]*world
                if a.profile_steps:
                    if world>1:dist.all_gather_object(timing_rows,local_timing)
                    else:timing_rows=[local_timing]
                elapsed=time.monotonic()-started
                rate=(step-logged_step)/(time.monotonic()-logged)
                if rank==0:
                    record=dict(status='RUNNING',step=step,learning_rates=rates,loss=float(values[0]),
                        terms={k:float(values[i+1]) for i,k in enumerate(term_names)},
                        rank_denominators=all_counts,objective_semantics=OBJECTIVE_SEMANTICS,
                        grad_norm=float(norm),elapsed_s=elapsed,updates_per_s=rate,
                        timing_by_rank=timing_rows if any(timing_rows) else None,
                        validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,
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
