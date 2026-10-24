"""Pre-registered diagnosis arms; explicit weights, batch and flow-only flag."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from remote_staged_data import read,write,sha,GIB,NAS
from diagnostic_components import DiagnosticWindows,validate_diagnosis
from ptv3_shared_wam_v6.prepared_reader import collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.precision import configure_strict_fp32
from ptv3_shared_wam_v6.objectives import forecast_loss
from ptv3_shared_wam_v6.goal_policy import assert_unknown_inputs
from ptv3_shared_wam_v5.training_contract import loader_options,restore_optimizer_state
from evaluate_no_goal_no_event_panel import cluster_ci
from train_pretrain import configure_training_ipc


class Steps:
    def __init__(self,dataset,first,last,rank):self.dataset,self.first,self.last,self.rank=dataset,first,last,rank
    def __iter__(self):
        for step in range(self.first,self.last+1):yield self.dataset.draw(96,170,step,self.rank)
    def __len__(self):return self.last-self.first+1


def panel(model,panel_manifest,directory,step,reference=None):
    meta=read(panel_manifest);base=Path(panel_manifest).parent
    sums=[];counts=[];repeated=None;before=model.training;model.eval()
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    # Torch2.0's efficient kernel has non-bit-exact repeated inference. Math
    # attention evaluates the same FP32 equations and is bit-exact here. This
    # scorer backend is declared separately from the efficient training backend.
    panel_config=dict(model.config,fp32_attention_backend='math')
    configure_strict_fp32(model,panel_config)
    with torch.inference_mode():
        for row in meta['files']:
            p=base/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
            with np.load(p,allow_pickle=False) as z:batch={k:torch.from_numpy(z[k]).cuda() for k in z.files}
            assert_unknown_inputs(batch);pred=model(batch)
            if repeated is None:
                repeat=model(batch);repeated=bool(torch.equal(pred['trajectory'],repeat['trajectory']) and torch.equal(pred['scene'],repeat['scene']))
            hv=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
            sv=batch['scene_valid']&batch['object_valid'][:,:,None,None]
            cf=(batch['interaction_valid']&(batch['contact_target']>.25)).any(dim=3).any(dim=1)
            first=torch.arange(24,device='cuda')[None]<8
            masks=[hv&first[:,:,None,None],hv&cf[:,:,None,None],sv&first[:,None,:,None]]
            errors=[(pred['trajectory']-batch['trajectory_target']).norm(dim=-1)*1000]*2+[(pred['scene']-batch['scene_target']).norm(dim=-1)*1000]
            sums.append(torch.stack([torch.where(m,e,0.).flatten(1).sum(1,dtype=torch.float64) for e,m in zip(errors,masks)],1).cpu().numpy())
            counts.append(torch.stack([m.flatten(1).sum(1) for m in masks],1).cpu().numpy())
    torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng);model.train(before)
    configure_strict_fp32(model,model.config)
    sums=np.concatenate(sums);counts=np.concatenate(counts);mothers=np.asarray([r['sequence_id'] for r in meta['rows']])
    assert len(sums)==512 and len(np.unique(mothers))==124 and counts.sum(0).tolist()==meta['contract_counts']
    assert np.isfinite(sums).all() and repeated
    means=sums.sum(0)/counts.sum(0);comparison=None
    if reference:
        with np.load(reference,allow_pickle=False) as z:
            assert np.array_equal(z['counts'],counts) and np.array_equal(z['mothers'],mothers)
            delta=means-z['sums'].sum(0)/counts.sum(0)
            ci=cluster_ci(sums,counts,mothers,reference=z['sums'])
        comparison=dict(delta_mm=delta.tolist(),paired_mother_cluster_ci95_mm=ci,
            winner=['remote' if bounds[1]<0 else 'local' if bounds[0]>0 else 'uncertain' for bounds in ci])
    directory.mkdir(exist_ok=False)
    np.savez(directory/'paired-panel.npz',sums=sums,counts=counts,mothers=mothers)
    (directory/'paired-panel.npz').chmod(0o664)
    result=dict(status='PASS',step=step,means_mm=means.tolist(),cluster_ci95_mm=cluster_ci(sums,counts,mothers),
        metrics_order=meta['metrics_order'],rows=512,mothers=124,bootstrap_repeats=4000,bootstrap_seed=170,
        anchor_sha256=meta['anchor_sha256'],same_A_conditions=True,repeat_bit_exact=repeated,
        panel_attention_backend='math',training_attention_backend=model.config['fp32_attention_backend'],
        versus_local3000=comparison,scope='fixed TRAIN panel only; no generalization or execution conclusion')
    write(directory/'summary.json',result,immutable=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--expand-from',type=Path);a=p.parse_args()
    recipe=validate_diagnosis(read(a.recipe),smoke=a.smoke);config=read(recipe['config']);stats=read(recipe['stats'])
    assert a.recipe.resolve()==Path(recipe['recipe_path']).resolve() and NAS in a.output.resolve().parents
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if world!=recipe['world_size']:raise ValueError('declared world size required')
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(170+rank);np.random.seed(170+rank);random.seed(170+rank)
    # Torch2.0 rejects even unweighted integer CUDA bincount. Request its
    # deterministic paths, retaining warnings for unsupported integer counts;
    # the fixed panel independently requires actual bit-exact repeats.
    legacy_torch=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))<(2,1)
    torch.use_deterministic_algorithms(True,warn_only=legacy_torch)
    model=SharedWAMV6(stats,config).cuda()
    if recipe.get('flow_only',False):
        for name,param in model.named_parameters():param.requires_grad_(name.startswith('flow_future.'))
    optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),weight_decay=config['weight_decay'],fused=True)
    first=1;parent=None
    if a.expand_from:
        saved=torch.load(a.expand_from,map_location='cpu',weights_only=False)
        if recipe['stage']!='full' or saved.get('stage')!='half' or saved.get('schema')!='ref2dex.remote-staged-checkpoint.v1':
            raise ValueError('only explicitly declared half-to-full expansion allowed')
        campaign_root=Path(recipe['manifest']).parent.parent
        if (saved['data_identity']!=sha(campaign_root/'half-census/manifest.json')
                or saved['recipe_sha256']!=sha(campaign_root/'half-recipe/recipe.json')
                or read(recipe['manifest'])['plan_sha256']!=read(campaign_root/'half-census/manifest.json')['plan_sha256']):
            raise ValueError('expansion must restore this campaign frozen half and selection plan')
        parent=dict(checkpoint=str(a.expand_from),sha256=sha(a.expand_from),data_identity=saved['data_identity'],
            recipe_sha256=saved['recipe_sha256'],new_data_identity=recipe.get('data_identity_sha256',recipe['manifest_sha256']),
            source_probabilities_before=saved['source_probabilities'],source_probabilities_after=recipe['source_probabilities'],
            optimizer_restored=True,identity_changed=True)
        model.load_state_dict(saved['model'],strict=True);restore_optimizer_state(optimizer,saved['optimizer'])
        state=saved['random_state'][rank];random.setstate(state['python']);np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda']);first=saved['step']+1
        del saved
    else:
        saved=torch.load(recipe['initialization'],map_location='cpu',weights_only=False)
        assert saved['step']==3000
        assert sha(recipe['initialization'])==recipe['initialization_sha256']
        for key,value in model.normalizer.state_dict().items():
            assert torch.equal(value.cpu(),saved['model']['normalizer.'+key].cpu()),'TRAIN normalizer changed'
        assert all(torch.isfinite(v).all() for v in saved['model'].values() if v.is_floating_point())
        model.load_state_dict(saved['model'],strict=True);del saved
        parent=dict(checkpoint=recipe['initialization'],source_step=3000,mode='weights-only',optimizer_restored=False)
    wrapped=DDP(model,device_ids=[local],find_unused_parameters=False);last=first if a.smoke else recipe['last_step']
    dataset=DiagnosticWindows(recipe['manifest'],'train',recipe['geometry_manifest'],recipe['manifest_sha256'],probabilities=recipe['source_probabilities'],curated=recipe.get('curated_manifest'),validity_gate=recipe.get('validity_gate'))
    configure_training_ipc()
    loader=DataLoader(dataset,batch_sampler=Steps(dataset,first,last,rank),collate_fn=collate,num_workers=2,
        pin_memory=True,generator=torch.Generator().manual_seed(170*1009+rank),**loader_options(2,180,'spawn'))
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    recipe_sha256=sha(a.recipe) if rank==0 else None
    if rank==0:
        if (out/'run-manifest.json').exists():raise FileExistsError('fresh run output required')
        write(out/'run-manifest.json',dict(recipe=recipe,recipe_sha256=recipe_sha256,parent=parent,
            world_size=world,local_batch=96,global_batch=recipe['global_batch'],git_commit=recipe['git_commit'],
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,smoke=a.smoke,automatic_restart=False,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
            environment={k:os.environ.get(k) for k in ('TMPDIR','CONDA_PREFIX','CUDA_VISIBLE_DEVICES','REF2DEX_RUN_ROOT')}),immutable=True)
    dist.barrier();started=time.monotonic();step=first-1;draws=Counter();loss_records=[];preclip_norms=[];last_saved=None
    reference=recipe.get('local3000_panel')

    def save(current):
        nonlocal last_saved
        state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
        states=[None]*world;dist.all_gather_object(states,state)
        if rank==0:
            target=out/('step%06d.pt'%current);assert not target.exists()
            pending=target.with_suffix('.pt.part')
            digest,reread_error=recipe_sha256,None
            for _ in range(5):
                try:
                    digest=sha(a.recipe);reread_error=None;break
                except PermissionError as exc:
                    reread_error=repr(exc);time.sleep(1)
            torch.save(dict(schema='ref2dex.remote-staged-checkpoint.v1',stage=recipe['stage'],step=current,
                data_identity=recipe.get('data_identity_sha256',recipe['manifest_sha256']),recipe_sha256=digest,recipe_reread_error=reread_error,parent=parent,
                source_probabilities=recipe['source_probabilities'],config=config,model=model.state_dict(),
                optimizer=optimizer.state_dict(),random_state=states,elapsed_s=time.monotonic()-started),pending)
            pending.replace(target);target.chmod(0o664)
        last_saved=current;dist.barrier()

    def evaluate(current):
        dist.barrier()
        if rank==0:
            result=panel(model,recipe['panel_manifest'],out/('panel-step%06d'%current),current,reference)
            print(json.dumps(dict(event='PANEL',**result)),flush=True)
        dist.barrier()

    if not a.smoke:
        evaluate(first-1)
    iterator=iter(loader);model.train();torch.cuda.reset_peak_memory_stats()
    started=time.monotonic() # 40 minute update budget starts after initial panel
    try:
        for step in range(first,last+1):
            if (out/'STOP').exists():raise RuntimeError('owner or watchdog STOP')
            if shutil.disk_usage(out).free<30*GIB:raise RuntimeError('disk floor')
            expired=torch.tensor(int(time.monotonic()-started>recipe['max_seconds']),device='cuda');dist.all_reduce(expired,op=dist.ReduceOp.MAX)
            if expired.item():
                step-=1;break
            begin=time.monotonic();batch=next(iterator)
            draws.update(map(int,batch['source_id'].tolist()))
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()};assert_unknown_inputs(batch)
            optimizer.zero_grad(set_to_none=True)
            warmup=min(1.,step/config['warmup_updates'])
            for group in optimizer.param_groups:
                group['lr']=(config['flow_lr'] if group['name']=='conditional_flow' else config['geometry_lr'] if group['name']=='joint_geometry' else config['shared_lr'])*warmup
            prediction=wrapped(batch);loss,terms=forecast_loss(prediction,batch,config,model.normalizer,True,step=step)
            finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'],error_if_nonfinite=True)
            if rank==0:preclip_norms.append(float(norm))
            optimizer.step();torch.cuda.synchronize()
            average=loss.detach().clone();dist.all_reduce(average);average/=world
            value=float(average);loss_records.append(value)
            if rank==0 and (step<=first+2 or step%20==0):
                record=dict(status='RUNNING',stage=recipe['stage'],step=step,stage_update=step-first+1,loss=value,
                    grad_norm=float(norm),elapsed_s=time.monotonic()-started,step_seconds=time.monotonic()-begin,
                    eta_seconds=(last-step)*(time.monotonic()-started)/(step-first+1),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,source_draws_rank0=dict(draws),
                    global_batch=recipe['global_batch'],data_identity=recipe.get('data_identity_sha256',recipe['manifest_sha256']))
                write(out/'progress.json',record)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            if not a.smoke and (step%500==0 or step==last):save(step)
            if not a.smoke and (step%500==0 or step==last):evaluate(step)
        if not a.smoke and last_saved!=step:save(step)
        if not a.smoke and step%500!=0 and step!=last:evaluate(step)
        if rank==0:
            block_means=[float(np.mean(loss_records[i:i+500])) for i in range(0,len(loss_records),500) if len(loss_records[i:i+500])==500]
            monotone=all(b<=a for a,b in zip(block_means,block_means[1:])) if len(block_means)>=2 else None
            primary=None
            if not a.smoke:
                with np.load(out/('panel-step%06d/paired-panel.npz'%(first-1))) as initial, np.load(out/('panel-step%06d/paired-panel.npz'%step)) as final:
                    assert np.array_equal(initial['counts'],final['counts']) and np.array_equal(initial['mothers'],final['mothers'])
                    delta=(final['sums'].sum(0)-initial['sums'].sum(0))/final['counts'].sum(0)
                    ci=cluster_ci(final['sums'],final['counts'],final['mothers'],reference=initial['sums'])
                    primary=dict(delta_vs_stage_initial_mm=delta.tolist(),paired_cluster_ci95_mm=ci,
                        all_three_improve_with_ci=bool((delta<0).all() and all(bounds[1]<0 for bounds in ci)),
                        loss_monotone=monotone,pass_all=bool((delta<0).all() and all(bounds[1]<0 for bounds in ci) and monotone is True))
            write(out/'result.json',dict(status='SMOKE_PASS' if a.smoke else 'COMPLETED',last_step=step,
                updates=step-first+1,elapsed_s=time.monotonic()-started,loss_block500_means=block_means,
                loss_block500_monotone_nonincreasing=monotone,primary=primary,
                preclip_grad_norm=preclip_norms,preclip_grad_norm_gt1=sum(v>1 for v in preclip_norms),
                automatic_restart=False),immutable=True)
        write(out/'progress.json',dict(status='COMPLETED',step=step,updates=step-first+1,elapsed_s=time.monotonic()-started,global_batch=recipe['global_batch'],automatic_restart=False)) if rank==0 else None
        dist.barrier()
    except BaseException as exc:
        write(out/('failure-rank%d.json'%rank),dict(reason=repr(exc),last_attempted_step=step,automatic_restart=False),immutable=True)
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
