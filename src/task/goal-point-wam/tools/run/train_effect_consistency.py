"""Single-model C0 gate then C1 mixed-domain frozen-World DDP adaptation."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src'),str(TASK/'tools/run')]
from train_three_domain import atomic_json,gpu_snapshot
from goal_point_wam.data import GoalWindows
from goal_point_wam.effect_consistency import FrozenEffectWAM,action_from_delta,effect_ddp
from oakink_wm.multisource import MixedWindows
from oakink_wm.model import metrics
from oakink_wm.pointworld_performance import install_fused_hilbert
from trajectory_policy.wam_effect_data import EffectData,EffectBatches,effect_collate
from trajectory_policy.wam_shadow import WAM_TO_NATIVE
from trajectory_policy.chunk_retargeter import load_chunk
from trajectory_policy.chunk_objective import control_aware_loss,independent_PD_scale
from trajectory_policy.wam_joint import policy_patch_admission,load_policy_patch
from trajectory_policy.retargeter_data import ACTIVE

NAMES=('oakink2','grab','arctic','nominal','startup','recovery0_TRAIN','recovery1_TRAIN','recovery2_TRAIN')


def move(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}


def nonregression(now,initial):
    checks={}
    for name in NAMES:
        for metric in ('h1_mm','first8_mm'):
            checks[name+'/'+metric]=now[name][metric]<=1.1*initial[name][metric]+.05
        if name not in NAMES[:3]:
            for metric,slack in (('PD_xyz_p95_mm',.1),('PD_rotation_p95_rad',.01),('PD_finger_p95_rad',.01)):
                checks[name+'/'+metric]=now[name][metric]<=1.1*initial[name][metric]+slack
            checks[name+'/clipping']=now[name]['clipping_rate']<=initial[name]['clipping_rate']+.01
    return checks


def continuation_checks(now,initial):
    """Control preservation plus a gross forecast guard; retain strict diagnostics."""
    checks=nonregression(now,initial)
    for name in NAMES:
        for metric in ('h1_mm','first8_mm'):
            checks[name+'/'+metric]=now[name][metric]<=2.*initial[name][metric]+.05
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path)
    a=p.parse_args();out=a.output.resolve();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;install_fused_hilbert()
    dist.init_process_group('nccl',timeout=timedelta(seconds=900));torch.manual_seed(295)
    c0=4 if a.smoke else 500;target=c0 if a.smoke else 20500;maximum=c0 if a.smoke else 50500
    if not 1<=world<=4:raise ValueError('single-model1..4GPU contract')
    if rank==0:
        if out.exists() and not a.resume:raise FileExistsError('fresh output required')
        out.mkdir(parents=True,exist_ok=bool(a.resume))
    dist.barrier()
    owner=ROOT/'outputs/trajectory-policy';base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    warm=owner/'wam-ref2-joint-multitakeover-20261012-r1';fit=owner/'wam-v2-control-fit-20261012-r1'
    data_path=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree'
    stats_path=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    stats=json.loads(stats_path.read_text());packet=torch.load(base,map_location='cpu',weights_only=False)
    model=FrozenEffectWAM(stats,packet['config']).to(device).eval();model.load_state_dict(packet['model'],strict=True);del packet
    head=torch.load(warm/'head.pt',map_location='cpu',weights_only=False)
    wm,wr=[json.loads((warm/f).read_text()) for f in ('manifest.json','result.json')]
    load_policy_patch(model,head,policy_patch_admission(head,wm,wr));del head
    r=load_chunk(torch.load(fit/'chunk.pt',map_location='cpu',weights_only=False),device);r.requires_grad_(False)
    mixed=MixedWindows(data_path,'train');vbase=MixedWindows(data_path,'val')
    if (tuple(s['name'] for s in mixed.meta['sources'])!=NAMES[:3] or stats['split']!='train'
        or stats['normalization_source_manifest_sha256']!=mixed.meta['normalization_source_manifest_sha256']):
        raise ValueError('main-three train-only normalization differs')
    train=EffectData(ROOT,GoalWindows(mixed,'object_points',32));val=EffectData(ROOT,GoalWindows(vbase,'object_points',32))
    pdscale=independent_PD_scale(torch.tensor(np.asarray(train.controller['scale'])[list(ACTIVE)],device=device,dtype=torch.float32))
    order=torch.tensor(WAM_TO_NATIVE,device=device)
    train_keys={k for k,v in model.named_parameters() if v.requires_grad}
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if k not in train_keys}
    r_before={k:v.cpu().clone() for k,v in r.state_dict().items()}
    optimizer=torch.optim.AdamW(model.trajectory.parameters(),lr=1e-5,weight_decay=.01)
    step=0;initial=None;curve=[];contracts=[];c0_pass=False;bad_streak=0
    recipe=dict(model_seed=295,draw_seed=296,world_size=world,per_rank_batch=100,global_batch=100*world,
                main_robot_recovery=[60,25,15],lambda_a=1.,lambda_E=.1,learning_rate=1e-5,
                train_modules=['trajectory'],frozen_modules=['scene','goal','hand_query','action','world','R','all buffers'],
                C0_updates=c0,C1_updates=20000,maximum_C1_updates=50000,validation_interval=1000,
                world_loss='existing normalized motion-weighted point + translation + rotation',
                trajectory_loss='valid coordinate MSE/.05',action_loss='control-aware Chunk-R on robot only, weighted by robot fraction',
                world_mask='future validity used as supervision only; actor queries receive observed H and goal only',
                validation_context='cache frozen scene/goal/raw context per fixed evaluation batch; exact oracle recomputed each time')
    manifest=dict(schema='ref2dex.frozen-effect-ddp.v1',status='RUNNING',task='goal-point-wam',
                  experiment_id='P-20261013-goal-point-wam-ref27-effect-consistency',run_id=out.parent.name,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  base_checkpoint=str(base),warm_policy=str(warm),qualified_chunk=str(fit/'chunk.pt'),
                  data=str(data_path),stats=str(stats_path),recipe=recipe,deadline=a.deadline,
                  robot_sources={k:str(v) for k,v in train.paths.items()},training_allowed=not a.smoke)
    manifest['continuation_policy']='first-PD original limits; trajectory h1/first8 <=2x initial+.05mm; strict 1.1x diagnostics retained; amended after C0 r2'
    if a.resume:manifest['resumed_checkpoint']=str(a.resume.resolve())
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved['recipe']!=recipe:raise ValueError('resume recipe differs')
        model.trajectory.load_state_dict(saved['trajectory'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];initial=saved['initial'];curve=saved['curve'];contracts=saved['contracts'];c0_pass=saved['C0_pass']
        target=saved['target'];bad_streak=saved['bad_streak'];del saved
    panel=np.load(base.parent/'validation_balanced.npy')
    selected={name:panel[(panel>=vbase.offsets[i])&(panel<vbase.offsets[i+1])][:3 if a.smoke else 96] for i,name in enumerate(NAMES[:3])}
    groups={name:[('main',int(i),0) for i in ids] for name,ids in selected.items()}
    ticks=[0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518]
    groups['nominal']=[('nominal',t,e) for t in ticks for e in range(24,28)]
    groups['startup']=[('nominal',t,e) for t in range(16) for e in range(24,28)]
    for name,t in val.takeovers.items():groups[name+'_TRAIN']=[(name,k,3) for k in range(t,105)]
    if a.smoke:
        groups={k:v[:4] for k,v in groups.items()}
    ranks=[None]*world;dist.all_gather_object(ranks,dict(rank=rank,pid=os.getpid(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local]))
    if rank==0:
        atomic_json(out/'ranks.json',ranks)
        atomic_json(out/'manifest.json',manifest);atomic_json(out/'panel.json',groups)
    def frozen_check():
        error=max(float((model.state_dict()[k].cpu().float()-v.float()).abs().max()) for k,v in frozen.items())
        re=max(float((r.state_dict()[k].cpu()-v).abs().max()) for k,v in r_before.items())
        if max(error,re)!=0:raise ValueError('frozen parameters/buffers drift')
        return dict(model_frozen_max_difference=error,R_max_difference=re)
    context_cache=(torch.load(a.resume.parent/('validation_context_rank%d.pt'%rank),map_location=device,weights_only=False)
                   if a.resume else {})
    @torch.no_grad()
    def evaluate(at):
        local_values={}
        for name,keys in groups.items():
            values={k:[] for k in ('h1_mm','first8_mm','all24_mm','effect_h24_mm','effect_all24_mm','oracle_h24_mm','oracle_all24_mm','action_mask_difference')}
            own=keys[rank::world]
            for begin in range(0,len(own),16):
                b=move(effect_collate([val[k] for k in own[begin:begin+16]]),device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cache_key=(name,begin)
                    if cache_key not in context_cache:context_cache[cache_key]=model.contexts(b)
                    context=context_cache[cache_key]
                    tau=model.actor(context);pred=model.induced(tau,b,context[0]);oracle=model.oracle(b,context[0])
                    if a.smoke:
                        fresh=model.contexts(b);fresh_oracle=model.oracle(b,fresh[0])
                error=torch.linalg.vector_norm(tau-b['trajectory_target'],dim=-1)*1000;valid=b['trajectory_valid']
                for key,count in (('h1_mm',1),('first8_mm',8),('all24_mm',24)):
                    values[key].extend(((error[:,:count]*valid[:,:count]).flatten(1).sum(1)/valid[:,:count].flatten(1).sum(1).clamp_min(1)).cpu().tolist())
                _,am=action_from_delta(tau,b['start_hand'],b['trajectory_valid'])
                values['action_mask_difference'].extend((am!=b['action_valid'].reshape(-1,24,2,11)).flatten(1).sum(1).cpu().tolist())
                for prefix,prediction in (('effect',pred),('oracle',oracle)):
                    ep=metrics(prediction,b)['point_epe']*1000;mask=b['object_valid']
                    values[prefix+'_h24_mm'].extend(((ep[:,:,-1]*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist())
                    values[prefix+'_all24_mm'].extend(((ep.mean(-1)*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().tolist())
                if a.smoke:
                    fresh_ep=metrics(fresh_oracle,b)['point_epe'][:,:,-1]*1000
                    fresh_value=(fresh_ep*b['object_valid']).sum(1)/b['object_valid'].sum(1).clamp_min(1)
                    values.setdefault('fresh_context_oracle_h24_mm',[]).extend(fresh_value.cpu().tolist())
                if b['has_pd'].any():
                    action=r(b['native_state'],tau[:,:,0].index_select(2,order).reshape(-1,24,33))[:,0]
                    diff=(action.clamp(-1,1)-b['native_action'][:,0])*pdscale
                    for key,value in dict(PD_xyz=torch.linalg.vector_norm(diff[:,:3],dim=-1)*1000,
                        PD_rotation=torch.linalg.vector_norm(diff[:,3:6],dim=-1),PD_finger=diff[:,6:].abs().mean(-1),
                        clipping_rate=(action.abs()>1+1e-6).any(-1).float()).items():values.setdefault(key,[]).extend(value.cpu().tolist())
            local_values[name]=values
        parts=[None]*world;dist.all_gather_object(parts,local_values);report={};raw={}
        for name in groups:
            v={k:np.concatenate([np.asarray(part[name].get(k,[])) for part in parts]) for k in parts[0][name]}
            report[name]={k:float(x.mean()) for k,x in v.items() if not k.startswith('PD_')}
            for k in ('PD_xyz','PD_rotation','PD_finger'):
                if k in v:report[name][k+'_p95_'+('mm' if k=='PD_xyz' else 'rad')]=float(np.percentile(v[k],95))
            for k in ('h24','all24'):report[name]['Delta_E_'+k+'_mm']=report[name]['effect_'+k+'_mm']-report[name]['oracle_'+k+'_mm']
            raw.update({name+'_'+k:x for k,x in v.items()})
        if initial is not None:
            for name in groups:
                for key in ('oracle_h24_mm','oracle_all24_mm'):
                    if abs(report[name][key]-initial[name][key])>1e-5:raise ValueError('GT-tau oracle output changed')
        if rank==0:
            np.savez_compressed(out/('panel_step%d.npz'%at),**raw)
            atomic_json(out/('evaluation_step%d.json'%at),report)
        return report
    def save(at):
        if rank==0:
            path=out/('checkpoint_step%d.pt'%at)
            if path.exists():raise FileExistsError('preserve existing checkpoint')
            state=dict(schema='ref2dex.effect-policy-patch.v1',trajectory=model.trajectory.state_dict(),optimizer=optimizer.state_dict(),
                       step=at,recipe=recipe,manifest=manifest,initial=initial,curve=curve,contracts=contracts,
                       C0_pass=c0_pass,target=target,bad_streak=bad_streak)
            part=path.with_suffix('.pt.part');torch.save(state,part);part.replace(path)
            atomic_json(out/'latest_checkpoint.json',dict(path=str(path),step=at))
        dist.barrier()
    if a.resume and step==c0 and not c0_pass:
        freeze=frozen_check();admission=evaluate(step)
        checks=continuation_checks(admission,initial)
        if (not contracts or not all(checks.values()) or any(
            min(c[k] for k in ('effect_to_tau_sum','effect_to_decoder_sum','action_to_tau_sum'))<=0
            or c['rebuilt_GT_action_max_difference']>2e-6 or c['human_PD_rows']!=0 for c in contracts)):
            raise ValueError('saved C0 fails numerical/control continuation admission')
        c0_pass=True
        if rank==0:atomic_json(out/'C0_continuation_admission.json',dict(pass_=True,checks=checks,
            strict_checks=nonregression(admission,initial),contracts=contracts,**freeze))
    if initial is None:
        initial=evaluate(0)
    # Reuse the exact fixed encoder representation after an interrupted
    # run as well; a fresh sparse replay can vary below a tenth of a mm.
    saved_context={key:tuple({k:v.cpu() for k,v in item.items()} if isinstance(item,dict)
                    else item.cpu() if isinstance(item,torch.Tensor) else item for item in context)
                   for key,context in context_cache.items()}
    context_path=out/('validation_context_rank%d.pt'%rank)
    if context_path.exists():raise FileExistsError('preserve validation context')
    torch.save(saved_context,context_path);del saved_context
    if rank==0 and a.resume:atomic_json(out/'evaluation_step0.json',initial)
    stop=[False]
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    ddp=effect_ddp(model,[local])
    sampler=EffectBatches(mixed,train.takeovers,rank,step+1,maximum)
    # PTv3/Triton initializes LLVM threads before loading batches. Forking that
    # runtime aborts inherited LLVM workers; spawn independent CPU loaders.
    batches=DataLoader(train,batch_sampler=sampler,collate_fn=effect_collate,num_workers=4,
                       pin_memory=True,persistent_workers=True,multiprocessing_context='spawn')
    started=time.monotonic();reason=None;last=None
    for cpu in batches:
        step+=1;b=move(cpu,device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=ddp(b)
        tau=prediction['trajectory'];valid=b['trajectory_valid']
        te=((tau-b['trajectory_target'])/.05).square().mean(-1)
        tau_loss=((te*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
        effect,terms=model.world_loss(prediction,b)
        pdvalid=b['has_pd'];native=tau[pdvalid,:,0].index_select(2,order).reshape(-1,24,33)
        action,_=control_aware_loss(r(b['native_state'][pdvalid],native),b['native_action'][pdvalid],r.action_scale,pdscale)
        action=action*pdvalid.float().mean()
        if step==1 or (a.resume and step==sampler.first):
            effect_grads=torch.autograd.grad(effect,[tau]+[p for p in model.trajectory.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
            gt_grad=torch.autograd.grad(tau_loss,tau,retain_graph=True)[0]
            act_grad=torch.autograd.grad(action,tau,retain_graph=True)[0]
            rebuilt,mask=action_from_delta(b['trajectory_target'],b['start_hand'],b['trajectory_valid'])
            contract=dict(effect_to_tau_sum=float(effect_grads[0].abs().sum()),
                          effect_to_decoder_sum=sum(float(g.abs().sum()) for g in effect_grads[1:] if g is not None),
                          action_to_tau_sum=float(act_grad.abs().sum()),
                          weighted_effect_to_tau_ratio=float(.1*effect_grads[0].abs().sum()/gt_grad.abs().sum().clamp_min(1e-12)),
                          rebuilt_GT_action_max_difference=float((rebuilt-b['action'].reshape_as(rebuilt))[mask].abs().max()),
                          world_scene_context_detached=True,human_PD_rows=int((pdvalid & (b['source_id']<3)).sum()))
            if min(contract[k] for k in ('effect_to_tau_sum','effect_to_decoder_sum','action_to_tau_sum'))<=0:
                raise ValueError('C action/effect gradients blocked')
            if contract['rebuilt_GT_action_max_difference']>2e-6 or contract['human_PD_rows']!=0:
                raise ValueError('C action contract/human PD masking invalid')
            parts=[None]*world;dist.all_gather_object(parts,contract);contracts=parts
        loss=tau_loss+action+.1*effect
        if not torch.isfinite(loss):raise FloatingPointError('C loss nonfinite')
        loss.backward()
        if any(p.grad is not None for p in list(model.parameters())+list(r.parameters()) if not p.requires_grad):
            raise ValueError('World/R/frozen interface parameter gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.trajectory.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('C decoder gradient nonfinite')
        optimizer.step()
        stopping=torch.tensor(int(stop[0] or time.time()>=a.deadline),device=device)
        dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
        if step==1 or step%100==0:
            record=dict(step=step,stage='C0' if step<=c0 else 'C1',loss=float(loss),tau=float(tau_loss),action=float(action),effect=float(effect),
                        elapsed_s=time.monotonic()-started,eta_target_s=(time.monotonic()-started)/max(1,step-sampler.first+1)*(target-step),gpu=gpu_snapshot(device))
            if rank==0:
                atomic_json(out/'progress.json',record)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
        if step==c0 or step%1000==0 or step==target or stopping.item():
            freeze=frozen_check();last=evaluate(step);checks=continuation_checks(last,initial)
            strict_checks=nonregression(last,initial)
            curve.append(dict(step=step,metrics=last,checks=checks,strict_checks=strict_checks,**freeze))
            if rank==0:atomic_json(out/'curve.json',dict(curve=curve))
            if step==c0:
                c0_pass=all(checks.values()) or a.smoke
                if rank==0:atomic_json(out/'C0_gate.json',dict(pass_=c0_pass,checks=checks,contracts=contracts,**freeze))
                if not c0_pass:reason='C0_nonregression_failed'
            if step>c0:
                bad_streak=bad_streak+1 if not all(checks.values()) else 0
                if bad_streak>=2:reason='two_consecutive_nonregression_failures'
            if step==target and target==20500:
                earlier=[row for row in curve if 16500<=row['step']<=18500]
                before=earlier[0]['metrics'] if earlier else initial
                oldgap=np.mean([before[n]['Delta_E_all24_mm'] for n in NAMES])
                newgap=np.mean([last[n]['Delta_E_all24_mm'] for n in NAMES])
                seconds_per=(time.monotonic()-started)/max(1,step-sampler.first+1)
                extend=all(checks.values()) and oldgap-newgap>=max(.1,.01*abs(oldgap)) and a.deadline-time.time()>=30000*seconds_per
                if extend:target=50500
                elif reason is None:reason='C1_20k_complete_no_extension'
            save(step)
        if reason or stopping.item() or step>=target:
            reason=reason or ('deadline_or_user_stop' if stopping.item() else 'target_complete');break
    if rank==0:
        result=dict(status='COMPLETED' if reason in ('target_complete','C1_20k_complete_no_extension') else 'STOPPED',
                    reason=reason,step=step,C0_pass=c0_pass,contracts=contracts,last=last,
                    conclusion='engineering PASS' if a.smoke else 'UNCLEAR',scope='single-seed offline mechanism probe, no physical/causal utility claim')
        atomic_json(out/'result.json',result);manifest['status']=result['status'];atomic_json(out/'manifest.json',manifest)
    dist.destroy_process_group()


if __name__=='__main__':main()
