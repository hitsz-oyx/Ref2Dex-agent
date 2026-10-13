"""Bounded current-state actor realignment to a separately qualified fixed R."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from train_control_context import (ROOT,FIELDS,SCHEMA,ControlContextWAM,GoalWindows,MixedWindows,
    control_collate,load_policy_patch,policy_patch_admission,effect_admission,chunk_admission,
    load_chunk,independent_PD_scale,control_aware_loss,ACTIVE,WAM_TO_NATIVE,concatenate,subset,
    atomic_json,gpu_snapshot,install_fused_hilbert)
from trajectory_policy.visited_control_data import VisitedControlData,collection_entries
from trajectory_policy.wam_retargeter_admission import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collections',nargs='+',type=Path,required=True)
    p.add_argument('--r-fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3);p.add_argument('--updates',type=int,default=3000)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/goal-point-wam' not in out.parents or not 1<=a.updates<=3000:raise ValueError('fresh bounded actor run required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(2);torch.manual_seed(295);torch.backends.cuda.matmul.allow_tf32=True
    torch.use_deterministic_algorithms(True);install_fused_hilbert();started=time.monotonic()
    owner=ROOT/'outputs/trajectory-policy';base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    warm=owner/'wam-ref2-joint-multitakeover-20261012-r1';c=ROOT/'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'
    parent=ROOT/'outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt'
    corpus=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree'
    state=torch.load(base,map_location='cpu',weights_only=False)
    model=ControlContextWAM(json.loads((corpus.parent/'norm_stats.json').read_text()),state['config']).cuda().eval()
    adapter={k:v for k,v in model.state_dict().items() if k.startswith('control_adapter.')}
    model.load_state_dict(dict(state['model'],**adapter),strict=True);del state,adapter
    h=torch.load(warm/'head.pt',map_location='cpu',weights_only=False)
    load_policy_patch(model,h,policy_patch_admission(h,json.loads((warm/'manifest.json').read_text()),json.loads((warm/'result.json').read_text())));del h
    identity,_=effect_admission(c,warm,owner/'wam-v2-control-fit-20261012-r1')
    packet=torch.load(parent,map_location='cpu',weights_only=False)
    if packet['manifest']['C_identity']!=identity:raise ValueError('parent C mismatch')
    model.trajectory.load_state_dict(packet['trajectory']);model.control_adapter.load_state_dict(packet['adapter']);del packet
    chunk,ri,_=chunk_admission(a.r_fit)
    if not json.loads((a.r_fit/'result.json').read_text())['offline_gate']:raise ValueError('R offline gate required')
    r=load_chunk(torch.load(chunk,map_location='cpu',weights_only=False),'cuda').requires_grad_(False)
    entries=collection_entries(a.collections);mixed=MixedWindows(corpus,'train');valbase=MixedWindows(corpus,'val')
    data=VisitedControlData(ROOT,GoalWindows(mixed,'object_points',32),entries)
    val=VisitedControlData(ROOT,GoalWindows(valbase,'object_points',32),entries)
    train_names=[n for n in data.takeovers if data.families[n]!=3];dev_names=[n for n in data.takeovers if data.families[n]==3]
    if not train_names or not dev_names:raise ValueError('whole-prefix TRAIN/DEV required')
    ticks=sorted(set(range(16))|set(range(16,128,4))|set(range(128,519,16)))
    nominal=[('nominal',t,e) for t in ticks for e in range(16)]
    exact=[(n,data.takeovers[n],3) for n in train_names]
    continuation=[(n,t,3) for n in train_names for t in data.valid_ticks[n] if t>data.takeovers[n]]
    human=[];rng=np.random.default_rng(296)
    for sid in range(3):human.extend(('main',int(i),0) for i in rng.choice(np.arange(mixed.offsets[sid],mixed.offsets[sid+1]),64,replace=False))
    keys=dict(nominal=nominal,exact=exact,continuation=continuation,human=human)
    groups=dict(nominal=[('nominal',t,e) for t in ticks for e in range(24,28)],startup=[('nominal',t,e) for t in range(16) for e in range(24,28)])
    for split,names in [('train',train_names),('dev',dev_names)]:
        groups['exact_'+split]=[(n,data.takeovers[n],3) for n in names]
        groups['contact_'+split]=[(n,t,3) for n in names for t in data.valid_ticks[n] if t<data.takeovers[n]+16]
    panel=np.load(base.parent/'validation_balanced.npy')
    for sid,name in enumerate(('oakink2','grab','arctic')):
        groups[name]=[('main',int(i),0) for i in panel[(panel>=valbase.offsets[sid])&(panel<valbase.offsets[sid+1])][:16]]
    out.mkdir();manifest=dict(schema=SCHEMA,status='RUNNING',experiment_id='P-20261013-goal-point-wam-visited-realign',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),C_identity=identity,parent_control=str(parent),retargeter_identity=ri,training_R_checkpoint=str(chunk),training_R_sha256=sha(chunk),alignment=True,collections=list(map(str,a.collections)),training_allowed=True,maximum_updates=a.updates,physical_gpu=a.gpu,loss='trajectory MSE/.05 + fixed-R control-aware + .1 World effect',mix=dict(nominal=30,human=20,exact=25,continuation=25),excluded=data.excluded)
    atomic_json(out/'manifest.json',manifest);atomic_json(out/'training_keys.json',keys);atomic_json(out/'panel.json',groups)
    @torch.no_grad()
    def cache(dataset,items):
        batches=[];contexts=[]
        for start in range(0,len(items),16):
            b={k:v.cuda() for k,v in control_collate([dataset[k] for k in items[start:start+16]]).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):context=model.contexts(b)
            batches.append({k:b[k] for k in FIELDS});contexts.append(context)
        return concatenate(contexts),concatenate(batches)
    pools={name:cache(data,ks) for name,ks in keys.items()};panels={name:cache(val,ks) for name,ks in groups.items()}
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith(('trajectory.','control_adapter.'))}
    oldr={k:v.cpu().clone() for k,v in r.state_dict().items()}
    scale=independent_PD_scale(torch.tensor(np.asarray(data.controller['scale'])[list(ACTIVE)],dtype=torch.float32,device='cuda'))
    order=torch.tensor(WAM_TO_NATIVE,device='cuda')
    def freeze_check():
        f=max(float((model.state_dict()[k].cpu().float()-v.float()).abs().max()) for k,v in frozen.items())
        rd=max(float((r.state_dict()[k].cpu()-v).abs().max()) for k,v in oldr.items())
        if f!=0 or rd!=0:raise ValueError('frozen module drift')
        return dict(frozen_max_difference=f,R_max_difference=rd)
    @torch.no_grad()
    def evaluate(step):
        reports={};raw={}
        for name,(context,batch) in panels.items():
            vals={k:[] for k in ('first8_mm','all24_mm','PD_xyz_mm','clipping')};taus=[]
            for start in range(0,len(batch['proprio']),16):
                sl=slice(start,start+16);b,ctx=subset(batch,sl),subset(context,sl)
                with torch.autocast('cuda',dtype=torch.bfloat16):tau=model.actor(ctx,b)
                taus.append(tau.cpu().numpy());error=(tau-b['trajectory_target']).norm(dim=-1)*1000;valid=b['trajectory_valid']
                for key,count in [('first8_mm',8),('all24_mm',24)]:vals[key].extend(((error[:,:count]*valid[:,:count]).flatten(1).sum(1)/valid[:,:count].flatten(1).sum(1).clamp_min(1)).cpu().tolist())
                if b['has_pd'].any():
                    pred=r(b['native_state'],tau[:,:,0].index_select(2,order).reshape(-1,24,33))[:,0]
                    delta=(pred.clamp(-1,1)-b['native_action'][:,0])*scale
                    vals['PD_xyz_mm'].extend((delta[:,:3].norm(dim=-1)*1000).cpu().tolist());vals['clipping'].extend((pred.abs()>1+1e-6).any(-1).float().cpu().tolist())
            reports[name]={k:float(np.mean(v)) for k,v in vals.items() if v}
            if vals['PD_xyz_mm']:reports[name]['PD_p95_mm']=float(np.percentile(vals['PD_xyz_mm'],95))
            raw.update({name+'_'+k:np.asarray(v) for k,v in vals.items() if v});raw[name+'_tau']=np.concatenate(taus)
        atomic_json(out/f'evaluation_step{step}.json',reports);np.savez_compressed(out/f'panel_step{step}.npz',**raw);return reports
    initial=evaluate(0);model.set_stage(2)
    optimizer=torch.optim.AdamW([{'params':model.control_adapter.parameters(),'lr':1e-4},{'params':[p for p in model.trajectory.parameters() if p.requires_grad],'lr':1e-5}],weight_decay=.01)
    curve=[];best=None;bad=0;train_start=time.monotonic()
    try:
        for step in range(1,a.updates+1):
            optimizer.zero_grad(set_to_none=True);draw=np.random.default_rng(np.random.SeedSequence([296,step]));total=0.
            for name,count in manifest['mix'].items():
                context,batch=pools[name];ids=torch.tensor(draw.integers(0,len(batch['proprio']),count),device='cuda');b,ctx=subset(batch,ids),subset(context,ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):tau=model.actor(ctx,b);effect=model.induced(tau,b,ctx[0])
                error=((tau-b['trajectory_target'])/.05).square().mean(-1);valid=b['trajectory_valid']
                tl=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean();el,_=model.world_loss(effect,b)
                al=torch.zeros((),device='cuda')
                if name!='human':al,_=control_aware_loss(r(b['native_state'],tau[:,:,0].index_select(2,order).reshape(-1,24,33)),b['native_action'],r.action_scale,scale)
                loss=(tl+al+.1*el)*count/100
                if not torch.isfinite(loss):raise FloatingPointError('actor loss')
                loss.backward();total+=float(loss)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('actor gradient')
            optimizer.step()
            if step==1 or step%100==0:
                row=dict(step=step,loss=total,elapsed_s=time.monotonic()-started,eta_s=(time.monotonic()-train_start)/step*(a.updates-step),gpu=gpu_snapshot(torch.device('cuda')));atomic_json(out/'progress.json',row);print(json.dumps(row),flush=True)
            if step%500==0 or step==a.updates:
                report=evaluate(step);checks={}
                for metric in ('first8_mm','PD_xyz_mm'):checks['exact_train/'+metric]=report['exact_train'][metric]<=.9*initial['exact_train'][metric]
                for name in ('exact_dev','contact_dev'):
                    for metric in ('first8_mm','PD_p95_mm'):checks[name+'/'+metric]=report[name][metric]<=1.1*initial[name][metric]+.1
                for name in ('nominal','startup'):checks[name+'/PD']=report[name]['PD_p95_mm']<=1.1*initial[name]['PD_p95_mm']+.1
                for name in ('oakink2','grab','arctic'):checks[name+'/trajectory']=report[name]['all24_mm']<=1.1*initial[name]['all24_mm']+.1
                for name in ('exact_train','contact_train','exact_dev','contact_dev','nominal','startup'):checks[name+'/clip']=report[name]['clipping']<=.01
                contracts=freeze_check();score=sum(report[n][k]/max(initial[n][k],.1) for n in ('exact_train','exact_dev') for k in ('first8_mm','PD_xyz_mm'))/4
                curve.append(dict(step=step,reports=report,checks=checks,score=score,**contracts));atomic_json(out/'curve.json',curve)
                key=(not all(checks.values()),score)
                if best is None or key<best[0]:
                    best=(key,step,report,checks);torch.save(dict(schema=SCHEMA,adapter=model.control_adapter.state_dict(),trajectory=model.trajectory.state_dict(),manifest=manifest,stage=2,step=step,contracts=contracts,optimizer=optimizer.state_dict()),out/'selected.pt')
                bad=bad+1 if any(report[n][k]>2*initial[n][k]+.1 for n in ('exact_train','exact_dev') for k in ('first8_mm','PD_xyz_mm')) else 0
                if bad>=2:raise RuntimeError('two gross contact regressions')
                pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if pids-{os.getpid()}:raise RuntimeError('foreign GPU process')
            if time.monotonic()-started>1200:raise TimeoutError('actor1200s budget')
        result=dict(status='COMPLETED',offline_gate=all(best[3].values()),selected_update=best[1],final_checkpoint=str(out/'selected.pt'),initial=initial,reports=best[2],checks=best[3],elapsed_s=time.monotonic()-started)
        atomic_json(out/'result.json',result);manifest['status']='COMPLETED';atomic_json(out/'manifest.json',manifest)
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e));atomic_json(out/'manifest.json',manifest);raise

if __name__=='__main__':main()
