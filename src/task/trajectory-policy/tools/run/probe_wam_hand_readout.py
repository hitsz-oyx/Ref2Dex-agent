"""Matched robot offline hand-readout Probe after qualified control-aware R."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit'),str(ROOT/'src/task/goal-point-wam/src')]
from audit_wam_ref2_shadow import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engineering-smoke',action='store_true');p.add_argument('--gated',action='store_true');a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents:raise ValueError('fresh owned output required')
    if subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU2 occupied')
    scratch=ROOT/('tmp/wam-hand-gated' if a.gated else 'tmp/wam-hand-readout');scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2',TMPDIR=str(scratch),TRITON_CACHE_DIR=str(scratch/'triton'))
    import torch
    from goal_point_wam.model import GoalPointWAM
    from goal_point_wam.hand_anchored import HandAnchoredWAM
    from goal_point_wam.hand_gated import GatedHandAnchoredWAM
    from goal_point_wam.data import goal_collate
    from trajectory_policy.wam_shadow import robot_goal_sample,WAM_TO_NATIVE
    from trajectory_policy.retargeter_data import features,ACTIVE
    from trajectory_policy.chunk_retargeter import load_chunk
    from trajectory_policy.chunk_objective import control_aware_loss,independent_PD_scale
    from trajectory_policy.wam_joint import POLICY_MODULES,policy_patch_admission,load_policy_patch
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;install_fused_hilbert()
    owner=ROOT/'outputs/trajectory-policy';fit=owner/'wam-v2-control-fit-20261012-r1'
    gt=owner/'wam-v2-control-GT-20261012-r1'
    if (not json.loads((gt/'result.json').read_text())['GT_execution_gate']
            or not json.loads((gt/'root_chunk_audit.json').read_text())['pass']):raise ValueError('qualified GT Chunk-R required')
    fm=json.loads((fit/'manifest.json').read_text());source=Path(fm['source'])
    def arrays(path):
        with np.load(path/'trajectory.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
    data={'nominal':arrays(source)}
    am=json.loads((Path(fm['recovery_aggregation'])/'manifest.json').read_text())
    for sid,entry in enumerate(am['admissions']):data['recovery%d'%sid]=arrays(Path(entry['native']))
    base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    warm=owner/'wam-ref2-joint-multitakeover-20261012-r1'
    wm,wr=[json.loads((warm/k).read_text()) for k in ('manifest.json','result.json')]
    stats=json.loads((ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json').read_text())
    with np.load(owner/'competent-reference-20261011-r1/reference.npz',allow_pickle=False) as f:reference=f['obj'].copy()
    with np.load(ROOT/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz',allow_pickle=False) as f:
        canonical={k:f[k].copy() for k in f.files}
    rows=[('nominal',t,e) for t in range(519) for e in range(16)]
    startup=np.arange(256);later=np.arange(256,len(rows));recovery=[]
    for sid,entry in enumerate(am['admissions']):
        ids=np.arange(len(rows),len(rows)+105-entry['takeover']);recovery.append(ids)
        rows.extend(('recovery%d'%sid,t,3) for t in range(entry['takeover'],105))
    groups={}
    panel=[0,1,2,4,8,16,32,64,96,128,192,256,320,384,448,518]
    for name,part in dict(nominal=[('nominal',t,e) for t in panel for e in range(24,28)],
                          startup=[('nominal',t,e) for t in range(16) for e in range(24,28)]).items():
        groups[name]=np.arange(len(rows),len(rows)+len(part));rows.extend(part)
    for sid,ids in enumerate(recovery):groups['recovery%d_TRAIN'%sid]=ids
    cache={}
    def get(ids):
        samples=[];labels=[]
        for index in ids:
            index=int(index)
            if index not in cache:
                kind,tick,env=rows[index];d=data[kind];past=np.maximum(tick+np.arange(-3,1),0)
                sample=robot_goal_sample(d['obj'][past,env],d['hand'][past,env],reference[min(tick+24,542)],canonical,sample_id=index)
                state,tau=features(*(d[k][tick,env] for k in ('q','dq','hand','obj','velocity')),d['hand'][tick+np.arange(1,25),env])
                action=d['applied'][tick+np.arange(8),env][:,list(ACTIVE)]
                if kind!='nominal' and (not d['expert_applied'][tick:tick+8,1].all() or d['expert_clipped'][tick:tick+8,1].any()):
                    raise ValueError('unapplied or clipped expert future')
                cache[index]=(sample,dict(state=state,tau=tau,actual_action=action))
            sample,label=cache[index];samples.append(sample);labels.append(label)
        b={k:v.cuda() for k,v in goal_collate(samples).items()}
        b.update({k:torch.as_tensor(np.stack([v[k] for v in labels]),device='cuda') for k in labels[0]})
        return b
    out.mkdir();updates=1 if a.engineering_smoke else 200
    manifest=dict(schema='ref2dex.hand-readout-comparison.v1',status='RUNNING',task='goal-point-wam',
        experiment_id='P-20261012-goal-point-wam-ref26-gated-hand' if a.gated else 'P-20261012-goal-point-wam-ref25-hand-readout',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),physical_gpu=2,
        warm_policy=str(warm),base_checkpoint=str(base),qualified_chunk=str(fit),source=str(source),
        recovery_sources=[entry['native'] for entry in am['admissions']],maximum_updates_per_arm=updates,
        model_seed=295,draw_seed=296,batch=64,nominal_sampling='16 startup+16 later',
        recovery_sampling='32 actual expert futures:11/11/10 per source',train_modules=list(POLICY_MODULES),
        frozen_modules=['R','action','world','normalization buffers'],learning_rates=dict(scene=1e-5,other_policy=1e-4),
        loss='true tau MSE/.05 plus frozen Chunk-R control-aware objective',
        hand_anchoring='full-packed semantic2x11 PTv3 features + unchanged raw-hand residual',
        goal='unchanged32-point endpoint/memory',world='unchanged teacher-forced interface; no world loss in this B Probe',
        budget_seconds=1200,artifact_limit_bytes=500_000_000,training_allowed=False)
    if a.gated:
        manifest.update(readout_mode='zero-init gated LayerNorm/MLP residual',
            train_modules=['trajectory','hand_residual (anchored only)'],
            frozen_modules=['scene','goal','hand_query','R','action','world','normalization buffers'],
            learning_rates=dict(trajectory=1e-4,hand_residual=1e-4),
            hand_anchoring='raw + alpha * P(LN(full-packed semantic2x11)); alpha initially0',
            initial_behavior_required='bitwise identical tau, first native action and shifted-goal predictions',
            useful_signal='all non-regression checks plus >=5% reduction in at least one h1/first8/firstPD panel',
            attribution='post-fit alpha=0 ablation with identical trained decoder; no additional fitting',
            C='paused throughout this revision')
    write(out/'manifest.json',manifest);started=time.monotonic();all_raw={};results={}
    order=torch.as_tensor(WAM_TO_NATIVE,device='cuda');controller=json.loads((source/'manifest.json').read_text())['native_controller']
    pdscale=independent_PD_scale(torch.tensor(np.asarray(controller['scale'])[list(ACTIVE)],dtype=torch.float32,device='cuda'))
    def guard(arm,step,train_started):
        if time.monotonic()-started>1200:raise TimeoutError('paired hand readout1200s budget')
        pids=set(map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
        if pids-{os.getpid()}:raise RuntimeError('foreign GPU2 process')
        size=sum(int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0]) for path in (out,scratch))
        if size>=500_000_000:raise RuntimeError('hand-readout artifact bound')
        gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
        row=dict(arm=arm,step=step,total=updates,elapsed_s=time.monotonic()-started,
                 eta_arm_s=(time.monotonic()-train_started)/max(step,1)*(updates-step),gpu=gpu)
        print(json.dumps(row),flush=True)
        with (out/'monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    for arm,cls in (('old',GoalPointWAM),('anchored',GatedHandAnchoredWAM if a.gated else HandAnchoredWAM)):
        torch.manual_seed(295);rng=np.random.default_rng(296)
        packet=torch.load(base,map_location='cpu',weights_only=False)
        model=cls(stats,packet['config']).cuda().eval()
        loaded=model.load_state_dict(packet['model'],strict=not (a.gated and arm=='anchored'))
        if a.gated and arm=='anchored':
            expected={'hand_residual.'+k for k in model.hand_residual.state_dict()}
            if set(loaded.missing_keys)!=expected or loaded.unexpected_keys:raise ValueError('unexpected historical checkpoint mismatch')
        head=torch.load(warm/'head.pt',map_location='cpu',weights_only=False)
        load_policy_patch(model,head,policy_patch_admission(head,wm,wr));del packet,head
        r=load_chunk(torch.load(fit/'chunk.pt',map_location='cpu',weights_only=False),'cuda');r.requires_grad_(False)
        for name in ('action','world'):getattr(model,name).requires_grad_(False)
        train_names=(('trajectory',)+(('hand_residual',) if arm=='anchored' else ())) if a.gated else POLICY_MODULES
        if a.gated:
            model.requires_grad_(False)
            for name in train_names:getattr(model,name).requires_grad_(True)
        modules={name:getattr(model,name) for name in train_names}
        frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
        buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        r_before={k:v.detach().cpu().clone() for k,v in r.state_dict().items()}
        optimizer=torch.optim.AdamW([dict(params=list(net.parameters()),lr=1e-5 if name=='scene' else 1e-4)
                                    for name,net in modules.items()],weight_decay=.01)
        params=[p for net in modules.values() for p in net.parameters()]
        def predict(b,scene=None):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if scene is None:scene=model.scene(b,return_points=True)
                if arm=='anchored':trajectory=model.actor_from_scene(scene,b)
                else:
                    goal,mask=model.goal(b['start_object_points'],b['goal_object_points'],b['object_goal_valid'])
                    trajectory=model.trajectory(scene['scene_tokens'],scene['scene_mask'],goal,mask,model.observed_hand_condition(b))
            return trajectory[:,:,0].index_select(2,order).reshape(-1,24,33),scene
        def evaluate(phase):
            report={}
            with torch.no_grad():
                for name,ids in groups.items():
                    collected={k:[] for k in ('tau','tau_true','action0','action0_true','raw_clipped')}
                    for begin in range(0,len(ids),16):
                        b=get(ids[begin:begin+16]);tau,_=predict(b);action=r(b['state'],tau)[:,0]
                        values=dict(tau=tau,tau_true=b['tau'],action0=action.clamp(-1,1),action0_true=b['actual_action'][:,0],
                                    raw_clipped=(action.abs()>1+1e-6).any(-1))
                        for k,v in values.items():collected[k].append(v.cpu().numpy())
                    values={k:np.concatenate(v) for k,v in collected.items()};values['row_ids']=ids
                    e=np.linalg.norm((values['tau']-values['tau_true']).reshape(-1,24,11,3),axis=-1)*1000
                    pd=(values['action0']-values['action0_true'])*pdscale.cpu().numpy().astype(np.float64)
                    report[name]=dict(h1_mm=float(e[:,0].mean()),first8_mm=float(e[:,:8].mean()),all24_mm=float(e.mean()),
                                      first_PD_xyz_p95_mm=float(np.percentile(np.linalg.norm(pd[:,:3],axis=-1),95)*1000),
                                      clipping_rate=float(values['raw_clipped'].mean()))
                    all_raw.update({arm+'_'+phase+'_'+name+'_'+k:v for k,v in values.items()})
                b=get(groups['nominal'][:16]);_,scene=predict(b);shifted=[]
                for sign in (-1,1):
                    changed=dict(b);changed['goal_object_points']=b['goal_object_points'].clone()
                    changed['goal_object_points'][...,0]+=sign*.02
                    tau,_=predict(changed,scene);shifted.append(tau.cpu().numpy())
                response=np.linalg.norm((shifted[1]-shifted[0]).reshape(-1,24,11,3),axis=-1)*1000
                report['goal_sensitivity_mm']=float(response.mean())
                all_raw[arm+'_'+phase+'_goal_minus_tau'],all_raw[arm+'_'+phase+'_goal_plus_tau']=shifted
            return report
        initial=evaluate('initial');draws=[];curve=[];gradient={}
        if a.gated and arm=='anchored':
            comparisons={k:float(np.max(np.abs(v.astype(np.float64)-all_raw[k.replace('anchored_initial_', 'old_initial_',1)].astype(np.float64))))
                         for k,v in all_raw.items() if k.startswith('anchored_initial_')}
            if any(comparisons.values()):raise ValueError('zero gate did not preserve actual initial behavior')
            gradient['initial_behavior_max_difference']=max(comparisons.values())
            gradient['initial_alpha']=float(model.hand_residual.alpha)
        train_started=time.monotonic();guard(arm,0,train_started)
        for step in range(1,updates+1):
            ids=np.concatenate([rng.choice(startup,16),rng.choice(later,16),
                                *[rng.choice(pool,count) for pool,count in zip(recovery,(11,11,10))]])
            draws.append(ids);b=get(ids);optimizer.zero_grad(set_to_none=True)
            tau,scene=predict(b);geometry=((tau-b['tau'])/.05).square().mean()
            action,_=control_aware_loss(r(b['state'],tau),b['actual_action'],r.action_scale,pdscale)
            if step==1:
                tau_grad=torch.autograd.grad(action,tau,retain_graph=True)[0]
                gradient['action_to_tau_sum']=float(tau_grad.abs().sum())
                if gradient['action_to_tau_sum']<=0:raise ValueError('frozen R blocks actor gradient')
            loss=geometry+action
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite readout loss')
            loss.backward()
            if any(p.grad is not None for p in list(r.parameters())+list(model.parameters()) if not p.requires_grad):raise ValueError('frozen module gradient')
            if a.gated and arm=='anchored' and step<=2:
                gate_grad=float(model.hand_residual.alpha.grad.abs())
                projection_grad=sum(float(p.grad.abs().sum()) for p in model.hand_residual.projection.parameters())
                gradient['gate_grad_step%d'%step]=gate_grad
                gradient['projection_grad_step%d'%step]=projection_grad
                if step==1 and (gate_grad<=0 or projection_grad!=0):raise ValueError('zero gate gradient wiring differs')
                if step==2 and projection_grad<=0:raise ValueError('gate failed to wake projection')
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite readout gradient')
            optimizer.step()
            if step==1 or step%25==0:
                curve.append(dict(step=step,tau_MSE=float(geometry),control=float(action),
                                  alpha=float(model.hand_residual.alpha) if a.gated and arm=='anchored' else None));guard(arm,step,train_started)
        final=evaluate('final');ablation=None
        if a.gated and arm=='anchored':
            gradient['final_alpha']=float(model.hand_residual.alpha)
            saved_alpha=model.hand_residual.alpha.detach().clone()
            with torch.no_grad():model.hand_residual.alpha.zero_()
            ablation=evaluate('gate_zero')
            with torch.no_grad():model.hand_residual.alpha.copy_(saved_alpha)
        frozen_error=max(float((model.state_dict()[k].cpu()-v).abs().max()) for k,v in frozen.items())
        buffer_error=max(float((v.cpu().float()-buffers[k].float()).abs().max()) for k,v in model.named_buffers())
        r_error=max(float((r.state_dict()[k].cpu()-v).abs().max()) for k,v in r_before.items())
        if max(frozen_error,buffer_error,r_error)>0:raise ValueError('frozen model or statistics changed')
        results[arm]=dict(initial=initial,final=final,curve=curve,gradient=gradient,
                          frozen_world_action_max_change=frozen_error,buffer_max_change=buffer_error,R_max_change=r_error,
                          frozen_parameter_max_change=frozen_error,gate_zero_ablation=ablation)
        all_raw[arm+'_draws']=np.asarray(draws)
        if not a.engineering_smoke:
            save_names=POLICY_MODULES+(('hand_residual',) if a.gated and arm=='anchored' else ())
            torch.save(dict(schema='ref2dex.hand-readout-robot-policy.v1',hand_anchored=arm=='anchored',
                            readout_mode='gated' if a.gated else 'direct',
                            modules={name:{k:v.detach().cpu() for k,v in getattr(model,name).state_dict().items()} for name in save_names},
                            base_checkpoint=str(base),qualified_R=str(fit/'chunk.pt')),out/(arm+'.pt'))
        del model,r,optimizer,modules,params;gc.collect();torch.cuda.empty_cache()
    if not np.array_equal(all_raw['old_draws'],all_raw['anchored_draws']):raise ValueError('unmatched training rows')
    checks={}
    for group in groups:
        old,new=[results[arm]['final'][group] for arm in ('old','anchored')]
        for metric in ('h1_mm','first8_mm'):
            checks[group+'_'+metric]=new[metric]<=1.1*old[metric]+.05
        checks[group+'_control']=new['first_PD_xyz_p95_mm']<=1.1*old['first_PD_xyz_p95_mm']+.1
        checks[group+'_clipping']=new['clipping_rate']<=old['clipping_rate']+.01
    checks['goal_sensitivity']=results['anchored']['final']['goal_sensitivity_mm']>=max(.001,.5*results['old']['final']['goal_sensitivity_mm'])
    passed=all(checks.values()) and not a.engineering_smoke
    np.savez_compressed(out/'predictions.npz',**all_raw,rows_kind=np.asarray([k for k,t,e in rows]),
                        rows_tick=np.asarray([t for k,t,e in rows]),rows_role=np.asarray([e for k,t,e in rows]))
    useful_panels={}
    if a.gated:
        for group in groups:
            old,new=[results[arm]['final'][group] for arm in ('old','anchored')]
            for key in ('h1_mm','first8_mm','first_PD_xyz_p95_mm'):
                useful_panels[group+'_'+key]=new[key]<=.95*old[key]
    useful=passed and (any(useful_panels.values()) if a.gated else True)
    result=dict(status='COMPLETED',arms=results,checks=checks,B_offline_gate=passed,
                useful_panels=useful_panels,B_useful_signal=useful,C_started=False,
                conclusion='engineering PASS' if a.engineering_smoke else 'PROMISING' if useful else 'UNCLEAR' if passed else 'UNPROMISING',
                elapsed_s=time.monotonic()-started,scope='fixed robot offline matched short fit; no autonomous physics or core hypothesis verdict')
    write(out/'result.json',result);manifest['status']='COMPLETED';write(out/'manifest.json',manifest)
    print(json.dumps(dict(B_offline_gate=passed,checks=checks,elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':main()
