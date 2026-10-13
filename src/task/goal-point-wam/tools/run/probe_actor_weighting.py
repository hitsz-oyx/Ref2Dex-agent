"""Independent matched short A/B: original sampler versus OakInk actor downweight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from train_three_domain import GoalPointWAM,GoalWindows,loader,move_batch,atomic_json,gpu_snapshot,parameter_hash
from oakink_wm.multisource import MixedWindows,mixed_indices
from oakink_wm.model import metrics
from goal_point_wam.actor_weighting import target_mismatch,weighted_trajectory_loss
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=300)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/goal-point-wam' not in out.parents:raise ValueError('fresh owned output required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('assigned GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    data=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree'
    stats=data.parent/'norm_stats.json'
    parent=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    census=ROOT/'outputs/goal-point-wam/main3-curation-audit-20261013-r1'
    train_base=MixedWindows(data,'train');val_base=MixedWindows(data,'val')
    train=GoalWindows(train_base,'object_points',32);val=GoalWindows(val_base,'object_points',32)
    with np.load(census/'oakink2.npz',allow_pickle=False) as f:
        if not np.array_equal(f['indices'],np.arange(len(train_base.sources[0]))):raise ValueError('full ordered census required')
        if not np.array_equal(f['rows'],train_base.sources[0].rows):raise ValueError('census row identity differs')
        mismatch=target_mismatch(f['metrics'])
        census_rows=f['metrics'].shape[0]
    with np.load(census/'model-hand-r1/val.npz',allow_pickle=False) as f:
        # Fixed first128 rows per source from the existing untouched VAL panel.
        vi=np.concatenate([np.flatnonzero(f['source']==sid)[:128] for sid in range(3)])
        val_indices=f['index'][vi];val_geometry=f['metrics'][vi]
    packet=torch.load(parent,map_location='cpu',weights_only=False);config=packet['config']
    if packet['checkpoint_kind']!='goal-point-wam.single.v1' or packet.get('artificial_labels'):raise ValueError('real base required')
    indices=mixed_indices(train_base,a.updates*a.batch,301)
    flags=np.zeros(len(indices),bool);oak=indices<train_base.offsets[1];flags[oak]=mismatch[indices[oak]]
    out.mkdir();np.save(out/'train_indices.npy',indices);np.save(out/'validation_indices.npy',val_indices)
    np.savez(out/'weight_contract.npz',train_mismatch=flags,val_geometry=val_geometry)
    started=time.monotonic()
    manifest=dict(status='RUNNING',experiment_id='P-20261013-goal-point-wam-ref31-actor-weight',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parent=str(parent),data=str(data),census=str(census),census_rows=census_rows,seed=300,draw_seed=301,updates=a.updates,batch=a.batch,physical_gpu=a.gpu,train_mismatch_count=int(flags.sum()),train_count=len(indices),learning_rate=1e-5,weight_decay=.01,downweight=.1,sampler='unchanged mixed_indices source/category distribution; identical sequence both arms',world='unchanged teacher-forced world loss and labels; all original trainable modules updated',scope='full-model short warm-start; separate from robot C/R; no deletion or GRAB/ARCTIC coverage change',training_allowed=False,deterministic_algorithms=True,cublas_workspace=':4096:8')
    atomic_json(out/'manifest.json',manifest)
    groups={};sid=np.searchsorted(val_base.offsets,val_indices,side='right')-1
    vm=target_mismatch(val_geometry)&(sid==0)
    for i,name in enumerate(('oakink2','grab','arctic')):
        groups[name]=sid==i
    groups['oakink2_mismatch']=vm;groups['oakink2_aligned_proxy']=(sid==0)&~vm
    groups['oakink2_target_interaction']=(sid==0)&(np.minimum(val_geometry[:,4],val_geometry[:,5])<=50)
    def evaluate(model,label):
        model.eval();arr={k:[] for k in ('actor_mm','first8_mm','world_mm')}
        with torch.no_grad():
            for cpu in loader(val,val_indices,a.batch,2):
                batch=move_batch(cpu,torch.device('cuda'))
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(batch)
                error=torch.linalg.vector_norm(pred['trajectory'].float()-batch['trajectory_target'],dim=-1);valid=batch['trajectory_valid']
                arr['actor_mm'].append(((error*valid).sum((1,2,3))/valid.sum((1,2,3))*1000).cpu().numpy())
                arr['first8_mm'].append(((error[:,:8]*valid[:,:8]).sum((1,2,3))/valid[:,:8].sum((1,2,3))*1000).cpu().numpy())
                e=metrics(pred,batch)['point_epe'][:,:,-1];v=batch['object_valid']
                arr['world_mm'].append(((e*v).sum(-1)/v.sum(-1)*1000).cpu().numpy())
        arr={k:np.concatenate(v) for k,v in arr.items()};np.savez(out/(label+'_val.npz'),**arr)
        return {name:dict(n=int(mask.sum()),**{k:float(v[mask].mean()) if mask.any() else None for k,v in arr.items()}) for name,mask in groups.items()}
    results={};initial_hash=None;first_contract=None;first_prediction=None
    try:
        for arm in ('original','downweight'):
            random.seed(300);np.random.seed(300);torch.manual_seed(300);torch.cuda.manual_seed_all(300)
            model=GoalPointWAM(json.loads(stats.read_text()),config).cuda();model.load_state_dict(packet['model'],strict=True)
            init=parameter_hash(model)
            if initial_hash is None:initial_hash=init
            elif init!=initial_hash:raise ValueError('initial weights differ')
            if arm=='original':results['initial']=evaluate(model,'initial')
            # Evaluation must not shift the matched stochastic training stream.
            random.seed(300);np.random.seed(300);torch.manual_seed(300);torch.cuda.manual_seed_all(300)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
            model.train();arm_start=time.monotonic();curve=[]
            for step,cpu in enumerate(loader(train,indices,a.batch,2),1):
                batch=move_batch(cpu,torch.device('cuda'))
                marked=flags[(step-1)*a.batch:step*a.batch]
                weight=torch.tensor(np.where(marked,.1,1.) if arm=='downweight' else np.ones(len(marked)),dtype=torch.float32,device='cuda')
                optimizer.zero_grad(set_to_none=True)
                rng_digest=hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest() if step==1 else None
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(batch)
                actor=weighted_trajectory_loss(pred['trajectory'],batch,model.trajectory_scale_m,weight)
                world,_=model.world_loss(pred,batch);loss=actor+model.world_weight*world
                if step==1:
                    baseline=model.trajectory_loss(pred['trajectory'],batch)
                    ones=weighted_trajectory_loss(pred['trajectory'],batch,model.trajectory_scale_m,torch.ones_like(weight))
                    if (ones-baseline).abs()>1e-7:raise ValueError('all-one weights changed original loss')
                    batch_digest=hashlib.sha256(b''.join(cpu[k].numpy().tobytes() for k in sorted(cpu))).hexdigest()
                    contract=dict(world=float(world),unweighted_actor=float(baseline),weighted_actor=float(actor),batch_digest=batch_digest,rng_digest=rng_digest)
                    atomic_json(out/(arm+'_contract.json'),contract)
                    if first_contract is None:
                        first_contract=contract
                        first_prediction={k:v.detach().cpu().clone() for k,v in pred.items()}
                    else:
                        contract['prediction_max_difference']={k:float((v.detach().cpu()-first_prediction[k]).abs().max()) for k,v in pred.items()}
                        contract['valid_prediction_max_difference']={k:float((v.detach().cpu()-first_prediction[k])[batch['trajectory_valid'].cpu() if k=='trajectory' else batch['object_valid'].cpu()].abs().max()) for k,v in pred.items()}
                        contract['world_relative_difference']=abs(contract['world']-first_contract['world'])/max(abs(first_contract['world']),1e-8)
                        atomic_json(out/(arm+'_contract.json'),contract)
                        if batch_digest!=first_contract['batch_digest'] or rng_digest!=first_contract['rng_digest']:
                            raise ValueError('paired input/random state differs')
                        if abs(contract['world']-first_contract['world'])>1e-7 or max(contract['prediction_max_difference'].values())>1e-6:
                            raise ValueError('paired first-batch numeric tolerance exceeded: '+str(contract))
                    atomic_json(out/(arm+'_contract.json'),contract)
                if not torch.isfinite(loss):raise FloatingPointError('loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('gradient')
                optimizer.step()
                if step==1 or step%50==0 or step==a.updates:
                    elapsed=time.monotonic()-arm_start
                    row=dict(arm=arm,step=step,loss=float(loss),actor=float(actor),world=float(world),elapsed_s=elapsed,eta_s=elapsed/step*(a.updates-step),gpu=gpu_snapshot(torch.device('cuda')))
                    curve.append(row);atomic_json(out/(arm+'_curve.json'),dict(curve=curve));print(json.dumps(row),flush=True)
                    pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                    if pids-{os.getpid()}:raise RuntimeError('foreign GPU process')
                if time.monotonic()-started>2400:raise TimeoutError('paired2400s budget')
            results[arm]=evaluate(model,arm)
            torch.save(dict(checkpoint_kind='goal-point-wam.single.v1',model=model.state_dict(),config=config,step=a.updates,identity=manifest,parent_step=packet['step']),out/(arm+'.pt'))
            del optimizer,model;torch.cuda.empty_cache()
        comparisons={}
        for name in groups:
            if results['original'][name]['n']:
                comparisons[name]={k:results['downweight'][name][k]-results['original'][name][k] for k in ('actor_mm','first8_mm','world_mm')}
        # Fixed operational signal; still only one-seed warm-start Probe.
        base=results['original'];new=results['downweight']
        improve=new['oakink2_target_interaction']['actor_mm']<=.95*base['oakink2_target_interaction']['actor_mm']
        preserve=all(new[k]['actor_mm']<=1.05*base[k]['actor_mm'] and new[k]['world_mm']<=1.05*base[k]['world_mm'] for k in ('grab','arctic','oakink2'))
        result=dict(status='COMPLETED',reports=results,delta_downweight_minus_original_mm=comparisons,initial_parameter_sha256=initial_hash,promising_gate=bool(improve and preserve),conclusion='PROMISING' if improve and preserve else 'UNCLEAR',elapsed_s=time.monotonic()-started,scope=manifest['scope'])
        atomic_json(out/'result.json',result);manifest['status']='COMPLETED';atomic_json(out/'manifest.json',manifest)
    except BaseException as e:
        manifest.update(status='FAILED',error=repr(e));atomic_json(out/'manifest.json',manifest);raise

if __name__=='__main__':main()
