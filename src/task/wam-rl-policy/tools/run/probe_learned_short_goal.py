"""Fit observed-state rigid endpoints, then qualify their frozen-WAM forecasts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

TASK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK.parent/'shadow-finetune/src')]
from shadow_finetune import paths
from video_data_preparation.unidexgrasp_data import object_poses
from wam_rl_policy.short_goal import ShortGoal,normalized_inputs,point_displacement
from wam_rl_policy.task_readout import object_split,observed_batch,forecast_from_encoded
from wam_rl_policy.fast_bc import batched_inputs
from wam_rl_policy.wam_features import FeatureController


def write(path,value):
    part=path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');part.replace(path)


def prepare(packet,geometries):
    """Observation fields and labels remain separate in every model call."""
    source=Path(packet['manifest']['source']);result={}
    append=lambda key,value:result.setdefault(key,[]).append(torch.as_tensor(value))
    queries=np.stack([g['surface'][np.linspace(0,len(g['surface'])-1,32,dtype=int)] for g in geometries])
    for ep,name in enumerate(['first-episode.npz','last-episode.npz']):
        with np.load(source/name) as d:
            obj=d['object'];hand=d['hand'];goal=d['goal'];reset=d['reset'];clock=d['time']
        np.testing.assert_allclose(np.diff(clock),1/60,atol=1e-6)
        poses=object_poses(obj)
        for tick in packet['tick'].unique().tolist():
            if tick+48>=len(obj):continue
            assert not reset[tick+1:tick+49].any()
            rows=((packet['episode']==ep)&(packet['tick']==tick)).nonzero().flatten()
            assert packet['object_id'][rows].tolist()==list(range(len(geometries)))
            now=poses[tick];r=now[:,:3,:3];inv=np.linalg.inv(now);future=inv@poses[tick+48]
            target=np.einsum('bij,bpj->bpi',future[:,:3,:3],queries)+future[:,None,:3,3]-queries
            relative_goal=np.einsum('bi,bij->bj',goal[tick]-obj[tick,:,:3],r)
            current_hand=np.einsum('bki,bij->bkj',hand[tick]-obj[tick,:,None,:3],r)
            progress=np.linalg.norm(goal[tick]-obj[tick,:,:3],axis=-1)-np.linalg.norm(goal[tick]-obj[tick+48,:,:3],axis=-1)
            hs=[];ss=[]
            for horizon in [16,32]:
                rel=inv@poses[tick+horizon]
                hs.append(np.einsum('bki,bij->bkj',hand[tick+horizon]-hand[tick],r))
                ss.append(np.einsum('bij,bpj->bpi',rel[:,:3,:3],queries)+rel[:,None,:3,3]-queries)
            labels=np.concatenate((np.stack(hs,1).reshape(len(obj[0]),-1),np.stack(ss,1).reshape(len(obj[0]),-1)),1)
            past=tick+np.array([-6,-4,-2,0])
            for key,value in dict(rows=rows,query=queries,target=target,target_rotation=future[:,:3,:3],
                target_translation=future[:,:3,3],relative_goal=relative_goal,current_hand=current_hand,
                progress=progress>.02,future_labels=labels,
                observed_objects=obj[past].swapaxes(0,1),observed_hands=hand[past].swapaxes(0,1),
                known_goal=goal[tick]).items():append(key,value)
    data={k:torch.cat(v) for k,v in result.items()}
    rows=data['rows']
    for key in ['fit','object_id','tick','episode']:data[key]=packet[key][rows]
    data={k:v.float() if v.is_floating_point() else v for k,v in data.items()}
    assert len(rows)==1792 and int(data['fit'].sum())==1344
    expected,_=object_split(packet['objects'])
    np.testing.assert_array_equal(data['fit'],expected[data['object_id']])
    return data


def aggregate(error,chosen,object_index):
    error=error.detach().cpu();chosen=chosen.cpu();object_index=object_index.cpu()
    if not chosen.any():return dict(n=0,mean=None,macro=None,per_object={})
    groups={str(i):float(error[chosen & (object_index==i)].mean()) for i in object_index[chosen].unique().tolist()}
    return dict(n=int(chosen.sum()),mean=float(error[chosen].mean()),macro=float(np.mean(list(groups.values()))),per_object=groups)


def endpoint_metrics(delta,translation,data,object_index):
    errors=dict(point_mm=(delta.cpu()-data['target']).norm(dim=-1).mean(-1)*1000,
        translation_mm=(translation.cpu()-data['target_translation']).norm(dim=-1)*1000)
    result={}
    for split,mask in [('fit',data['fit']),('heldout',~data['fit'])]:
        result[split]={}
        for subset,chosen in [('all',mask),('progress',mask & data['progress'])]:
            result[split][subset]={k:aggregate(v,chosen,object_index) for k,v in errors.items()}
    return result


def forecast_metrics(pred,data,object_index):
    result={}
    for domain,part,points in [('hand',slice(0,66),11),('scene',slice(66,None),32)]:
        truth=data['future_labels'][:,part].reshape(-1,2,points,3)
        error=(pred[:,part].reshape(-1,2,points,3)-truth).norm(dim=-1).mean(-1)*1000
        for j,horizon in enumerate([16,32]):
            result[f'{domain}{horizon}']={}
            for split,mask in [('fit',data['fit']),('heldout',~data['fit'])]:
                result[f'{domain}{horizon}'][split]={subset:aggregate(error[:,j],chosen,object_index)
                    for subset,chosen in [('all',mask),('progress',mask & data['progress'])]}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    a.output.mkdir(parents=True);started=time.monotonic()
    packet=torch.load(a.parent/'features/features.pt',map_location='cpu',weights_only=False)
    objects=packet['objects'];meta=json.loads((a.cache/'manifest.json').read_text())
    lookup={r['object_code']:r['geometry'] for r in meta['records']};geometries=[]
    for row in objects:
        with np.load(lookup[row['object_code']]) as g:
            n=g['normals'];length=np.linalg.norm(n,axis=-1);valid=length>1e-6
            geometries.append(dict(surface=g['surface'][valid],normals=n[valid]/length[valid,None]))
    data=prepare(packet,geometries)
    torch.save(data,a.output/'dataset.pt')
    codes=sorted({r['object_code'] for r in objects})
    object_index=torch.tensor([codes.index(objects[i]['object_code']) for i in data['object_id'].tolist()])
    manifest=dict(status='RUNNING',task='wam-rl-policy',experiment_id='P-20261016-wam-learned-short-goal',
        run_id=a.output.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        pid=os.getpid(),parent=str(a.parent.resolve()),cache=str(a.cache.resolve()),gpu=a.gpu,
        samples=len(data['rows']),fit_samples=int(data['fit'].sum()),steps=1000,batch=128,
        init_seed=246,batch_seed=247,lr=3e-4,loss='point displacement / .05m SmoothL1(beta1)',
        input_contract='observed state/history/task goal only; endpoint48 is label, never input',
        scope='predict recorded mixed behavior, not optimal or certified reachable goals',max_seconds=600)
    write(a.output/'manifest.json',manifest)

    def resource(stage,step,total,begin):
        elapsed=time.monotonic()-begin
        row=dict(stage=stage,step=step,total=total,seconds=elapsed,eta=elapsed/max(1,step)*(total-step),
            resource=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
        print(json.dumps(row),flush=True)
        with (a.output/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if time.monotonic()-started>600:raise TimeoutError('Short-goal probe budget exceeded')

    rows=data['rows'];fit=data['fit'].cuda();query=data['query'].cuda();target=data['target'].cuda()
    fit_ids=fit.nonzero().flatten();state=packet['state'][rows].cuda()
    baselines={};identity=torch.eye(3).expand(len(rows),3,3)
    for name,factor in [('zero',torch.zeros(len(rows))),('scheduled',(48/(199-data['tick']).float()).clamp_max(1)),('task',torch.ones(len(rows)))]:
        translation=data['relative_goal']*factor[:,None]
        delta=point_displacement(data['query'],identity,translation)
        baselines[name]=endpoint_metrics(delta,translation,data,object_index)
    predictions={};results={}
    for arm in ['raw-query','wam-mean']:
        use_mean=arm=='wam-mean';torch.manual_seed(246)
        tokens=packet['mean' if use_mean else 'raw'][rows].cuda()
        valid=packet['raw_valid'][rows].cuda()
        x,z,stats=normalized_inputs(state,tokens,fit,mean=use_mean,valid=None if use_mean else valid)
        model=ShortGoal(x.shape[-1],z.shape[-1],mean=use_mean).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        generator=torch.Generator(device='cuda').manual_seed(247);begin=time.monotonic();history=[]
        for step in range(1,1001):
            ix=fit_ids[torch.randint(len(fit_ids),(128,),device='cuda',generator=generator)]
            out=model(x[ix],z[ix],valid[ix],query[ix])
            loss=F.smooth_l1_loss(out['delta']/.05,target[ix]/.05,beta=1.)
            if not torch.isfinite(loss):raise FloatingPointError(arm)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step==1 or step%250==0:
                with torch.no_grad():
                    outputs=[model(x[ids],z[ids],valid[ids],query[ids]) for ids in torch.arange(len(rows),device='cuda').split(128)]
                    pred={key:torch.cat([o[key] for o in outputs]).cpu() for key in outputs[0]}
                metrics=endpoint_metrics(pred['delta'],pred['translation'],data,object_index)
                history.append(dict(step=step,loss=float(loss),metrics=metrics))
                print(json.dumps(dict(arm=arm,step=step,fit_mm=metrics['fit']['all']['point_mm']['macro'],
                    held_mm=metrics['heldout']['all']['point_mm']['macro'])),flush=True)
                resource(arm,step,1000,begin)
        torch.testing.assert_close(pred['rotation'].transpose(-1,-2)@pred['rotation'],identity,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(torch.linalg.det(pred['rotation']),torch.ones(len(rows)),atol=1e-4,rtol=1e-4)
        predictions[arm]=pred
        results[arm]=dict(endpoint=metrics,history=history,parameters=sum(v.numel() for v in model.parameters()),seconds=time.monotonic()-begin)
        torch.save(dict(model=model.cpu().state_dict(),normalization={k:v.cpu() for k,v in stats.items()},
                        arm=arm,prediction=pred,manifest=manifest),a.output/(arm+'.pt'))
        del model,optimizer,x,z,tokens,valid,outputs,out
        write(a.output/'fit-results.json',dict(models=results,baselines=baselines))
    del state,query,target,fit,fit_ids
    torch.cuda.empty_cache()
    controller=FeatureController(packet['manifest']['checkpoint'],objects,a.cache)
    wam=controller.model.forecaster;point_budget=max(len(g['surface']) for g in geometries)
    forecasts={arm:[] for arm in predictions};begin=time.monotonic();total=len(rows)//16
    with torch.inference_mode():
        for start in range(0,len(rows),16):
            sl=slice(start,start+16);envs=data['object_id'][sl].tolist()
            samples=batched_inputs(data['observed_objects'][sl].numpy(),data['observed_hands'][sl].numpy(),
                                  data['known_goal'][sl].numpy(),[geometries[i] for i in envs])
            batch=observed_batch(samples,point_budget)
            torch.testing.assert_close(batch['start_object_points'].cpu(),data['query'][sl])
            for arm,pred in predictions.items():
                # Only network output conditions WAM here. Future endpoint labels
                # are never used to construct an inference batch.
                batch['goal_object_delta']=pred['delta'][sl].cuda()
                shared,valid,geometry=wam.encode(batch)
                hand,scene=forecast_from_encoded(wam,batch,shared,valid,geometry['tokens'].shape[1])
                forecasts[arm].append(torch.cat((hand[:,[7,15],0].flatten(1),scene[:,:,[7,15]].flatten(1)),-1).cpu())
            step=start//16+1
            if step==1 or step%28==0:resource('frozen-wam',step,total,begin)
    forecasts={k:torch.cat(v) for k,v in forecasts.items()}
    with np.load(a.parent/'goal-contract/diagnostic-only-predictions.npz') as previous:
        np.testing.assert_array_equal(previous['fit'],data['fit'])
        np.testing.assert_allclose(previous['target'],data['future_labels'],rtol=1e-5,atol=1e-7)
        # Existing predictions are read only now, for scoring; never for fitting.
        comparison={name:forecast_metrics(torch.from_numpy(previous[name]).float(),data,object_index)
                    for name in ['task','scheduled','zero','oracle']}
    for arm,pred in predictions.items():
        hand_delta=point_displacement(data['current_hand'],pred['rotation'],pred['translation'])
        factors=torch.tensor([16/48,32/48])[None,:,None,None]
        interpolation=torch.cat(((hand_delta[:,None]*factors).flatten(1),(pred['delta'][:,None]*factors).flatten(1)),-1)
        results[arm]['forecast']=forecast_metrics(forecasts[arm],data,object_index)
        results[arm]['interpolation']=forecast_metrics(interpolation,data,object_index)
    gates={}
    for arm,result in results.items():
        endpoint=result['endpoint']['heldout']['all']['point_mm']['macro']
        endpoint_base=min(baselines[n]['heldout']['all']['point_mm']['macro'] for n in ['zero','scheduled'])
        ratios={key:result['forecast'][key]['heldout']['all']['macro']/comparison['zero'][key]['heldout']['all']['macro']
                for key in ['hand16','hand32','scene16','scene32']}
        progress={key:result['forecast'][key]['heldout']['progress']['macro']/comparison['zero'][key]['heldout']['progress']['macro']
                  for key in ['hand32','scene32']}
        progress['endpoint']=result['endpoint']['heldout']['progress']['point_mm']['macro']/baselines['zero']['heldout']['progress']['point_mm']['macro']
        passed=endpoint<=.9*endpoint_base and ratios['hand32']<=.9 and ratios['scene32']<=.9 and max(ratios['hand16'],ratios['scene16'])<=1.05 and max(progress.values())<=1.05
        gates[arm]=dict(endpoint_ratio=endpoint/endpoint_base,forecast_ratios=ratios,progress_ratios=progress,passed=passed)
    conclusion='PROMISING' if any(g['passed'] for g in gates.values()) else 'UNPROMISING'
    summary=dict(status='COMPLETED',conclusion=conclusion,scope=manifest['scope'],models=results,
                 endpoint_baselines=baselines,forecast_baselines=comparison,gates=gates)
    write(a.output/'summary.json',summary)
    torch.save(forecasts,a.output/'predicted-goal-forecasts.pt')
    manifest.update(status='COMPLETED',seconds=time.monotonic()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),conclusion=conclusion)
    write(a.output/'manifest.json',manifest)
    print(json.dumps(dict(conclusion=conclusion,gates=gates,seconds=manifest['seconds']),indent=2),flush=True)


if __name__=='__main__':main()
