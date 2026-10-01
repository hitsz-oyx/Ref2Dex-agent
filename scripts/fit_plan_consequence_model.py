#!/usr/bin/env python3
"""Bounded actual H10 plan-Cm fitting; pre-action inputs, fixed three controls."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission
from analyze_executable_contact_opportunity import split_group


def run(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve();output=args.output.resolve()
    if output.parent!=base or output.is_symlink():raise ValueError('unique owned output required')
    parent=json.loads((args.run/'run_manifest.json').read_text())
    opportunity=json.loads(args.opportunity.read_text())
    if parent['run_status']!='COMPLETED' or parent['smoke_only'] or opportunity['label']!='PROMISING':raise ValueError('positive terminal opportunity required')
    if any(sha(Path(k))!=v for k,v in parent['input_sha256'].items()):raise ValueError('source/input drift')
    admission=None
    for gpu in args.gpus:
        try:admission=gpu_admission(gpu);break
        except RuntimeError:continue
    if admission is None:raise RuntimeError('allowed devices occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.plan_consequence_model import PlanConsequenceModel,StateOptionConsequenceModel,plan_features,physical_targets,consequence_loss
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    inputs=[Path(__file__),ROOT/'src/task/CmResidual/plan_consequence_model.py',args.run/'run_manifest.json',args.opportunity]
    records=[]
    for p in parent['phases']:
        path=Path(p['directory'])/'records.pt'
        if p['run_status']!='COMPLETED' or sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
        b=torch.load(path,map_location='cpu',weights_only=False)
        if b['schema']!='ref2dex.orientation_feedback_options.v1' or b['future_done'].any():raise ValueError('actual complete program contract')
        records.append(b);inputs.append(path)
    hashes={str(p.resolve()):sha(p) for p in inputs};output.mkdir(exist_ok=False);begin=time.monotonic()
    manifest=dict(run_status='RUNNING',experiment_id='P-20261002-plan-consequence-control',family='HF15',probe_index_in_family=2,pid=os.getpid(),gpu=admission,input_sha256=hashes,
                  source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),updates_per_model=1000,seeds=[10501,10502,10503],modes=['cm','state_heads','shuffled'],wall_limit_seconds=600,storage_limit_bytes=1<<30,prior_engineering_seconds=10.)
    def save():(output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>570:raise TimeoutError('bounded fitter')
    save()
    try:
        keys=['history','state','candidate_actions','hold_target','initial_clearance','rest_z','assignment','motion_id','start_frame']
        d={k:torch.cat([b[k] for b in records]).cuda() for k in keys}
        obs=torch.cat([b['native_observation'][:,0] for b in records]).cuda() # NEVER future observations
        target=torch.cat([physical_targets(b) for b in records]).cuda()
        split=torch.tensor([split_group(int(i),int(j)) for i,j in zip(d['motion_id'].cpu(),d['start_frame'].cpu())],device='cuda')
        masks=dict(fit=split<50,cal=(split>=50)&(split<70),held=split>=70)
        clear=(d['initial_clearance']>=.002)&(d['state'][:,38]-d['rest_z']>=.03)
        hm=d['history'][masks['fit']].mean((0,1));hs=d['history'][masks['fit']].std((0,1),unbiased=False).clamp_min(.001)
        om=obs[masks['fit']].mean(0);ost=obs[masks['fit']].std(0,unbiased=False).clamp_min(.001)
        h=((d['history']-hm)/hs).clamp(-8,8);o=((obs-om)/ost).clamp(-8,8)
        programs=plan_features(d['candidate_actions'],d['hold_target'][:,3:6]/torch.pi);base_program=programs[:,4];actual=programs[torch.arange(len(h),device='cuda'),d['assignment']]
        fit_idx=masks['fit'].nonzero().flatten();cal_idx=(masks['cal']&clear).nonzero().flatten()
        if len(cal_idx)<48:raise ValueError('cal initiallyclear support')
        for column in (30,31):
            positives=int(target[cal_idx,column].sum());negatives=len(cal_idx)-positives
            if min(positives,negatives)<5:raise ValueError('cal event support '+str(column))
        models={};results={};checkpoint=dict(schema='ref2dex.plan_consequence.v1',native_observation_dim=o.shape[-1],normalization={k:v.cpu() for k,v in dict(history_mean=hm,history_std=hs,native_mean=om,native_std=ost).items()},models={},calibration={},margins_mm={},fit_best_fixed_option=opportunity['best_fixed_option'],input_sha256=hashes)
        def predict(model,mode,rows,all_programs=False):
            values=[]
            for start in range(0,len(rows),128):
                ids=rows[start:start+128]
                if mode=='state_heads':
                    value=model(h[ids],o[ids])
                    if not all_programs:value=value[torch.arange(len(ids),device='cuda'),d['assignment'][ids]]
                elif all_programs:
                    count=len(ids);value=model(h[ids,None].expand(-1,8,-1,-1).reshape(count*8,10,69),o[ids,None].expand(-1,8,-1).reshape(count*8,-1),programs[ids].reshape(count*8,23),base_program[ids,None].expand(-1,8,-1).reshape(count*8,23)).reshape(count,8,32)
                else:value=model(h[ids],o[ids],actual[ids],base_program[ids])
                values.append(value)
            return torch.cat(values)
        all_idx=torch.arange(len(h),device='cuda')
        for mode in manifest['modes']:
            models[mode]=[];run_results=[]
            for seed in manifest['seeds']:
                check();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=(StateOptionConsequenceModel(o.shape[-1]) if mode=='state_heads' else PlanConsequenceModel(o.shape[-1],mode='shuffled' if mode=='shuffled' else 'cm')).cuda()
                optimizer=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0001)
                g=torch.Generator(device='cuda').manual_seed(seed+30000);permuted=actual.clone()
                if mode=='shuffled':
                    shuffle_g=torch.Generator(device='cuda').manual_seed(seed+40000)
                    for stratum in (False,True):
                        ids=(masks['fit']&(clear==stratum)).nonzero().flatten();permuted[ids]=actual[ids[torch.randperm(len(ids),device='cuda',generator=shuffle_g)]]
                model.train()
                for update in range(1000):
                    if update%100==0:check()
                    ids=fit_idx[torch.randint(len(fit_idx),(128,),device='cuda',generator=g)]
                    if mode=='state_heads':pred=model(h[ids],o[ids])[torch.arange(len(ids),device='cuda'),d['assignment'][ids]]
                    else:pred=model(h[ids],o[ids],permuted[ids],base_program[ids])
                    loss=consequence_loss(pred,target[ids]);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if not torch.isfinite(loss):raise ValueError('nonfinite fit')
                model.eval();models[mode].append(model)
                with torch.no_grad():
                    rows=masks['held'].nonzero().flatten();pred=predict(model,mode,rows);truth=target[rows]
                    result=dict(seed=seed,updates=1000,final_fit_loss=float(loss),held_height_rmse_mm=float(((pred[:,:10]-truth[:,:10]).square().mean().sqrt())*10),held_clearance_mae_mm=float((pred[:,10:20]-truth[:,10:20]).abs().mean()*10))
                run_results.append(result);print(json.dumps(dict(mode=mode,**result)),flush=True)
            results[mode]=dict(runs=run_results)
            checkpoint['models'][mode]=[{k:v.detach().cpu() for k,v in model.state_dict().items()} for model in models[mode]]
            with torch.no_grad():
                cal_logits=torch.stack([predict(model,mode,cal_idx)[:,30:32] for model in models[mode]]).mean(0)
            raw_slope=torch.zeros(2,device='cuda',requires_grad=True);bias=torch.zeros(2,device='cuda',requires_grad=True)
            opt=torch.optim.Adam([raw_slope,bias],lr=.03)
            for _ in range(200):
                loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.nn.functional.softplus(raw_slope)*cal_logits+bias,target[cal_idx,30:32]);opt.zero_grad();loss.backward();opt.step()
            slope=torch.nn.functional.softplus(raw_slope).detach();bias=bias.detach();checkpoint['calibration'][mode]=dict(slope=slope.cpu(),bias=bias.cpu())
            with torch.no_grad():
                factual=torch.stack([predict(model,mode,all_idx) for model in models[mode]]).mean(0)
                prob=torch.sigmoid(factual[:,30:32]*slope+bias)
                height=(d['state'][:,None,38]+factual[:,:10]*.01-d['rest_z'][:,None])[:,-3:].amin(-1).clamp_min(0)
                score=height*prob[:,0]-(d['state'][:,38]-d['rest_z']).clamp_min(0)
                true_height=(d['state'][:,None,38]+target[:,:10]*.01-d['rest_z'][:,None])[:,-3:].amin(-1).clamp_min(0)
                true_score=true_height*target[:,30]-(d['state'][:,38]-d['rest_z']).clamp_min(0)
                margin=max(2.,float(torch.quantile((score[cal_idx]-true_score[cal_idx]).abs()*1000,.75)))
                checkpoint['margins_mm'][mode]=margin
                metrics={}
                for label,take in masks.items():
                    for subset in ('all','initially_clear'):
                        mask=take if subset=='all' else take&clear;truth=target[mask];prediction=factual[mask]
                        metrics[label+'/'+subset]=dict(rows=int(mask.sum()),height_rmse_mm=float((prediction[:,:10]-truth[:,:10]).square().mean().sqrt()*10),clearance_mae_mm=float((prediction[:,10:20]-truth[:,10:20]).abs().mean()*10),joint_force_brier=float((prediction[:,20:30].sigmoid()-truth[:,20:30]).square().mean()),retention_brier=float((prob[mask,0]-truth[:,30]).square().mean()),clearance_loss_brier=float((prob[mask,1]-truth[:,31]).square().mean()))
                results[mode].update(metrics=metrics,cal_margin_mm=margin,calibration_slope=slope.cpu().tolist(),calibration_bias=bias.cpu().tolist())
                check()
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('input drift')
        checkpoint['scope']='actual fullH10 program physics; no old2+8 reuse; pre-action native observation ONLY; no V/PPO or learned utility claim'
        torch.save(checkpoint,output/'plan_consequence.pt')
        result=dict(run_status='COMPLETED',experiment_id=manifest['experiment_id'],kind='ACTUAL_PHYSICAL_MODEL_FIT_AND_CALIBRATION',models=results,split={k:int(v.sum()) for k,v in masks.items()},initially_clear_cal=len(cal_idx),checkpoint_sha256=sha(output/'plan_consequence.pt'),elapsed_seconds=time.monotonic()-begin,utility_label='NOT_TESTED',state_only_definition='eight fixed-program heads encode catalog identity; no candidate action/anchor inputs',fixed_option=opportunity['best_fixed_option'])
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');manifest.update(run_status='COMPLETED',checkpoint_sha256=result['checkpoint_sha256'],input_hashes_unchanged=True)
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>1<<30:raise ValueError('storage limit')
        print(json.dumps(dict(status='COMPLETED',elapsed_seconds=result['elapsed_seconds'],margins=checkpoint['margins_mm'])),flush=True)
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--opportunity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',nargs='+',type=int,default=[0,1]);run(p.parse_args())
