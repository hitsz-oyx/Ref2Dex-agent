#!/usr/bin/env python3
"""Matched state outcome/head-shuffled controls, reusing the frozen catalog Cm."""
import argparse,ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def bucket(motion,start):
    return int(hashlib.sha256(f'9851/{motion}/{start}'.encode()).hexdigest()[:8],16)%100


def run(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve();out=args.output.resolve()
    if out.parent!=base or args.output.is_symlink():raise ValueError('unique owned output required')
    audit=json.loads(args.fit_audit.read_text());fm=json.loads((args.fit/'run_manifest.json').read_text());m=json.loads((args.run/'run_manifest.json').read_text())
    if audit['status']!='COMPLETED' or fm['run_status']!='COMPLETED' or m['run_status']!='COMPLETED':raise ValueError('audited terminal inputs required')
    checkpoint=args.fit/'plan_consequence.pt'
    if sha(checkpoint)!=audit['checkpoint_sha256']:raise ValueError('old checkpoint changed')
    model_path=ROOT/'src/task/CmResidual/plan_consequence_model.py'
    old=subprocess.check_output(['git','show',fm['source_git_commit']+':src/task/CmResidual/plan_consequence_model.py'],cwd=ROOT)
    if hashlib.sha256(old).hexdigest()!=fm['input_sha256'][str(model_path.resolve())]:raise ValueError('frozen model source provenance')
    def definitions(source):return {n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    before=definitions(old);now=definitions(model_path.read_text())
    if any(before[k]!=now[k] for k in ['StateOptionConsequenceModel','physical_targets','consequence_loss']):raise ValueError('reused physical decoder/labels changed')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('opportunity input drift')
    admission=None
    for gpu in args.gpus:
        try:admission=gpu_admission(gpu);break
        except RuntimeError:continue
    if admission is None:raise RuntimeError('allowed GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.plan_consequence_model import StateOptionConsequenceModel,StateOptionOutcomeModel,physical_targets,consequence_loss
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    p=torch.load(checkpoint,map_location='cpu',weights_only=False);bs=[];inputs=[Path(__file__),model_path,checkpoint,args.fit_audit,args.run/'run_manifest.json']
    for phase in m['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['result']['record_sha256']:raise ValueError('actual record changed')
        b=torch.load(path,map_location='cpu',weights_only=False)
        if b['schema']!='ref2dex.orientation_feedback_options.v1' or b['future_done'].any():raise ValueError('complete actual labels')
        bs.append(b);inputs.append(path)
    hashes={str(path.resolve()):sha(path) for path in inputs};out.mkdir(exist_ok=False);begin=time.monotonic()
    prior=audit['elapsed_seconds']+10.+sum(json.loads((ROOT/'docs/experiments/probes'/name).read_text())['elapsed_seconds'] for name in ['P-20261002-plan-selector-coverage.json','P-20261002-plan-selector-coverage-r2.json'])
    manifest=dict(run_status='RUNNING',experiment_id='P-20261002-plan-consequence-control',family='HF15',probe_index_in_family=2,pid=os.getpid(),gpu=admission,input_sha256=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),prior_seconds=prior,prior_bytes=audit['scoped_bytes'],wall_limit_seconds=600,seeds=[10501,10502,10503],updates_each=1000,physical_model_reused_exactly=True)
    def save():(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>570:raise TimeoutError('control fitting budget')
    save()
    try:
        keys=['history','state','rest_z','assignment','motion_id','start_frame','initial_clearance']
        d={k:torch.cat([b[k] for b in bs]).cuda() for k in keys};obs=torch.cat([b['native_observation'][:,0] for b in bs]).cuda();target=torch.cat([physical_targets(b) for b in bs]).cuda()
        buckets=torch.tensor([bucket(int(i),int(j)) for i,j in zip(d['motion_id'].cpu(),d['start_frame'].cpu())],device='cuda');fit=buckets<50;cal=(buckets>=50)&(buckets<70);held=buckets>=70
        clear=(d['initial_clearance']>=.002)&(d['state'][:,38]-d['rest_z']>=.03);fit_ids=fit.nonzero().flatten();cal_ids=(cal&clear).nonzero().flatten();norm={k:v.cuda() for k,v in p['normalization'].items()}
        h=((d['history']-norm['history_mean'])/norm['history_std']).clamp(-8,8);o=((obs-norm['native_mean'])/norm['native_std']).clamp(-8,8)
        rm=d['rest_z'][fit].mean();rs=d['rest_z'][fit].std(unbiased=False).clamp_min(.001);rest=((d['rest_z']-rm)/rs).clamp(-8,8)
        height=(d['state'][:,None,38]+target[:,:10]*.01-d['rest_z'][:,None])[:,-3:].amin(-1).clamp_min(0)
        actual_score=(height*target[:,30]-(d['state'][:,38]-d['rest_z']).clamp_min(0))*1000
        outcome_target=torch.cat((actual_score[:,None]/10,target[:,30:32]),-1)
        bundle=dict(schema='ref2dex.catalog_consequence_controls.v1',native_observation_dim=o.shape[-1],normalization=p['normalization'],task_rest_mean=rm.cpu(),task_rest_std=rs.cpu(),models={'catalog_cm':p['models']['state_heads']},calibration={'catalog_cm':p['calibration']['state_heads']},margins_mm={'catalog_cm':p['margins_mm']['state_heads']},fit_best_fixed_option=p['fit_best_fixed_option'],original_checkpoint_sha256=sha(checkpoint),input_sha256=hashes)
        result={}
        def predict(model,mode,ids):
            values=[]
            for offset in range(0,len(ids),128):
                rows=ids[offset:offset+128];values.append(model(h[rows],o[rows],rest[rows]) if mode=='state_policy' else model(h[rows],o[rows]))
            return torch.cat(values)
        all_ids=torch.arange(len(h),device='cuda')
        for mode in ['state_policy','catalog_shuffled']:
            trained=[];runs=[]
            for seed in manifest['seeds']:
                check();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=(StateOptionOutcomeModel(o.shape[-1]) if mode=='state_policy' else StateOptionConsequenceModel(o.shape[-1])).cuda()
                opt=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0001);g=torch.Generator(device='cuda').manual_seed(seed+30000);assigned=d['assignment'].clone()
                if mode=='catalog_shuffled':
                    sg=torch.Generator(device='cuda').manual_seed(seed+40000)
                    for stratum in [False,True]:
                        ids=(fit&(clear==stratum)).nonzero().flatten();assigned[ids]=d['assignment'][ids[torch.randperm(len(ids),device='cuda',generator=sg)]]
                model.train()
                for update in range(1000):
                    if update%100==0:check()
                    ids=fit_ids[torch.randint(len(fit_ids),(128,),device='cuda',generator=g)];bank=model(h[ids],o[ids],rest[ids]) if mode=='state_policy' else model(h[ids],o[ids]);pred=bank[torch.arange(128,device='cuda'),assigned[ids]]
                    loss=(torch.nn.functional.smooth_l1_loss(pred[:,:1],outcome_target[ids,:1])+torch.nn.functional.binary_cross_entropy_with_logits(pred[:,1:],outcome_target[ids,1:])) if mode=='state_policy' else consequence_loss(pred,target[ids])
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                    if not torch.isfinite(loss):raise ValueError('nonfinite control loss')
                model.eval().requires_grad_(False);trained.append(model);runs.append(dict(seed=seed,updates=1000,final_loss=float(loss)));print(json.dumps(dict(mode=mode,**runs[-1])),flush=True)
            bundle['models'][mode]=[{k:v.cpu() for k,v in model.state_dict().items()} for model in trained]
            with torch.no_grad():
                pred=torch.stack([predict(model,mode,all_ids) for model in trained]).mean(0);factual=pred[torch.arange(len(h),device='cuda'),d['assignment']];cols=slice(1,3) if mode=='state_policy' else slice(30,32)
                logits=factual[cal_ids,cols]
            raw=torch.zeros(2,device='cuda',requires_grad=True);bias=torch.zeros_like(raw,requires_grad=True);opt=torch.optim.Adam([raw,bias],lr=.03)
            for _ in range(200):
                loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.nn.functional.softplus(raw)*logits+bias,target[cal_ids,30:32]);opt.zero_grad();loss.backward();opt.step()
            slope=torch.nn.functional.softplus(raw).detach();bias=bias.detach();bundle['calibration'][mode]=dict(slope=slope.cpu(),bias=bias.cpu())
            with torch.no_grad():
                prob=torch.sigmoid(factual[:,cols]*slope+bias)
                if mode=='state_policy':score=factual[:,0]*10
                else:
                    heights=(d['state'][:,None,38]+factual[:,:10]*.01-d['rest_z'][:,None])[:,-3:].amin(-1).clamp_min(0);score=(heights*prob[:,0]-(d['state'][:,38]-d['rest_z']).clamp_min(0))*1000
                margin=max(2.,float(torch.quantile((score[cal_ids]-actual_score[cal_ids]).abs(),.75)));bundle['margins_mm'][mode]=margin
                metrics={}
                for name,mask in [('fit',fit),('cal',cal),('held',held),('held_clear',held&clear)]:
                    metrics[name]=dict(rows=int(mask.sum()),score_rmse_mm=float((score[mask]-actual_score[mask]).square().mean().sqrt()),retention_brier=float((prob[mask,0]-target[mask,30]).square().mean()),clearance_loss_brier=float((prob[mask,1]-target[mask,31]).square().mean()))
                result[mode]=dict(runs=runs,margin_mm=margin,metrics=metrics)
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('control inputs changed')
        bundle['scope']='catalog physicalfuture vs direct state localoutcome vs shuffled catalog; no retroactive win over original fixedhead itself'
        torch.save(bundle,out/'catalog_controls.pt');report=dict(run_status='COMPLETED',models=result,reused_catalog_cm_margin_mm=bundle['margins_mm']['catalog_cm'],checkpoint_sha256=sha(out/'catalog_controls.pt'),elapsed_seconds=time.monotonic()-begin,utility_label='NOT_TESTED')
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');manifest.update(run_status='COMPLETED',input_hashes_unchanged=True,checkpoint_sha256=report['checkpoint_sha256'],cumulative_seconds=prior+time.monotonic()-begin,output_bytes=manifest['prior_bytes']+sum(path.stat().st_size for path in out.rglob('*') if path.is_file()))
        if manifest['output_bytes']>1<<30:raise ValueError('control storage limit')
        print(json.dumps(dict(status='COMPLETED',seconds=report['elapsed_seconds'],margins=bundle['margins_mm'])),flush=True)
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--fit-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',default=[0,1]);run(p.parse_args())
