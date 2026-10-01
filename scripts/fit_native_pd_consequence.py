#!/usr/bin/env python3
"""Bounded matched GPU information probe of actual native control consequences."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if args.output.resolve().parent!=base or args.output.exists() or args.output.is_symlink():raise ValueError('unique owned output required')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from analyze_executable_contact_opportunity import split_group
    from src.task.CmResidual.native_pd_consequence import NativePDConsequenceModel,transitions,loss
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    source=json.loads((args.source/'run_manifest.json').read_text())
    if source['run_status']!='COMPLETED' or source['smoke_only']:raise ValueError('terminal original program data required')
    inputs=[args.source/'run_manifest.json',ROOT/'scripts/fit_native_pd_consequence.py',ROOT/'src/task/CmResidual/native_pd_consequence.py',ROOT/'docs/decisions/D-20261002-native-pd-consequence.md']
    for k,v in source['input_sha256'].items():
        if sha(Path(k))!=v:raise ValueError('original source/input drift')
    inputs+=[Path(k) for k in source['input_sha256']]
    args.output.mkdir();begin=time.monotonic();manifest=dict(run_status='STARTED',pid=os.getpid(),experiment_id='P-20261002-native-pd-consequence',family='HF16',probe_index_in_family=1,command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpu=admission,input_sha256={},wall_limit_seconds=3600,output_limit_bytes=8<<30,models=[],source='original HF15 whole-program records, no new held utility fitting')
    def save():(args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        parts=[];buckets=[];episodes=[];groups=[]
        for p in source['phases']:
            path=Path(p['directory'])/'records.pt'
            if sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
            inputs.append(path);b=torch.load(path,map_location='cpu',weights_only=False);parts.append(transitions(b))
            buckets.extend([split_group(int(i),int(j)) for i,j in zip(b['motion_id'],b['start_frame']) for _ in range(10)])
            episodes.extend([ep for ep in b['episode_id'] for _ in range(10)]);groups.extend([f'{int(i)}/{int(j)}' for i,j in zip(b['motion_id'],b['start_frame']) for _ in range(10)])
        hashes={str(p.resolve()):sha(p) for p in inputs};manifest['input_sha256']=hashes
        data={k:torch.cat([p[k] for p in parts]).cuda() for k in parts[0]};bucket=torch.tensor(buckets,device='cuda');split={'fit':bucket<50,'cal':(bucket>=50)&(bucket<70),'held':bucket>=70}
        normalization={}
        for key in ['history','physical','goal']:
            x=data[key][split['fit']];dims=(0,1) if key=='history' else 0;mean=x.mean(dims);std=x.std(dims,unbiased=False).clamp_min(.001)
            normalization[key+'_mean']=mean;normalization[key+'_std']=std;data[key]=((data[key]-mean)/std).clamp(-8,8)
        fit=split['fit'].nonzero().flatten();cal=split['cal'].nonzero().flatten();held=split['held'];clear=held&data['clear'];clear_cpu=clear.cpu().tolist()
        support=dict(transitions=int(clear.sum()),episodes=len({ep for ep,m in zip(episodes,clear_cpu) if m}),initial_groups=len({g for g,m in zip(groups,clear_cpu) if m}))
        manifest.update(run_status='RUNNING',split_rows={k:int(v.sum()) for k,v in split.items()},clear_support=support);save();models={};metrics={}
        def check():
            if time.monotonic()-begin>3540:raise TimeoutError('whole probe budget')
            if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>8<<30:raise ValueError('storage limit')
            if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('source/input drift')
        for mode in ['cm','state_only','shuffled']:
            trained=[];pred=[];runs=[]
            for seed in [10601,10602,10603]:
                check();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=NativePDConsequenceModel(data['physical'].shape[-1],mode).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0001)
                gen=torch.Generator(device='cuda').manual_seed(seed+30000);goals=data['goal'].clone()
                if mode=='shuffled':
                    shuffle=torch.Generator(device='cuda').manual_seed(seed+40000)
                    for condition in [False,True]:
                        rows=fit[data['clear'][fit]==condition];goals[rows]=data['goal'][rows[torch.randperm(len(rows),device='cuda',generator=shuffle)]]
                model.train()
                for update in range(1000):
                    ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=gen)];optimizer.zero_grad(set_to_none=True)
                    y=model(data['history'][ids],data['physical'][ids],goals[ids],data['prior'][ids]);value=loss(y,data['target'][ids]);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if not torch.isfinite(value):raise ValueError('nonfinite loss')
                model.eval().requires_grad_(False);outputs=[]
                with torch.no_grad():
                    for offset in range(0,len(bucket),512):outputs.append(model(data['history'][offset:offset+512],data['physical'][offset:offset+512],data['goal'][offset:offset+512],data['prior'][offset:offset+512]))
                pred.append(torch.cat(outputs));trained.append({k:v.cpu().clone() for k,v in model.state_dict().items()});runs.append(dict(seed=seed,updates=1000,final_loss=float(value)));manifest['models'].append(dict(mode=mode,**runs[-1]));save();print(json.dumps(dict(mode=mode,seed=seed,status='COMPLETED')),flush=True)
                del model,optimizer
            mean=torch.stack(pred).mean(0);target=data['target'];report={}
            for name,mask in [('held',held),('held_clear',clear),('cal',split['cal'])]:
                delta=mean[mask]-target[mask];joint=torch.sigmoid(mean[mask,2]);report[name]=dict(rows=int(mask.sum()),height_rmse_mm=float(delta[:,0].square().mean().sqrt()*2),clearance_mae_mm=float(delta[:,1].abs().mean()*2),joint_brier=float((joint-target[mask,2]).square().mean()),persist_joint_brier=float((data['persist_joint'][mask]-target[mask,2]).square().mean()),joint_supported_height_mae_mm=float(delta[:,5].abs().mean()*10))
            models[mode]=trained;metrics[mode]=report
        adequate=support['transitions']>=500 and support['episodes']>=24 and support['initial_groups']>=8;cm=metrics['cm']['held_clear'];gates=dict(support=adequate,joint_no_worse_than_persist=cm['joint_brier']<=cm['persist_joint_brier'])
        for control in ['state_only','shuffled']:
            other=metrics[control]['held_clear'];gates[control+'_height5pct']=cm['height_rmse_mm']<=.95*other['height_rmse_mm'];gates[control+'_clearance5pct']=cm['clearance_mae_mm']<=.95*other['clearance_mae_mm'];gates[control+'_joint_height']=cm['joint_supported_height_mae_mm']<=other['joint_supported_height_mae_mm']
        gates['passed']=all(gates.values());label='PROMISING' if gates['passed'] else ('UNPROMISING' if adequate else 'UNCLEAR')
        checkpoint=dict(schema='ref2dex.native_pd_consequence.v1',physical_dim=data['physical'].shape[-1],models=models,normalization={k:v.cpu() for k,v in normalization.items()},target_definition='native actual one-step height/CLR/joint/support/loss/joint-supported-height; velocity and observed proxy skip',input_sha256=hashes)
        torch.save(checkpoint,args.output/'native_pd_consequence.pt');check();result=dict(run_status='COMPLETED',label=label,metrics=metrics,gate=gates,support=support,elapsed_seconds=time.monotonic()-begin,checkpoint_sha256=sha(args.output/'native_pd_consequence.pt'),scope='factual one-step intervention information, pre motor-target/raw-force state only; no multistep forecast/planning/utility/final grasp claim')
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');manifest.update(run_status='COMPLETED',result=result,input_hashes_unchanged=True);print(json.dumps(result,indent=2))
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
