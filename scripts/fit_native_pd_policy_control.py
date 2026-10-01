#!/usr/bin/env python3
"""Frozen physical weights, one matched direct-state outcome control and calibration."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if args.output.resolve().parent!=base or args.output.exists():raise ValueError('unique owned output')
    audit_path=ROOT/'docs/experiments/probes/P-20261002-native-pd-consequence-fit-audit.json';audit=json.loads(audit_path.read_text());original=json.loads((args.original/'run_manifest.json').read_text());oldpath=args.original/'native_pd_consequence.pt'
    if audit['status']!='COMPLETED' or original['run_status']!='COMPLETED' or sha(oldpath)!=audit['checkpoint_sha256']:raise ValueError('terminal positive fit required')
    if any(sha(Path(k))!=v for k,v in original['input_sha256'].items()):raise ValueError('source drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from analyze_executable_contact_opportunity import split_group
    from src.task.CmResidual.native_pd_consequence import transitions,NativePDConsequenceModel
    from src.task.CmResidual.native_pd_policy_controls import NativeStateOutcomePolicy
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir();begin=time.monotonic();prior=audit['cumulative_probe_seconds'];m=dict(run_status='STARTED',experiment_id='P-20261002-native-pd-direct-control',family='HF16',probe_index_in_family=2,pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpu=admission,prior_seconds=prior,prior_bytes=audit['output_bytes'],models=[])
    inputs=[oldpath,args.original/'run_manifest.json',audit_path,Path(__file__),ROOT/'src/task/CmResidual/native_pd_policy_controls.py',ROOT/'docs/experiments/probes/P-20261002-native-pd-direct-control.md']+[Path(k) for k in original['input_sha256']];hashes={str(p.resolve()):sha(p) for p in inputs};m['input_sha256']=hashes
    def save():(args.output/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    save()
    try:
        old=torch.load(oldpath,map_location='cpu',weights_only=False);parts=[];bucket=[];program=[];native=[]
        records=[Path(k) for k in original['input_sha256'] if Path(k).name=='records.pt']
        for path in records:
            b=torch.load(path,map_location='cpu',weights_only=False);parts.append(transitions(b));native.append(b['native_observation'].reshape(-1,b['native_observation'].shape[-1]));program.extend(b['assignment'].repeat_interleave(10).tolist());bucket.extend([split_group(int(i),int(j)) for i,j in zip(b['motion_id'],b['start_frame']) for _ in range(10)])
        data={k:torch.cat([v[k] for v in parts]).cuda() for k in parts[0]};native=torch.cat(native).cuda();program=torch.tensor(program,device='cuda');bucket=torch.tensor(bucket,device='cuda');fit=(bucket<50).nonzero().flatten();cal=((bucket>=50)&(bucket<70)&data['clear']).nonzero().flatten();norm={k:v.cuda() for k,v in old['normalization'].items()}
        for key in ['history','physical','goal']:data[key]=((data[key]-norm[key+'_mean'])/norm[key+'_std']).clamp(-8,8)
        mean=native[fit].mean(0);std=native[fit].std(0,unbiased=False).clamp_min(.001);native=((native-mean)/std).clamp(-8,8);states=[];pred=[];row=torch.arange(len(bucket),device='cuda')
        for seed in [10601,10602,10603]:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=NativeStateOutcomePolicy(old['physical_dim'],native.shape[-1]).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0001);gen=torch.Generator(device='cuda').manual_seed(seed+30000)
            for update in range(1000):
                ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=gen)];optimizer.zero_grad(set_to_none=True);y=model(data['history'][ids],data['physical'][ids],native[ids],data['prior'][ids])[torch.arange(len(ids),device='cuda'),program[ids]];target=data['target'][ids][:,[5,3,4]]
                value=torch.nn.functional.smooth_l1_loss(y[:,0],target[:,0])+torch.nn.functional.binary_cross_entropy_with_logits(y[:,1:],target[:,1:]);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if not torch.isfinite(value):raise ValueError('nonfinite loss')
            model.eval().requires_grad_(False);values=[]
            with torch.no_grad():
                for offset in range(0,len(bucket),512):values.append(model(data['history'][offset:offset+512],data['physical'][offset:offset+512],native[offset:offset+512],data['prior'][offset:offset+512])[row[offset:offset+512]-offset,program[offset:offset+512]])
            pred.append(torch.cat(values));states.append({k:v.cpu().clone() for k,v in model.state_dict().items()});m['models'].append(dict(seed=seed,updates=1000,final_loss=float(value)));save();print(json.dumps(m['models'][-1]),flush=True)
        predictions={'state_policy':torch.stack(pred).mean(0)}
        for mode in ['cm','shuffled']:
            values=[]
            for state in old['models'][mode]:
                model=NativePDConsequenceModel(old['physical_dim'],mode).cuda();model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);chunks=[]
                with torch.no_grad():
                    for offset in range(0,len(bucket),512):chunks.append(model(data['history'][offset:offset+512],data['physical'][offset:offset+512],data['goal'][offset:offset+512],data['prior'][offset:offset+512])[:,[5,3,4]])
                values.append(torch.cat(chunks))
            predictions[mode]=torch.stack(values).mean(0)
        target=data['target'][cal][:,[5,3,4]];counts=target[:,1:].sum(0)
        if (counts<5).any() or (len(cal)-counts<5).any():raise ValueError('calibration event support insufficient')
        calibrations={};margins={};reports={}
        for mode,prediction in predictions.items():
            logits=prediction[cal,1:].detach();raw=torch.nn.Parameter(torch.full((2,),.5413248546,device='cuda'));bias=torch.nn.Parameter(torch.zeros(2,device='cuda'));optimizer=torch.optim.Adam([raw,bias],lr=.03)
            for _ in range(200):
                optimizer.zero_grad(set_to_none=True);value=torch.nn.functional.binary_cross_entropy_with_logits(logits*torch.nn.functional.softplus(raw)+bias,target[:,1:]);value.backward();optimizer.step()
            calibrations[mode]=dict(slope=torch.nn.functional.softplus(raw).detach().cpu(),bias=bias.detach().cpu());margins[mode]=max(2.,float(torch.quantile((prediction[cal,0]*10-target[:,0]*10).abs(),.75)))
            mask=(bucket>=70)&data['clear'];reports[mode]=dict(held_clear_rows=int(mask.sum()),joint_supported_height_mae_mm=float((prediction[mask,0]*10-data['target'][mask,5]*10).abs().mean()),cal_margin_mm=margins[mode])
        bundle=dict(schema='ref2dex.native_pd_controls.v1',physical_dim=old['physical_dim'],native_dim=native.shape[-1],models={mode:old['models'][mode] for mode in ['cm','shuffled']},state_policy_models=states,normalization=old['normalization'],native_mean=mean.cpu(),native_std=std.cpu(),calibration=calibrations,margins_mm=margins,fixed=7,original_checkpoint_sha256=sha(oldpath),input_sha256=hashes)
        torch.save(bundle,args.output/'native_pd_controls.pt')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('input drift')
        elapsed=time.monotonic()-begin;result=dict(run_status='COMPLETED',physical_weights_reused_exactly=True,checkpoint_sha256=sha(args.output/'native_pd_controls.pt'),cal_clear_rows=len(cal),cal_positive_events=counts.tolist(),cal_negative_events=(len(cal)-counts).tolist(),reports=reports,elapsed_seconds=elapsed,cumulative_seconds=prior+elapsed,scoped_bytes=audit['output_bytes']+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()))
        if result['cumulative_seconds']>3540 or result['scoped_bytes']>8<<30:raise ValueError('budget')
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');m.update(run_status='COMPLETED',result=result);print(json.dumps(result,indent=2))
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:m['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
