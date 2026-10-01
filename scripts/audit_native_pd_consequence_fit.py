#!/usr/bin/env python3
"""Terminal independent pre-step timing/label/normalizer audit for HF16."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    begin=time.monotonic();m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or args.output.exists():raise ValueError('terminal/unique audit required')
    try:os.kill(m['pid'],0);raise ValueError('fit PID still present')
    except ProcessLookupError:pass
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('input drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.native_pd_consequence import transitions
    from src.task.CmResidual.executable_contact_options import INDEPENDENT
    from analyze_executable_contact_opportunity import split_group
    torch.set_num_threads(2);records=[Path(k) for k in m['input_sha256'] if Path(k).name=='records.pt'];fit=[];rows=0
    for path in records:
        b=torch.load(path,map_location='cpu',weights_only=False);d=transitions(b);n=len(b['state']);rows+=n*10
        for step in range(10):
            ids=torch.arange(n)*10+step;pre=b['state'] if step==0 else b['future_state'][:,step-1]
            hand=b['initial_hand_force'] if step==0 else b['future_hand_force'][:,step-1];obj=b['initial_object_force'] if step==0 else b['future_object_force'][:,step-1]
            gap=b['initial_clearance'] if step==0 else b['future_clearance'][:,step-1]
            height=(pre[:,38]-b['rest_z']).clamp_min(0);weight=b['mass_kg'][:,None]*b['gravity_magnitude']
            force=torch.cat((hand.flatten(-2),obj),-1)/weight;force=force.sign()*force.abs().log1p()
            expected=torch.cat((force,gap[:,None],height[:,None]),-1)
            if not torch.equal(d['history'][ids,-1,:49],pre) or not torch.equal(expected,d['physical'][ids]):raise ValueError('post-action state/forces in input')
            if not torch.equal(d['goal'][ids],(b['actual_pd_targets'][:,step]-pre[:,:18])[:,list(INDEPENDENT)]):raise ValueError('native goal alignment')
            pair=b['future_contact'][:,step].all(-1);support=pair&(b['future_clearance'][:,step]>=.002)
            dz=(b['future_state'][:,step,38]-pre[:,38])/.002;dc=(b['future_clearance'][:,step]-gap)/.002
            score=((b['future_state'][:,step,38]-b['rest_z']).clamp_min(0)*support-height)/.01
            expected=torch.stack((dz,dc,pair.float(),support.float(),(b['future_clearance'][:,step]<.002).float(),score),-1)
            if not torch.equal(expected,d['target'][ids]):raise ValueError('joint physical target mismatch')
        mask=torch.tensor([split_group(int(i),int(j))<50 for i,j in zip(b['motion_id'],b['start_frame'])]).repeat_interleave(10)
        fit.append({k:d[k][mask] for k in ['history','physical','goal']})
    checkpoint=args.run/'native_pd_consequence.pt';p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if sha(checkpoint)!=m['result']['checkpoint_sha256']:raise ValueError('weight drift')
    max_error=0.
    for key in fit[0]:
        x=torch.cat([v[key] for v in fit]).cuda();dims=(0,1) if key=='history' else 0;mean=x.mean(dims).cpu();std=x.std(dims,unbiased=False).clamp_min(.001).cpu()
        max_error=max(max_error,float((mean-p['normalization'][key+'_mean']).abs().max()),float((std-p['normalization'][key+'_std']).abs().max()))
    if max_error>2e-6:raise ValueError('normalization not fit-only')
    result=dict(status='COMPLETED',rows=rows,own_fit_pid_exited=True,input_hashes_verified=True,pre_step_state_force_goal_timing_verified=True,joint_target_independently_recomputed=True,fit_only_normalization_max_error=max_error,checkpoint_sha256=sha(checkpoint),original_gate=m['result']['label'],elapsed_seconds=time.monotonic()-begin,cumulative_probe_seconds=m['elapsed_seconds']+time.monotonic()-begin,output_bytes=sum(p.stat().st_size for p in args.run.rglob('*') if p.is_file()),gpu=admission,result=m['result'])
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
