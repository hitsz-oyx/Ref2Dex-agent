#!/usr/bin/env python3
"""GPU verify exact frozen weights and cached motor-conditioned scorer wiring."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    start=time.monotonic();admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.native_pd_selector import FrozenNativePDSelector,physical_inputs
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=torch.load(args.original,map_location='cpu',weights_only=False);bundle=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if args.output.exists():raise ValueError('audit exists')
    for mode in ['cm','shuffled']:
        for a,b in zip(old['models'][mode],bundle['models'][mode]):
            if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('physical weights changed')
    if any(not torch.equal(old['normalization'][k],bundle['normalization'][k]) for k in old['normalization']):raise ValueError('physical normalization changed')
    selector=FrozenNativePDSelector(args.checkpoint,'cuda');m=json.loads((args.source/'run_manifest.json').read_text());p=m['phases'][0];path=Path(p['directory'])/'records.pt'
    if sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
    b=torch.load(path,map_location='cpu',weights_only=False);ids=torch.arange(min(96,len(b['state'])));h=b['history'][ids].cuda();native=b['native_observation'][ids,0].cuda();rest=b['rest_z'][ids].cuda();hand=b['initial_hand_force'][ids].cuda();obj=b['initial_object_force'][ids].cuda();mass=b['mass_kg'][ids].cuda();clr=b['initial_clearance'][ids].cuda();motor=b['actual_pd_targets'][ids,0].cuda();candidates=motor[:,None].expand(-1,8,-1)
    physical,prior=physical_inputs(h,hand,obj,mass,b['gravity_magnitude'],clr,rest);norm=selector.norm
    nh=((h-norm['history_mean'])/norm['history_std']).clamp(-8,8);nf=((physical-norm['physical_mean'])/norm['physical_std']).clamp(-8,8)
    from src.task.CmResidual.executable_contact_options import INDEPENDENT
    goal=(motor-h[:,-1,:18])[:,list(INDEPENDENT)];ng=((goal-norm['goal_mean'])/norm['goal_std']).clamp(-8,8)
    error=0.;reports={}
    for mode in ['cm','shuffled','state_policy']:
        choice,d=selector.choose(mode,h,native,rest,candidates,hand,obj,mass,b['gravity_magnitude'],clr)
        if not torch.isfinite(d['score_mm']).all():raise ValueError('nonfinite scorer')
        if mode!='state_policy':
            with torch.no_grad():point=torch.stack([model(nh,nf,ng,prior) for model in selector.models[mode]]).mean(0)
            error=max(error,float((d['score_mm'][:,0]-point[:,5]*10).abs().max()))
            if (choice!=4).any() or not torch.equal(d['score_mm'],d['score_mm'][:,:1].expand(-1,8)):raise ValueError('identical actuated commands get different physics score')
        reports[mode]=dict(rows=len(ids),ood=int(d['ood'].sum()),nonbase=int((choice!=4).sum()))
    if error>2e-3:raise ValueError('cached encoder changed physical prediction')
    result=dict(status='COMPLETED',checkpoint_sha256=sha(args.checkpoint),selector_source_sha256=sha(ROOT/'src/task/CmResidual/native_pd_selector.py'),physical_weights_reused_exactly=True,cached_vs_factual_forward_score_max_error_mm=error,identical_native_motor_commands_invariant=True,modes=reports,gpu=admission,elapsed_seconds=time.monotonic()-start,scope='NN wiring on repeated actual motor command; state-policy heads are not a candidate opportunity or utility measurement')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
