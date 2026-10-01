#!/usr/bin/env python3
"""Terminal GPU score/execution replay; no training or counterfactual outcomes."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import gpu_admission,sha


def run(args):
    m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only'] or not m.get('learnable_guide'):raise ValueError('terminal HF12 science required')
    if args.output.exists():raise ValueError('audit output exists')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.recovery_option_policy import RecoveryOptionPolicy
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();results={};artifacts={}
    with torch.no_grad():
        for mode in ['on','off']:
            directory=args.run/f'train_{mode}';saved=torch.load(directory/'policy.pt',map_location='cpu',weights_only=False)
            model=RecoveryOptionPolicy(learnable_guide=True).cuda();model.load_state_dict(saved['policy']);model.eval()
            coefficient=float(model.guide_weight);advantages=[];margins=[];greedy_errors=0;count=0
            for seed in m['evaluation_seeds']:
                directory=args.run/f'eval_{mode}_s{seed}'
                phase=next(p for p in m['phases'] if p['name']==directory.name)
                if sha(directory/'policy.pt')!=phase['result']['policy_sha256'] or sha(directory/'decisions.pt')!=phase['result']['decisions_sha256']:raise ValueError('artifact drift')
                artifacts[str(directory/'decisions.pt')]=sha(directory/'decisions.pt')
                b=torch.load(directory/'decisions.pt',map_location='cpu',weights_only=False)[0]
                for offset in range(0,len(b['selected']),256):
                    x={k:v[offset:offset+256].cuda() for k,v in b['inputs'].items()};selected=b['selected'][offset:offset+256].cuda()
                    dist,_=model(**x);guide=x['physics'][:,:,-1];reference=guide.argmax(-1)
                    row=torch.arange(len(selected),device='cuda');score=dist.logits-coefficient*guide
                    relative=score-score[row,reference,None];relative[row,reference]=-torch.inf
                    advantages.append(relative.max(-1).values.cpu());margins.append((dist.logits[row,reference]-dist.logits.masked_fill(guide.bool(),-torch.inf).max(-1).values).cpu())
                    # Permit only arithmetic-level ties from different batch shape.
                    greedy_errors+=int((dist.logits.max(-1).values-dist.logits[row,selected]>2e-6).sum());count+=len(row)
            if greedy_errors:raise ValueError('greedy checkpoint execution replay mismatch')
            advantage=torch.cat(advantages);margin=torch.cat(margins)
            results[mode]=dict(decisions=count,learned_guide_weight=coefficient,initial_guide_weight=1.6094379124341003,
                               maximum_nn_competitor_advantage=float(advantage.max()),minimum_total_guide_winner_margin=float(margin.min()),
                               decisions_nn_overrides_guide=int((margin<0).sum()),greedy_replay_errors=greedy_errors)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    output=dict(run_status='COMPLETED',gpu=admission,elapsed_seconds=elapsed,modes=results,input_sha256=artifacts,
                scope='mechanical score/execution audit only; coefficient belongs to trained actor, not a fixed external prior; no alternative physical outcomes or learning utility inferred')
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output['modes'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=3)
    run(p.parse_args())
