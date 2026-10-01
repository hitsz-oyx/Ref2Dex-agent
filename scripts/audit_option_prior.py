#!/usr/bin/env python3
"""Mechanical learned-logit/command audit; no outcome-based policy selection."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    admission=gpu_admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in [str(args.gpu),admission['uuid']]:raise ValueError('explicit admitted GPU required')
    import torch
    from src.task.CmResidual.recovery_option_policy import RecoveryOptionPolicy
    from src.task.CmResidual.physical_value_contract import private_initialization
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;begin=time.monotonic();report={};hashes={}
    for mode in ['on','off']:
        checkpoint=args.run/f'train_{mode}/policy.pt';hashes[str(checkpoint)]=sha(checkpoint)
        saved=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
        with private_initialization(9381):policy=RecoveryOptionPolicy().to('cuda:0')
        policy.load_state_dict(saved['policy']);policy.eval().requires_grad_(False)
        gains=[];margins=[];unguided=[];guided=[];probdiff=[]
        with torch.no_grad():
            for seed in range(391,395):
                p=args.run/f'eval_{mode}_s{seed}/decisions.pt';hashes[str(p)]=sha(p)
                buffers=torch.load(p,map_location='cpu',weights_only=False)
                for b in buffers:
                    for start in range(0,len(b['selected']),256):
                        x={k:v[start:start+256].to('cuda:0') for k,v in b['inputs'].items()}
                        prior=x['log_prior'].argmax(-1);rows=torch.arange(len(prior),device='cuda:0')
                        dist,_=policy(**x);zero=dict(x,log_prior=torch.zeros_like(x['log_prior']));free,_=policy(**zero)
                        # Normalization adds the same constant to every logit.
                        residual=free.logits-free.logits[rows,prior,None]
                        other=residual.clone();other[rows,prior]=-torch.inf
                        gains.append(other.amax(-1).cpu());margins.append((dist.logits[rows,prior]-dist.logits.masked_fill(torch.nn.functional.one_hot(prior,6).bool(),-torch.inf).amax(-1)).cpu())
                        unguided.append((free.probs.argmax(-1)!=prior).cpu());guided.append((dist.probs.argmax(-1)!=prior).cpu())
                        probdiff.append((dist.probs-x['log_prior'].exp()).abs().amax(-1).cpu())
                        if not torch.equal(dist.probs.argmax(-1).cpu(),b['selected'][start:start+256]):raise ValueError('actual command replay mismatch')
        def quantile(chunks):return dict(zip(['min','median','p90','max'],torch.quantile(torch.cat(chunks),torch.tensor([0.,.5,.9,1.])).tolist()))
        report[mode]=dict(rows=sum(len(v) for v in gains),learned_other_advantage_logits=quantile(gains),actual_prior_winner_margin_logits=quantile(margins),
                          learned_probability_shift_max_per_row=quantile(probdiff),unguided_argmax_differs_from_prior=int(torch.cat(unguided).sum()),
                          actual_guided_argmax_differs_from_prior=int(torch.cat(guided).sum()))
    result=dict(run_status='COMPLETED',label='MECHANICAL_AUDIT',variants=report,fixed_prior_log_odds=float(torch.tensor(45.).log()),
                checkpoint_and_records_unchanged=all(sha(Path(p))==v for p,v in hashes.items()),input_sha256=hashes,gpu=admission,
                elapsed_seconds=time.monotonic()-begin,scope='removing prior only in offline arithmetic, no actual alternative execution or utility claim; no refit or threshold search')
    if args.output.exists():raise ValueError('audit exists')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','gpu']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);run(p.parse_args())
