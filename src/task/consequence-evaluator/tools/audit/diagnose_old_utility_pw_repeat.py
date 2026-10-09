"""Bounded first-batch feedback loop for frozen PW numeric repeatability."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--stable-mean',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TRITON_CACHE_DIR=str(ROOT/'tmp/triton_old_utility'))
    import torch
    sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/cm-pointflow-effect-pretrain/src')]
    from consequence_evaluator.old_utility import pw_sample,deterministic_group_mean
    from consequence_evaluator.data import sha
    from oakink_wm.pointworld_temporal import model_from_config,capped_collate
    from oakink_wm.pointworld_performance import install_fused_hilbert
    if a.stable_mean:
        import oakink_wm.pointworld as spatial
        import oakink_wm.pointworld_temporal as temporal
        spatial.mean_groups=temporal.mean_groups=deterministic_group_mean
    torch.set_num_threads(2);torch.manual_seed(292);install_fused_hilbert();start=time.monotonic()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=model_from_config(state['identity']['stats'],state['config']).cuda().eval();model.load_state_dict(state['model']);del state
    with np.load(a.data/'windows.npz') as f:d={k:f[k] for k in ('pw_object_history','pw_hand_history','pw_hand_future')}
    with np.load(a.canonical) as f:c={k:f[k] for k in f.files}
    trials=[]
    for size in (1,8):
        samples=[pw_sample(d['pw_object_history'][i],d['pw_hand_history'][i],d['pw_hand_future'][i],c,i) for i in range(size)]
        batch={k:v.cuda() for k,v in capped_collate(samples).items()};before={k:v.clone() for k,v in batch.items()}
        for amp in (True,False):
            torch.backends.cuda.matmul.allow_tf32=amp
            rng=torch.cuda.get_rng_state();outs=[]
            with torch.inference_mode():
                for step in range(3):
                    if step==2:torch.cuda.set_rng_state(rng)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):outs.append(model(batch,'action'))
            diff=lambda x,y:{k:float((x[k]-y[k]).abs().max()) for k in ('translation','rotation')}
            row=dict(batch=size,amp=amp,repeat_max_abs=diff(outs[0],outs[1]),rng_reset_max_abs=diff(outs[0],outs[2]),
                input_mutated=any(not torch.equal(before[k],batch[k]) for k in batch),
                cuda_rng_advanced=not torch.equal(rng,torch.cuda.get_rng_state()),
                stochastic_eval_modules=sum(bool(getattr(m,'shuffle_orders',False)) for m in model.modules()))
            trials.append(row);print(json.dumps(row),flush=True)
    record=dict(trials=trials,stable_mean=a.stable_mean,elapsed_s=time.monotonic()-start,
        hypotheses=['atomic reduction numeric differences','stochastic serialization','mutable input','mixed precision sensitivity'],
        input_sha256={str(path.resolve()):sha(path) for path in (a.data/'windows.npz',a.canonical,a.checkpoint,Path(__file__))})
    a.output.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
