"""Isolate frozen-BC GPU repeatability and same-packet zero-residual parity."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    scratch=ROOT/'tmp'/a.output.stem;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name in ('wam-rl-policy','shadow-finetune'):sys.path.insert(0,str(TASK.parent/name/'src'))
    import numpy as np
    import torch
    from shadow_finetune.execution import Controller
    from wam_rl_policy.ppo import apply_residual
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('240s reference replay')))
    signal.alarm(240);torch.set_num_threads(2)
    manifest=json.loads((a.run/'manifest.json').read_text())
    data=np.load(a.run/'last-episode.npz')
    identities=json.loads((a.run/'environment-objects.json').read_text())
    controller=Controller(Path(manifest['base_checkpoint']),identities,None,
        ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1')
    frozen={k:v.detach().clone() for k,v in controller.model.state_dict().items()}
    snapshot={k:data[k][0].copy() for k in ('q','dq','hand','hand_root','object','observation','goal','success','reset')}
    obs=torch.as_tensor(snapshot['observation'],device='cuda')
    values={};max_zero=0.
    for context in ('no_grad','inference'):
        for fixed in (False,True):
            rows=[]
            for _ in range(8):
                if fixed:torch.manual_seed(37);torch.cuda.manual_seed_all(37)
                scope=torch.no_grad() if context=='no_grad' else torch.inference_mode()
                with scope:
                    base=controller.act(snapshot,obs,1)
                    action,_,_=apply_residual(base,torch.zeros_like(base))
                    max_zero=max(max_zero,float((action-base).abs().max()))
                    rows.append(base.cpu().numpy().copy())
            values[context+('_fixed_rng' if fixed else '_free_rng')]=np.stack(rows)
    differences={k:dict(max_repeat_command_difference=float(abs(v-v[0]).max()),
        max_saved_initial_command_difference=float(abs(v-data['base'][0]).max())) for k,v in values.items()}
    unchanged=all(torch.equal(v,controller.model.state_dict()[k]) for k,v in frozen.items())
    report=dict(status='PASS' if max_zero==0 and unchanged else 'FAIL',zero_command_max_error=max_zero,
        frozen_weights_unchanged=unchanged,repeatability=differences,
        context_max_difference=float(abs(values['no_grad_fixed_rng']-values['inference_fixed_rng']).max()),
        scope='same saved initial16env input,8 forward repeats per context/RNG condition; not a physics determinism claim')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    np.savez(a.output.with_suffix('.npz'),**values)
    signal.alarm(0)


if __name__=='__main__':main()
