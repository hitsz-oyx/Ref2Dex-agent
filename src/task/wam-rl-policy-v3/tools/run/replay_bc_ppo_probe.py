"""Reconstruct every PPO update on GPU7 from saved rollouts and compare final weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh output required')
    occupied=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    if occupied>64:raise RuntimeError('GPU7 occupied; wait for paired assessment to exit')
    os.environ.update(CUDA_VISIBLE_DEVICES='7',TMPDIR=str(ROOT/'tmp'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    run=a.run.resolve();m=json.loads((run/'manifest.json').read_text())
    if m['status']!='COMPLETED' or not m['ppo']:raise ValueError('completed PPO run required')
    sys.path.insert(0,str(run/'runtime-code/src/task/wam-rl-policy-v3/src'))
    import torch
    from wam_rl_policy_v3.intent_residual import IntentResidualPolicy,ppo_update,RESIDUAL_SCALE
    torch.set_num_threads(2)
    policy=IntentResidualPolicy().cuda();policy.load_state_dict(torch.load(m['residual_checkpoint'],map_location='cpu',weights_only=False)['model']);optimizer=torch.optim.Adam(policy.parameters(),lr=1e-5)
    records=[]
    for row in m['runs']:
        with np.load(run/row['id']/'ppo-rollout.npz') as z:rollout={k:z[k].copy() for k in z.files}
        saved=row['ppo_update'];before=policy.state_dict()['mean.weight'].detach().clone()
        reconstructed=ppo_update(policy,optimizer,**rollout,bootstrap=saved['bootstrap'],epochs=2,target_kl=.03)
        kl_error=max(abs(x['exact_kl']-y['exact_kl']) for x,y in zip(saved['epochs'],reconstructed['epochs']))
        clip_error=max(abs(x['clip_fraction']-y['clip_fraction']) for x,y in zip(saved['epochs'],reconstructed['epochs']))
        parts=json.loads((run/row['id']/'reward-parts.json').read_text());reward_error=float(np.max(np.abs(rollout['rewards']-np.asarray([sum(v.values()) for v in parts]))))
        bound_excess=float(np.maximum(np.abs(np.tanh(rollout['latents'])*RESIDUAL_SCALE)-RESIDUAL_SCALE,0).max())
        gae_error=abs(saved['target_return_mean']-reconstructed['target_return_mean'])
        match=bool(kl_error<2e-5 and clip_error<2e-5 and reward_error<1e-9 and bound_excess==0 and gae_error<2e-5 and saved['optimizer_updates']==reconstructed['optimizer_updates'])
        records.append(dict(id=row['id'],match=match,kl_error=kl_error,clip_error=clip_error,reward_parts_error=reward_error,bound_excess=bound_excess,gae_return_error=gae_error,behavior_logprob_error=reconstructed['behavior_logprob_error']))
    final=torch.load(run/'residual-final.pt',map_location='cpu',weights_only=False)['model']
    parameter_error=max(float((v.detach().cpu()-final[k]).abs().max()) for k,v in policy.state_dict().items())
    status='PASS' if records and all(r['match'] for r in records) and parameter_error<2e-5 else 'FAIL'
    result=dict(status=status,episodes=len(records),records=records,final_parameter_max_error=parameter_error,device='cuda:0',physical_gpu=7,python=sys.executable,torch=torch.__version__,scope='independent sequential replay of saved PPO batches; checks reward-parts sum, bounds, GAE, KL, ratios and final model; raw physical scoring is separately replayed')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2));return int(status!='PASS')


if __name__=='__main__':raise SystemExit(main())
