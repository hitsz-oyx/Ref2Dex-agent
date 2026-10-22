"""Audit saved on-policy latents/logprob, bounded intents, GAE and post-update KL."""
import argparse,json,os,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[4,5,6],default=5);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit output required')
    occupied=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if a.device=='cuda' and int(occupied)>64:raise RuntimeError('GPU occupied; queue audit')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu) if a.device=='cuda' else '',TMPDIR=str(ROOT/'tmp'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import torch
    from wam_rl_policy_v3.intent_residual import IntentResidualPolicy,RESIDUAL_SCALE
    torch.set_num_threads(2)
    manifest=json.loads((a.run/'ppo/manifest.json').read_text())
    if manifest['status']!='COMPLETED':raise ValueError('wait for both PPO rollouts')
    old_path=a.run/'bc/bc.pt';records=[]
    for row in manifest['runs']:
        folder=a.run/'ppo'/row['id'];new_path=a.run/'ppo'/('residual-'+row['id']+'.pt')
        with np.load(folder/'ppo-rollout.npz') as z:d={k:z[k].copy() for k in z.files}
        old=IntentResidualPolicy().to(a.device).eval();new=IntentResidualPolicy().to(a.device).eval()
        old.load_state_dict(torch.load(old_path,map_location='cpu',weights_only=False)['model']);new.load_state_dict(torch.load(new_path,map_location='cpu',weights_only=False)['model'])
        with torch.inference_mode():
            obs=torch.tensor(d['observations'],device=a.device);latent=torch.tensor(d['latents'],device=a.device);od,_=old.distribution(obs);nd,_=new.distribution(obs)
            replay=od.log_prob(latent).sum(-1).cpu().numpy();post=nd.log_prob(latent).sum(-1).cpu().numpy();exact=float(torch.distributions.kl_divergence(od,nd).sum(-1).mean())
        logp_error=float(np.abs(replay-d['old_logp']).max());residual=np.tanh(d['latents'])*RESIDUAL_SCALE
        bound_error=float(np.maximum(np.abs(residual)-RESIDUAL_SCALE,0).max())
        values=np.r_[d['values'],row['ppo_update']['bootstrap']];advantages=np.zeros(len(d['rewards']),np.float32);tail=0.
        for t in reversed(range(len(advantages))):
            tail=d['rewards'][t]+.96*values[t+1]-values[t]+.96*.95*tail;advantages[t]=tail
        mean_return=float((advantages+d['values']).astype(np.float32).mean());return_error=abs(mean_return-row['ppo_update']['target_return_mean'])
        match=bool(logp_error<1e-3 and bound_error==0 and return_error<1e-5)
        records.append(dict(id=row['id'],controls=row['controls'],onpolicy_logp_max_error=logp_error,bound_excess=bound_error,gae_mean_error=return_error,
            post_update_sampled_kl=float(np.mean(d['old_logp']-post)),post_update_exact_gaussian_kl=exact,optimizer_updates=row['ppo_update']['optimizer_updates'],match=match,
            maximum_residual_translation_coordinate_m=float(np.abs(residual[:,:3]).max()),maximum_residual_rotation_coordinate_rad=float(np.abs(residual[:,3:6]).max()),maximum_residual_finger_rad=float(np.abs(residual[:,6:]).max())))
        old_path=new_path
        del old,new
    result=dict(status='PASS' if records and all(r['match'] for r in records) else 'FAIL',gpu=a.gpu if a.device=='cuda' else None,device=a.device,cpu_reason='registered native jobs occupy permitted GPU4/5/6; small read-only policy audit, no GPU sharing' if a.device=='cpu' else None,python=sys.executable,torch=torch.__version__,records=records,
        scope=a.device+' replay of saved old/post policies and latent actions; KL threshold stops subsequent epochs, is not a hard bound on the first update; no task efficacy claim')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return int(result['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
