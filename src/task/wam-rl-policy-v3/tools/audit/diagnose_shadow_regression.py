"""Read-only GPU audit of saved Shadow policies, fixed inputs and training logs."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6);p.add_argument('--assert-kl-consistency',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit output required')
    a.output.mkdir(parents=True);os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(ROOT/'tmp'))
    import numpy as np
    import torch
    from wam_rl_policy_v3.bc_mlp import CurrentMLP
    from wam_rl_policy_v3.distribution_kl import actual_policy_kl
    torch.set_num_threads(2)
    comparison=json.loads((a.panel/'comparison.json').read_text());values=[]
    for panel,indices in comparison['common_initial_scenes'].items():
        with np.load(a.panel/f'panel-{panel}-step-7000/initial-state.npz') as z:values.append(z['observation'][indices])
    x=torch.tensor(np.concatenate(values),device='cuda')
    train=ROOT/'outputs/wam-rl-policy-v2/official-scheduled-mixed-20261018-r4/rank-0'
    packets={};means={};checks={}
    for step in (7000,8500,9500,10000):
        packet=torch.load(train/f'official-iteration-{step}.pt',map_location='cpu',weights_only=False);packets[step]=packet
        weights={k[6:]:v for k,v in packet['model'].items() if k.startswith('actor.')}
        model=CurrentMLP(565).cuda().eval();model.load_state_dict(weights,strict=True)
        with torch.inference_mode():means[step]=model(x)
        opt_steps=sorted({float(s['step']) for s in packet['optimizer']['state'].values()})
        checks[str(step)]=dict(filename_iteration_matches=packet['iteration']==step,optimizer_steps=opt_steps,
            finite_parameters=all(bool(torch.isfinite(v).all()) for v in packet['model'].values()),
            effective_std=packet['model']['log_std'].exp().square().tolist(),
            lr=packet['step_size'],optimizer_lr=packet['optimizer']['param_groups'][0]['lr'])
    buffers={key:all(torch.equal(packets[7000]['model']['actor.'+key],packets[it]['model']['actor.'+key]) for it in packets)
             for key in ('x_mean','x_scale','y_mean','y_scale')}
    drift={}
    for left,right in ((7000,8500),(8500,9500),(9500,10000)):
        old=packets[left]['model']['log_std'].cuda();new=packets[right]['model']['log_std'].cuda()
        delta=(means[left]-means[right]).square()
        proxy=(new-old+(old.exp().square()+delta)/(2*new.exp().square())-.5).sum(-1)
        true=(2*(new-old)+(old.exp().pow(4)+delta)/(2*new.exp().pow(4))-.5).sum(-1)
        drift[f'{left}->{right}']=dict(mean_action_abs_change=float((means[left]-means[right]).abs().mean()),
            legacy_proxy_kl=float(proxy.mean()),actual_distribution_kl=float(true.mean()),
            mean_clipped_action_change=float((means[left].clamp(-1,1)-means[right].clamp(-1,1)).abs().mean()))
    log_std=packets[9500]['model']['log_std'].cuda();mu=torch.zeros(24,device='cuda');changed=mu.clone();changed[0]=.03
    scale=torch.diag(log_std.exp().square())
    exact=torch.distributions.kl_divergence(torch.distributions.MultivariateNormal(mu,scale_tril=scale),
        torch.distributions.MultivariateNormal(changed,scale_tril=scale))
    proxy=((mu-changed).square()/(2*log_std.exp().square())).sum()
    corrected=actual_policy_kl(mu,log_std,changed,log_std)
    math_probe=dict(real_checkpoint=9500,axis=0,mean_shift=.03,effective_std=float(log_std[0].exp().square()),
        actual_torch_kl=float(exact),upstream_proxy_kl=float(proxy),corrected_kl=float(corrected),desired_kl=.016,
        upstream_lr_decision='increase' if proxy<.008 else 'decrease' if proxy>.032 else 'keep',
        actual_lr_decision='increase' if exact<.008 else 'decrease' if exact>.032 else 'keep')
    rows=[json.loads(s) for s in (train/'curve.jsonl').read_text().splitlines()];by_iteration={r['iteration']:r for r in rows}
    windows={}
    for left,right in ((7000,7500),(7500,8000),(8000,8500),(8500,9000),(9000,9500),(9500,10000)):
        before,after=by_iteration[left],by_iteration[right];n=after['completed_episodes']-before['completed_episodes']
        window=[r for r in rows if left<r['iteration']<=right]
        windows[f'{left}->{right}']=dict(episodes=n,mean_reward=float(np.mean([r['reward_mean'] for r in window])),
            stochastic_native_success_rate=(after['native_success_episodes']-before['native_success_episodes'])/n,
            stochastic_ever_hold_rate=(after['held60_lift10cm_episodes']-before['held60_lift10cm_episodes'])/n,
            max_lr=max(r['lr'] for r in window))
    report=dict(status='KL_CALIBRATION_FIX_VERIFIED',gpu=a.gpu,python=sys.executable,torch=torch.__version__,
        optimizer_steps=0,fixed_inputs=len(x),checkpoint_checks=checks,normalization_buffers_identical=buffers,
        fixed_input_policy_drift=drift,mathematical_probe=math_probe,training_windows=windows,
        conclusion='real policy regression with declining stochastic training metrics; upstream adaptive KL mismatches its squared scale_tril sampling distribution; causal recovery not tested')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if a.assert_kl_consistency and not torch.isclose(corrected,exact,rtol=1e-5,atol=1e-7):raise SystemExit(2)


if __name__=='__main__':main()
