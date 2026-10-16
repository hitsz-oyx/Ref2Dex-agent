"""Bounded TRAIN-only value-regression qualification on recorded native episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from wam_rl_policy.bc_mlp import CurrentMLP
from wam_rl_policy.ppo import Critic, observation


def finite_returns(reward, next_done, gamma=.99):
    if reward.shape != next_done.shape:
        raise ValueError('Action-aligned rewards and next-state terminal flags required')
    target = torch.empty_like(reward)
    running = torch.zeros_like(reward[-1])
    for step in reversed(range(len(reward))):
        running = reward[step] + gamma * (~next_done[step]).float() * running
        target[step] = running
    return target


def write(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def metrics(prediction, target):
    error = target - prediction
    variance = target.double().var(unbiased=False)
    return dict(n=target.numel(), mse=float(error.square().mean()),
                mae=float(error.abs().mean()), prediction_std=float(prediction.std(unbiased=False)),
                target_std=float(target.std(unbiased=False)),
                explained_variance=float(1-error.double().var(unbiased=False)/variance) if variance > 1e-8 else None)


@torch.no_grad()
def evaluate(model, x, y, train_mask, terminal_mask):
    pred = torch.cat([model(chunk) for chunk in x.split(2048)])
    result = {}
    for name, selected in [('fit', train_mask), ('heldout_objects', ~train_mask)]:
        result[name] = metrics(pred[selected], y[selected])
        result[name]['interior'] = metrics(pred[selected & ~terminal_mask], y[selected & ~terminal_mask])
        result[name]['terminal7'] = metrics(pred[selected & terminal_mask], y[selected & terminal_mask])
    # Saturation is measured on both object groups at fixed uniformly spaced states.
    h = x[::max(1,len(x)//2048)]; saturation = []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Tanh):
            saturation.append(float((h.abs() > .99).float().mean()))
    result['tanh_saturation'] = saturation
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Existing free rank00 directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--allow-owned-gpu-share', action='store_true')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seconds', type=int, default=600)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.steps <= 2000 or not 1 <= a.seconds <= 600:
        raise ValueError('Fresh bounded output required')
    if not a.allow_owned_gpu_share and subprocess.check_output(
            ['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied; sharing requires explicit owned-task authorization')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    a.output.mkdir(parents=True)
    started = time.monotonic()
    source_manifest = json.loads((a.source/'manifest.json').read_text())
    objects = json.loads((a.source/'environment-objects.json').read_text())
    assert all(row['split'] == 'train' for row in objects)
    codes = sorted({row['object_code'] for row in objects})
    heldout_codes = codes[::4]
    fit_env = torch.tensor([row['object_code'] not in heldout_codes for row in objects], device=device)
    packet = torch.load(source_manifest['base_checkpoint'], map_location='cpu', weights_only=False)
    bc = CurrentMLP().to(device).eval().requires_grad_(False)
    bc.load_state_dict(packet['model'], strict=True)
    xs=[]; ys=[]; fits=[]; terminals=[]; ticks=[]; input_files=[]
    with torch.no_grad():
        for name in ['first-episode.npz', 'last-episode.npz']:
            file = a.source/name
            with np.load(file, allow_pickle=False) as data:
                tensor = lambda key: torch.as_tensor(data[key].copy(),device=device)
                reward=tensor('reward'); reset=tensor('reset').bool(); controls=reward.shape[0]
                assert controls == 199 and reward.shape[1] == len(objects)
                assert not reset[1:-1].any() and reset[-1].all()
                assert torch.equal(reward,tensor('native_reward'))  # fixed native-reward free source
                raw=tensor('observation')[:-1]; base=tensor('base')
                hidden=bc.net[:-1]((raw.clamp(-5,5)-bc.x_mean)/bc.x_scale)
                relative=tensor('goal')[:-1]-tensor('object')[:-1,:,:3]
                x=observation(raw.flatten(0,1),relative.flatten(0,1),base.flatten(0,1),hidden.flatten(0,1))
                y=finite_returns(reward,reset[1:])
                # Independent numpy recursion guards the frame/action alignment of the target.
                check=np.zeros_like(data['reward']); running=np.zeros(len(objects),dtype=np.float32)
                for step in range(controls-1,-1,-1):
                    running=data['reward'][step]+.99*(1-data['reset'][step+1])*running
                    check[step]=running
                np.testing.assert_allclose(y.cpu().numpy(),check,rtol=2e-5,atol=2e-5)
                step_ids=torch.arange(controls,device=device)[:,None].expand_as(reward).flatten()
                xs.append(x);ys.append(y.flatten());fits.append(fit_env.expand(controls,-1).flatten())
                terminals.append(step_ids>=controls-7);ticks.append(step_ids)
            input_files.append(str(file))
        x=torch.cat(xs);y=torch.cat(ys);fit=torch.cat(fits);terminal=torch.cat(terminals);tick=torch.cat(ticks)
    del bc,packet
    manifest=dict(status='RUNNING',run_id=a.output.name,pid=os.getpid(),
        experiment_id='P-20261016-shadow-scale-continuous',task='wam-rl-policy',
        git_commit=os.environ.get('REF2DEX_CODE_COMMIT'),input_files=input_files,
        source_manifest=str(a.source/'manifest.json'),base_checkpoint=source_manifest['base_checkpoint'],
        fit_objects=[code for code in codes if code not in heldout_codes],heldout_train_objects=heldout_codes,
        samples=len(y),fit_samples=int(fit.sum()),gamma=.99,finite_terminal_bootstrap=0,
        target='actual finite discounted return of recorded within-episode policy mixture; not GAE',
        network='1024/1024/1024/512',steps=a.steps,batch=1024,lr=3e-4,gradient_clip=1.,
        policy_seed=246,batch_seed=1000247,gpu=a.gpu,max_seconds=a.seconds,
        shared_with_owned_wam_none=a.allow_owned_gpu_share,
        scope='TRAIN-only fresh critic regression qualification; no fixed-policy V or robot improvement claim')
    write(a.output/'manifest.json',manifest)
    train_indices=fit.nonzero().flatten()
    mean=y[fit].mean()
    clock_mean=torch.stack([y[fit & (tick==t)].mean() for t in range(199)])
    baselines=dict(global_mean=metrics(torch.full_like(y[~fit],mean),y[~fit]),
        train_time_conditional_mean=metrics(clock_mean[tick[~fit]],y[~fit]))
    results={}
    for activation in ['tanh','elu']:
        for clock in ['zero','remaining']:
            name=activation+'-'+clock
            with torch.no_grad():
                clock_values=(199-tick).float()/199 if clock=='remaining' else torch.zeros_like(y)
                obs=torch.cat((x[:,:-24],clock_values[:,None],x[:,-24:]),dim=-1)
            torch.manual_seed(246)
            model=Critic(obs_dim=540,network_kind='large',activation=activation).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            generator=torch.Generator(device=device).manual_seed(1000247)
            history=[]
            for step in range(a.steps+1):
                if step in (0,100,300,a.steps):
                    report=evaluate(model,obs,y,fit,terminal)
                    history.append(dict(step=step,elapsed_s=time.monotonic()-started,**report))
                    print(json.dumps(dict(arm=name,step=step,heldout=report['heldout_objects'],fit=report['fit'])),flush=True)
                    write(a.output/(name+'.json'),history)
                if step==a.steps:break
                if time.monotonic()-started>a.seconds:raise TimeoutError('Bounded critic regression budget')
                ids=train_indices[torch.randint(len(train_indices),(1024,),device=device,generator=generator)]
                loss=(model(obs[ids])-y[ids]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic fit')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            torch.save(dict(critic=model.state_dict(),activation=activation,clock_kind=clock,steps=a.steps),a.output/(name+'.pt'))
            results[name]=history[-1]
            del model,optimizer,obs
    write(a.output/'summary.json',dict(status='UNCLEAR',baselines=baselines,arms=results,
        interpretation='Regression qualification only; improvement requires matched online PPO test'))
    manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    write(a.output/'manifest.json',manifest)


if __name__=='__main__':main()
