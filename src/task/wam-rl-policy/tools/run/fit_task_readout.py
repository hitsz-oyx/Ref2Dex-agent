"""Fixed-budget, object-held-out qualification of task-conditioned WAM readouts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from wam_rl_policy.task_readout import TaskReadout

ARMS = ['raw-query-aux', 'wam-mean-aux', 'wam-query-return', 'wam-query-aux', 'wam-query-forecast-aux']


def write(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def normalize(x, fit, mask=None):
    selected = x[fit] if mask is None else x[fit][mask[fit]]
    mean = selected.float().mean(0)
    scale = selected.float().std(0, unbiased=False).clamp_min(.01)
    result = (x.float() - mean) / scale
    if mask is not None:
        result = torch.where(mask[..., None], result, 0)
    return result, mean, scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.steps <= 1000:
        raise ValueError('Fresh output and bounded steps required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                               '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    a.output.mkdir(parents=True)
    started = time.monotonic()
    packet = torch.load(a.features, map_location='cpu', weights_only=False)
    d = {k: v.cuda() for k, v in packet.items() if torch.is_tensor(v)}
    fit = d['fit']; fit_ids = fit.nonzero().flatten()
    assert fit.any() and (~fit).any()
    state, sm, ss = normalize(d['state'], fit)
    targets, ym, ys = normalize(d['targets'], fit)
    forecast, fm, fs = normalize(d['forecast'], fit)
    raw, rm, rs = normalize(d['raw'], fit, d['raw_valid'])
    # Per-token layer norm follows existing PPO feature scaling; it uses no
    # held-out population statistics. Mean retains three separate types.
    wam = F.layer_norm(d['wam'].float(), (d['wam'].shape[-1],))
    mean = F.layer_norm(d['mean'].reshape(-1, 3, 768), (768,)).flatten(1)
    object_codes = [row['object_code'] for row in packet['objects']]
    codes = sorted(set(object_codes))
    object_index = torch.tensor([codes.index(object_codes[i]) for i in d['object_id'].tolist()], device='cuda')
    manifest = dict(status='RUNNING', experiment_id='P-20261016-wam-task-readout',
        run_id=a.output.parent.name, features=str(a.features.resolve()), gpu=a.gpu, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        arms=a.arms, samples=len(fit), fit_samples=int(fit.sum()), heldout_samples=int((~fit).sum()),
        steps=a.steps, batch=128, lr=3e-4, init_seed=246, batch_seed=247, aux_weight=.2,
        token_normalization='WAM per token LN; raw TRAIN-valid-token per channel moments',
        target_normalization='TRAIN object per dimension mean/std, min std .01', max_seconds=900)
    write(a.output / 'manifest.json', manifest)

    def score(pred):
        result = {}
        for name, chosen in [('fit', fit), ('heldout_objects', ~fit)]:
            error = pred[chosen] - d['targets'][chosen]
            groups = {}
            for i in object_index[chosen].unique().tolist():
                e = pred[chosen & (object_index == i)] - d['targets'][chosen & (object_index == i)]
                groups[codes[i]] = dict(return_mse=float(e[:, 0].square().mean()),
                    physics_normalized_mse=float((e[:, 1:] / ys[1:]).square().mean()))
            result[name] = dict(return_mse=float(error[:, 0].square().mean()),
                return_macro_mse=sum(v['return_mse'] for v in groups.values())/len(groups),
                physics_normalized_mse=float((error[:, 1:] / ys[1:]).square().mean()),
                physics_mae_m=error[:, 1:].abs().mean(0).tolist(), per_object=groups)
        return result

    results = {}
    for arm in a.arms:
        torch.manual_seed(246)
        token = raw if arm.startswith('raw') else mean if 'mean' in arm else wam
        valid = d['raw_valid'] if arm.startswith('raw') else d['wam_valid']
        use_forecast = 'forecast' in arm
        use_aux = arm.endswith('aux')
        model = TaskReadout(state.shape[-1], token.shape[-1], mean='mean' in arm,
                            forecast_width=forecast.shape[-1] if use_forecast else 0).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator(device='cuda').manual_seed(247)
        history = []
        arm_started = time.monotonic()
        for step in range(1, a.steps + 1):
            model.train()
            ix = fit_ids[torch.randint(len(fit_ids), (128,), generator=generator, device='cuda')]
            value, physics = model(state[ix], token[ix], valid[ix], forecast[ix] if use_forecast else None)
            vl = F.mse_loss(value, targets[ix, 0]); pl = F.mse_loss(physics, targets[ix, 1:])
            loss = vl + (.2 * pl if use_aux else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError(arm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 250 == 0:
                model.eval()
                with torch.no_grad():
                    outputs = []
                    for ids in torch.arange(len(fit), device='cuda').split(128):
                        v, ph = model(state[ids], token[ids], valid[ids], forecast[ids] if use_forecast else None)
                        outputs.append(torch.cat((v[:, None], ph), -1) * ys + ym)
                    metrics = score(torch.cat(outputs))
                row = dict(step=step, elapsed=time.monotonic()-arm_started, value_loss=float(vl),
                           physics_loss=float(pl), metrics=metrics)
                history.append(row)
                progress = dict(arm=arm, step=step, elapsed=row['elapsed'],
                    eta_seconds=row['elapsed']/step*(a.steps-step),
                    fit_mse=metrics['fit']['return_macro_mse'],
                    heldout_mse=metrics['heldout_objects']['return_macro_mse'],
                    resource=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                        '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
                print(json.dumps(progress), flush=True)
                with (a.output/'resources.jsonl').open('a') as f:
                    f.write(json.dumps(progress)+'\n')
            if time.monotonic()-started > 900:
                raise TimeoutError('Readout fitting budget exceeded')
        # Always measure the final endpoint, including a short engineering smoke.
        model.eval()
        with torch.no_grad():
            outputs=[]
            for ids in torch.arange(len(fit),device='cuda').split(128):
                v,ph=model(state[ids],token[ids],valid[ids],forecast[ids] if use_forecast else None)
                outputs.append(torch.cat((v[:,None],ph),-1)*ys+ym)
            pred=torch.cat(outputs)
        results[arm]=dict(metrics=score(pred), parameters=sum(p.numel() for p in model.parameters()),
                          seconds=time.monotonic()-arm_started, history=history)
        torch.save(dict(model=model.cpu().state_dict(), arm=arm,
                        normalization={k:v.cpu() for k,v in dict(state_mean=sm,state_scale=ss,
                        target_mean=ym,target_scale=ys,raw_mean=rm,raw_scale=rs,
                        forecast_mean=fm,forecast_scale=fs).items()},
                        prediction=pred.cpu()),a.output/(arm+'.pt'))
        write(a.output/'results.json',results)
        del optimizer,model
    # The clock baseline is fitted only on TRAIN objects at each observed tick.
    baseline = torch.empty_like(d['targets'])
    for tick in d['tick'].unique():
        mask=d['tick']==tick
        baseline[mask]=d['targets'][mask & fit].mean(0)
    manifest.update(status='COMPLETED',seconds=time.monotonic()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),clock_baseline=score(baseline))
    write(a.output/'manifest.json',manifest)
    print(json.dumps(dict(status='COMPLETED',seconds=manifest['seconds'])),flush=True)


if __name__ == '__main__':
    main()
