"""Clone Scheduled686, fit one recorded trajectory only; default requires dry-run/execute choice.

Dry-run performs one CPU forward with no optimizer, simulator, or GPU allocation.
Real fitting requires a separately recorded, explicitly contracted single-trajectory
NPZ (see docs/audit/v6-retrain-calibration-20261022.md). No RL or WAM training.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = ROOT/'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/latest.pt'
DATA_SCHEMA = 'ref2dex.v6-single-trajectory-calibration.v1'


def validate_data(observation, action, tick, meta):
    n = len(observation)
    if observation.shape != (n, 686) or action.shape != (n, 12) or not 1 <= n <= 301:
        raise ValueError('one bounded trajectory (1..301 rows): observation[N,686], action[N,12] required')
    if not np.isfinite(observation).all() or not np.isfinite(action).all() or (np.abs(action) > 1).any():
        raise ValueError('finite observation and executed canonical12 commands in [-1,1] required')
    if tick.shape != (n,) or tick.dtype.kind not in 'iu' or tick[0] != 0 or not np.all(np.diff(tick) == 1):
        raise ValueError('one contiguous episode beginning at tick zero required')
    expected = dict(schema=DATA_SCHEMA, native_hz=30, tau_frame='world',
                    tau_representation='displacement_from_current_hand', hand_order='native_inspire11')
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('explicit schema/30Hz/world displacement/native11 metadata required')
    if not isinstance(meta.get('source_sequence'), str) or not meta['source_sequence'] or not meta.get('task_semantics') or not meta.get('source_recording'):
        raise ValueError('one source sequence, task semantics and source recording provenance required')
    if meta.get('tau_kind') not in ('gt', 'wam'):
        raise ValueError('declare gt or wam tau; never substitute channels after fitting')
    if meta['tau_kind'] == 'wam' and not meta.get('wam_checkpoint'):
        raise ValueError('WAM channel requires checkpoint identity')
    if meta['tau_kind'] == 'gt' and n < 12:
        raise ValueError('GT trajectory needs H4 and eight observed future frames')
    if not np.allclose(meta.get('tau_times_s', []), np.arange(1, 9)/30):
        raise ValueError('tau times must be 1/30 through 8/30 seconds')
    valid = observation[:, -1]
    if not np.isin(valid, [0, 1]).all() or valid[:3].any() or (meta['tau_kind'] == 'gt' and valid[-8:].any()):
        raise ValueError('explicit tau validity; H4 startup and missing GT future must be invalid')
    if observation[valid == 0, 421:685].any():
        raise ValueError('invalid tau must be zero, with validity false')
    phase = observation[:, 418:421]
    if not np.isin(phase, [0, 1]).all() or not np.all(phase.sum(-1) == 1):
        raise ValueError('one declared execution phase in task10 required')
    if not (valid == 1).any():
        raise ValueError('calibration must include the chosen valid tau channel')


def gpu_snapshot(gpu):
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    index, memory, utilization = map(int, raw.strip().split(','))
    return dict(index=index, memory_mib=memory, utilization_percent=utilization)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute-training', action='store_true', help='Run only after Hermes confirmation')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu', type=int, choices=(4, 5, 6), default=5)
    p.add_argument('--updates', type=int, default=2000)
    p.add_argument('--seconds', type=float, default=1800.)
    p.add_argument('--seed', type=int, default=196)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if not 1 <= a.updates <= 2000 or not 0 < a.seconds <= 1800:
        raise ValueError('hard limit: 2000 updates or 1800 seconds, whichever comes first')
    if not a.dry_run and (a.data is None or a.output is None):
        raise ValueError('real fitting requires --data and a fresh --output')
    telemetry = None
    if a.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        telemetry = gpu_snapshot(a.gpu)
        if telemetry['memory_mib'] > 128 or telemetry['utilization_percent'] > 5:
            raise ValueError('selected GPU is occupied; leave it untouched')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    sys.path.insert(0, str(ROOT/'src/task/wam-rl-policy-v3/src'))
    import torch
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    packet = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    baseline = MeasuredInspirePolicy(with_tau=True).eval().requires_grad_(False)
    baseline.load_state_dict(packet['model'], strict=True)
    model = copy.deepcopy(baseline)
    assert all(x.data_ptr() != y.data_ptr() for x, y in zip(model.parameters(), baseline.parameters()))
    if a.data:
        with np.load(a.data, allow_pickle=False) as z:
            x, y, tick = z['observation'].copy(), z['action'].copy(), z['tick'].copy()
            meta = json.loads(str(z['metadata_json'].item()))
        validate_data(x, y, tick, meta)
    else:
        # Synthetic tensor is an interface fixture only, never a fit target.
        x, y = np.zeros((1, 686), np.float32), np.zeros((1, 12), np.float32)
        x[:, 418] = 1
        meta = dict(source_sequence='SYNTHETIC_DRY_RUN_ONLY')
    report = dict(status='DRY_RUN_PASS' if a.dry_run else 'STARTING', checkpoint=str(a.checkpoint.resolve()),
        model='MeasuredInspirePolicy(with_tau=True)', actor_hidden_dims=list(model.actor.hidden_dims),
        input_dim=686, action_dim=12, source=meta, samples=len(x), fit_updates=0,
        optimizer_created=False, simulator_started=False, gpu_allocated=False,
        update_limit=a.updates, seconds_limit=a.seconds, gpu_before=telemetry,
        python=sys.version.split()[0], python_executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        conda_env=os.environ.get('CONDA_DEFAULT_ENV'), seed=a.seed,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='one seen trajectory; action fitting only; no physical success/generalization claim')
    if a.dry_run:
        with torch.inference_mode():
            prediction = model.actor(torch.as_tensor(x[:1], dtype=torch.float32))
        assert prediction.shape == (1, 12) and torch.isfinite(prediction).all()
        report['baseline_copy_max_error'] = max(float((v-baseline.state_dict()[k]).abs().max()) for k, v in model.state_dict().items())
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    out = a.output.resolve()
    if ROOT/'outputs/ptv3-shared-wam-v6' not in out.parents:
        raise ValueError('real artifacts must be under outputs/ptv3-shared-wam-v6/<fresh-run>')
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise ValueError('free space below 20GiB')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['TMPDIR'] = str(out)
    device = torch.device('cuda:0')
    model = model.to(device)
    model.actor.requires_grad_(True).train()
    train_x = torch.as_tensor(x, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
    report.update(optimizer_created=True, gpu_allocated=True, status='FITTING', curve=[])
    def save_report():
        (out/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def rmse():
        # Bounded chunks even if the input trajectory is long.
        with torch.no_grad():
            return float(torch.cat([(model.actor(b)-t).square().mean(-1) for b, t in
                zip(train_x.split(64), train_y.split(64))]).mean().sqrt())
    report['initial_action_rmse'] = rmse()
    started = time.monotonic(); last_monitor = started
    save_report()
    try:
        for step in range(a.updates):
            if time.monotonic()-started >= a.seconds:
                break
            ids = torch.randint(len(x), (64,), device=device)
            loss = ((model.actor(train_x[ids])-train_y[ids])/model.actor.y_scale.clamp(min=.05)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite calibration loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            report['fit_updates'] = step+1
            elapsed = time.monotonic()-started
            if step == 0 or elapsed >= a.seconds or time.monotonic()-last_monitor >= 10 or (step+1) % 100 == 0:
                row = dict(update=step+1, loss=float(loss), elapsed_s=elapsed,
                    eta_s=min((a.updates-step-1)*elapsed/(step+1), max(0, a.seconds-elapsed)),
                    peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20, gpu=gpu_snapshot(a.gpu))
                report['curve'].append(row); last_monitor = time.monotonic()
                print(json.dumps(row), flush=True); save_report()
                if shutil.disk_usage(ROOT).free < 20*2**30:
                    report['stop_reason'] = 'free_space'; break
        report.update(status='COMPLETED', fit_seconds=time.monotonic()-started,
            final_action_rmse=rmse(), stop_reason=report.get('stop_reason', 'updates' if report['fit_updates'] == a.updates else 'seconds'))
        # Only one final model; retain the original checkpoint and its normalizers.
        with (out/'calibrated.pt').open('xb') as f:
            torch.save(dict(schema=DATA_SCHEMA, model=model.cpu().state_dict(), source=meta,
                base_checkpoint=str(a.checkpoint.resolve()), updates=report['fit_updates']), f)
        if (out/'calibrated.pt').stat().st_size > 2*2**30:
            raise ValueError('artifact exceeds registered 2GiB budget')
    except BaseException as error:
        report.update(status='FAILED', error=repr(error)); raise
    finally:
        save_report()
    return report


if __name__ == '__main__':
    main()
