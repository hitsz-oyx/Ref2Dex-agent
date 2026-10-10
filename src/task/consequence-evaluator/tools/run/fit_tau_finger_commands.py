"""Fit legal applied finger commands of an owned tau-only controller on GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(TASK / 'tools/run')]
from probe_reference_tracking import gpu_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=279)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--extra', type=Path, help='One DAgger student-state packet; labels from frozen owned tau teacher')
    args = parser.parse_args()
    output = args.output.resolve(); output.relative_to(ROOT / 'outputs/consequence-evaluator')
    if output.exists() or not 1 <= args.epochs <= 1500 or not 30 <= args.seconds <= 180:
        raise ValueError('fresh bounded task-owned fit required')
    before = gpu_state(args.gpu)
    if before['utilization'] > 10 or before['used_mib'] > 512 or before['total_mib'] - before['used_mib'] < 20000:
        raise RuntimeError('GPU is busy: ' + repr(before))
    scratch = ROOT / 'tmp/tau-finger-fit'; scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    import torch
    from consequence_evaluator.reference_tracking import native_action
    from consequence_evaluator.tau_tracking import (
        SCHEMA, FINGERS, TauTracker, canonical_finger_targets, configure_finger_fit)
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0'); started = time.monotonic()
    teacher_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if teacher_payload['schema'] != SCHEMA or not teacher_payload['manifest']['tau_only']:
        raise ValueError('owned migrated tau-only checkpoint required')
    teacher = TauTracker().to(device); teacher.load_state_dict(teacher_payload['state_dict']); teacher.eval()
    policy = TauTracker().to(device); policy.load_state_dict(teacher.state_dict()); policy.eval()
    frozen = {key: val.clone() for key, val in policy.state_dict().items()}
    hashes = {str(args.checkpoint.resolve()): sha(args.checkpoint), str(Path(__file__).resolve()): sha(__file__),
              str(TASK / 'src/consequence_evaluator/tau_tracking.py'): sha(TASK / 'src/consequence_evaluator/tau_tracking.py')}
    xs = []; bases = []; labels = []; controllers = []
    for packet_index, packet in enumerate([args.source] + ([args.extra] if args.extra else [])):
        manifest = json.loads((packet / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETED' or not manifest['tau_only']:
            raise ValueError('completed tau-only source states required')
        geometry = Path(manifest['geometry_reference']) / 'geometry.npz'
        if sha(geometry) != manifest['input_sha256'][str(geometry.resolve())]:
            raise ValueError('geometry drift')
        hashes.update({str((packet / name).resolve()): sha(packet / name) for name in ('manifest.json', 'trajectory.npz')})
        hashes[str(geometry.resolve())] = sha(geometry)
        geometry_manifest = geometry.parent / 'manifest.json'
        hashes[str(geometry_manifest.resolve())] = sha(geometry_manifest)
        controllers.append(manifest['native_controller'])
        if controllers[-1] != controllers[0]:
            raise ValueError('source native controller mismatch')
        ids = np.flatnonzero(np.asarray(manifest['roles']) == 'tracker')
        if len(ids) != 16:
            raise ValueError('all sixteen tracker rows required; no outcome selection')
        with np.load(packet / 'trajectory.npz', allow_pickle=False) as arrays:
            x = torch.tensor(arrays['student_features'][:, ids].reshape(-1, 897), device=device)
            live_q = torch.tensor(arrays['dof_position'][:-1, ids].reshape(-1, 18), device=device)
            latent = torch.tensor(arrays['latent'][:, ids].reshape(-1, 12), device=device)
            applied = torch.tensor(arrays['pd_targets'][:, ids].reshape(-1, 18), device=device)
        with np.load(geometry) as arrays:
            q = torch.tensor(np.broadcast_to(arrays['q'][1:, None], (542, 16, 18)).copy().reshape(-1, 18), device=device)
        offset = torch.tensor(controllers[0]['offset'], device=device)
        scale = torch.tensor(controllers[0]['scale'], device=device)
        with torch.no_grad():
            expected_q = x[:, :18] + x[:, 861:879]
            if (expected_q[:, list(FINGERS)] - q[:, list(FINGERS)]).abs().max() > 1e-5:
                raise ValueError('recorded geometry error does not match tau-only input')
            if packet_index == 0:
                if (teacher.actor(x) - latent).abs().max() > 2e-5:
                    raise ValueError('source actions do not come from the declared frozen tau teacher')
                # The frame-local loaded joint posture is not the command label.
                target = canonical_finger_targets(applied, offset, scale)
            else:
                teacher_target = teacher.target(q, teacher.actor(x))
                teacher_native = native_action(teacher_target, live_q, offset, scale).clamp(-1, 1)
                teacher_pd = offset + scale * ((teacher_native + 1) / 2)
                target = canonical_finger_targets(teacher_pd, offset, scale)
            xs.append(x); bases.append(q); labels.append(target)
    x = torch.cat(xs); base = torch.cat(bases); label = torch.cat(labels)
    if not torch.isfinite(x).all() or not torch.isfinite(label).all():
        raise ValueError('finite state/command labels required')
    with torch.no_grad():
        hidden = policy.actor[:4](x)

    def score():
        with torch.no_grad():
            mean = policy.actor[-1](hidden)
            target = policy.target(base, mean)[:, list(FINGERS)]
            scaled = (target - offset[list(FINGERS)]) / scale[list(FINGERS)]
            clip = ((scaled < -5e-7) | (scaled > 1 + 5e-7)).any(-1)
            return dict(command_rmse_rad=float((target - label).square().mean().sqrt()),
                        clipping_rate=float(clip.float().mean()),
                        maximum_excess_rad=float(torch.maximum(-target, target - scale[list(FINGERS)]).clamp_min(0).max()))

    initial = score(); parameters = configure_finger_fit(policy)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    output.mkdir(parents=True)
    manifest = dict(schema=SCHEMA, status='RUNNING', task='consequence-evaluator', run_id=output.name,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    seed=args.seed, physical_gpu=args.gpu, gpu_before=before, budget_s=args.seconds,
                    input_sha256=hashes, tau_only=True, wrist_feedforward=True,
                    source=str(args.source.resolve()), extra=str(args.extra.resolve()) if args.extra else None,
                    inference_contract=teacher_payload['manifest']['inference_contract'],
                    training_contract='Only six finger output rows; all tracker rows, owned tau-only applied PD commands with .005 native-range interior margin; no loaded future q/object label; wrist/trunk/critic/logstd frozen',
                    initialization=str(args.checkpoint.resolve()), margin_fraction=.005, epochs=args.epochs,
                    native_controller=controllers[0], source_rows=len(x), initial=initial,
                    claim='Engineering command canonicalization; live behavior and original clip gate remain unmodified')
    write(output / 'manifest.json', manifest)
    try:
        for epoch in range(1, args.epochs + 1):
            if time.monotonic() - started > args.seconds - 5:
                raise TimeoutError('fixed epoch fit budget exceeded')
            mean = policy.actor[-1](hidden)
            target = policy.target(base, mean)[:, list(FINGERS)]
            ranges = scale[list(FINGERS)]
            scaled = (target - offset[list(FINGERS)]) / ranges
            fit = ((target - label) / ranges).square().mean()
            boundary = (torch.relu(.005 - scaled).square() + torch.relu(scaled - .995).square()).mean()
            loss = fit + 10 * boundary
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite finger fit')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if epoch == 1 or epoch % 100 == 0:
                gpu = gpu_state(args.gpu)
                prior_pids = {row.split(',')[0].strip() for row in before['processes'].splitlines()}
                for row in gpu['processes'].splitlines():
                    parts = row.split(',')
                    if len(parts) == 2 and parts[0].strip() not in prior_pids | {str(os.getpid())} and int(parts[1]) > 512:
                        raise RuntimeError('foreign GPU use; stop own fit')
                elapsed = time.monotonic() - started
                print(json.dumps(dict(epoch=epoch, loss=float(loss.detach()), elapsed_s=elapsed,
                                      eta_s=elapsed / epoch * (args.epochs - epoch), gpu=gpu)), flush=True)
        for key, val in policy.state_dict().items():
            if key in ('actor.4.weight', 'actor.4.bias'):
                if not torch.equal(val[:6], frozen[key][:6]):
                    raise ValueError('wrist output rows changed')
            elif not torch.equal(val, frozen[key]):
                raise ValueError('nonfinger parameter changed: ' + key)
        final = score()
        manifest.update(status='COMPLETED', elapsed_s=time.monotonic() - started, final=final,
                        frozen_wrist_trunk_critic_logstd=True)
        torch.save(dict(schema=SCHEMA, state_dict=policy.state_dict(), updates=args.epochs, manifest=manifest), output / 'final.pt')
        result = dict(status='COMPLETED', initial=initial, final=final, checkpoint_sha256=sha(output / 'final.pt'),
                      claim='Command fit only; no success claim before live rollout')
        write(output / 'result.json', result); print(json.dumps(result), flush=True)
    except Exception as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic() - started)
        raise
    finally:
        write(output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
