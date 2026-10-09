"""Bounded GT-geometry retargeter Probe; launch-isolated teacher-only data."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.retargeter import (SCHEMA, HandTrajectoryRetargeter,
    Standardizer, commanded_targets, relative_targets, absolute_targets,
    native_control, trajectory_input, ACTIVE_FINGERS, DOF_NAMES)
from consequence_evaluator.contracts import is_within

SOURCES = [
    'act-native-chunk-engineering-20261009-r2',
    'act-native-chunk-engineering-20261009-r3',
    'act-temporal-open_loop24-20261009-r1',
    'act-temporal-receding8-20261009-r1',
    'act-temporal-overlap8-20261009-r1',
    'act-temporal-temporal1-20261009-r1',
    'act-receding1-20261009-r1',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def teacher_data(path):
    with Path(path).open('rb') as stream:
        packet = pickle.load(stream)
    if (packet.get('engineering_only') is not True or packet.get('seed') != 282
            or packet.get('role_names', [None])[0] != 'reactive_teacher'
            or packet['role_outcomes']['reactive_teacher']['maximum_held_frames'] < 45
            or packet['replay_identity']['backend']['name'] != 'gpu_physx_gpu_pipeline'):
        raise ValueError('fixed sustained-hold teacher packet required')
    q = np.asarray(packet['dof_position'][:, 0], dtype='float32')
    dq = np.asarray(packet['dof_velocity'][:, 0], dtype='float32')
    hand = np.asarray(packet['hand_keypoints'][:, 0], dtype='float32')
    action = np.asarray(packet['actions'][:, 0], dtype='float32')
    if (q.shape != (543, 18) or dq.shape != q.shape or hand.shape != (543, 11, 3)
            or action.shape != (542, 18) or np.asarray(packet['done'])[:-1, 0].any()
            or not all(np.isfinite(x).all() for x in (q, dq, hand, action))):
        raise ValueError('complete finite no-reset teacher trajectory required')
    target = commanded_targets(q[:-1], action)
    ticks = np.arange(0, 519, 4)
    future = hand[ticks[:, None]+np.arange(1, 25)]
    desired = target[ticks[:, None]+np.arange(24)]
    y = relative_targets(q[ticks], desired)
    inverse = absolute_targets(q[ticks], y)
    live_q = q[ticks[:, None]+np.arange(24)]
    rebuilt = commanded_targets(live_q, native_control(inverse, live_q))
    # Euler branches are representation-equivalent; these traces must also
    # roundtrip the actual native commanded coordinates before fitting.
    error = float(np.max(np.abs(rebuilt-desired)))
    if error > 2e-5:
        raise ValueError('native target roundtrip failed: %s' % error)
    return dict(hand=trajectory_input(hand[ticks], future),
                state=np.concatenate((q[ticks], dq[ticks]), -1), target=y,
                ticks=ticks, roundtrip_max_abs=error,
                identity=packet['replay_identity'], source=str(Path(path).resolve()),
                sha256=digest(path), episode_outcome=packet['role_outcomes']['reactive_teacher'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=292)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3000 or not 30 <= args.seconds <= 300:
        parser.error('bounded <=3000 steps/300 seconds required')
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / 'outputs/consequence-evaluator'):
        raise ValueError('fresh task-owned output required')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda:%d' % args.gpu)
    rows = [teacher_data(ROOT / 'outputs/consequence-evaluator' / name / 'act.pkl') for name in SOURCES]
    for row in rows:
        if row['identity'] != rows[0]['identity']:
            raise ValueError('teacher controller/backend/environment identity mismatch')
    # Freeze complete-launch splits before any statistics or optimization.
    split = dict(train=rows[:5], val=rows[5:6], test=rows[6:])
    arrays = {part: {key: np.concatenate([row[key] for row in group])
                    for key in ('hand', 'state', 'target')} for part, group in split.items()}
    stats = {key: Standardizer.fit(arrays['train'][key]) for key in ('hand', 'state', 'target')}
    data = {part: {key: torch.as_tensor(stats[key].encode(value), device=device)
                   for key, value in group.items()} for part, group in arrays.items()}
    model = HandTrajectoryRetargeter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    manifest = dict(schema=SCHEMA, engineering_only=True, data_scope='sustained_hold_not_full_task_success',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        seed=args.seed, gpu=args.gpu, steps_cap=args.steps, seconds_cap=args.seconds,
        splits={part: [{k: row[k] for k in ('source', 'sha256', 'roundtrip_max_abs', 'episode_outcome')}
                      for row in group] for part, group in split.items()},
        window_counts={part: len(group['hand']) for part, group in arrays.items()},
        input_contract='GT world-frame future hand minus measured current hand; current q18+dq18 only',
        target_contract='recorded commanded PD targets relative to query q; xyz, local rotvec, six active fingers',
        horizon=24, first_future_offset=1, action_offset=0, stride=4, padding=False,
        controller_identity=rows[0]['identity'], statistics_fit_split='train')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    start = time.monotonic()
    best = float('inf')
    history = []
    for step in range(1, args.steps+1):
        model.train()
        index = torch.as_tensor(rng.integers(len(data['train']['hand']), size=128), device=device)
        batch = data['train']
        pred = model(batch['hand'][index], batch['state'][index])
        loss = (pred-batch['target'][index]).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        elapsed = time.monotonic()-start
        if step == 1 or step % 200 == 0 or step == args.steps or elapsed > args.seconds:
            model.eval()
            with torch.no_grad():
                val = data['val']
                val_loss = float((model(val['hand'], val['state'])-val['target']).abs().mean())
            record = dict(step=step, train_l1=float(loss), val_l1=val_loss,
                          elapsed_s=elapsed, eta_s=max(0, (args.steps-step)*elapsed/step))
            print(json.dumps(record), flush=True)
            history.append(record)
            if val_loss < best:
                best = val_loss
                torch.save(dict(schema=SCHEMA, width=model.width, state_dict=model.state_dict(),
                    statistics={key: value.as_dict() for key, value in stats.items()},
                    best_step=step, best_val_l1=best, manifest=manifest, dof_names=DOF_NAMES,
                    active_fingers=ACTIVE_FINGERS.tolist()), output / 'best.pt')
            if elapsed > args.seconds:
                break
    payload = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(payload['state_dict']); model.eval()
    metrics = {}
    with torch.no_grad():
        for part in ('train', 'val', 'test'):
            batch = data[part]
            value = model(batch['hand'], batch['state']).cpu().numpy()
            prediction = stats['target'].decode(value)
            delta = prediction-arrays[part]['target']
            # State-only/zero-displacement diagnostics use the same frozen model.
            zero = torch.as_tensor(stats['hand'].encode(np.zeros_like(arrays[part]['hand'])), device=device)
            zero_pred = stats['target'].decode(model(zero, batch['state']).cpu().numpy())
            metrics[part] = dict(relative_position_rmse_mm=float(np.sqrt(np.mean(delta[..., :3]**2))*1000),
                local_rotvec_rmse_rad=float(np.sqrt(np.mean(delta[..., 3:6]**2))),
                active_finger_rmse_rad=float(np.sqrt(np.mean(delta[..., 6:]**2))),
                normalized_l1=float(np.mean(np.abs(value-batch['target'].cpu().numpy()))),
                zero_hand_normalized_l1=float(np.mean(np.abs(stats['target'].encode(zero_pred)-batch['target'].cpu().numpy()))))
    result = dict(status='COMPLETED', best_step=payload['best_step'], elapsed_s=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), metrics=metrics, history=history,
        checkpoint_sha256=digest(output / 'best.pt'), claim='offline fit only; real execution gate pending')
    (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main()
