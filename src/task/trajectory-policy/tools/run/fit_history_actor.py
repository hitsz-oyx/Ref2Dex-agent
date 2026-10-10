"""Bounded pure-history BC initializer, with all source rows retained."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/consequence-evaluator/src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'geometry', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task output required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    util, memory = map(int, before.split(','))
    if util > 10 or memory > 512:
        raise ValueError('GPU not idle: '+before)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from trajectory_policy.actor import HistoryActor, SCHEMA
    from trajectory_policy.history import measured_history, SCHEMA as HISTORY_SCHEMA
    from trajectory_policy.dense_decoder import DenseDecoder
    from trajectory_policy.inputs import load_geometry
    from consequence_evaluator.data import sha
    torch.set_num_threads(2)
    torch.manual_seed(295)
    torch.backends.cuda.matmul.allow_tf32 = False
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    geometry, initial, _, hashes = load_geometry(args.geometry, args.reference, urdf)
    source_manifest = json.loads((args.evaluation/'manifest.json').read_text())
    if source_manifest['status'] != 'COMPLETED':
        raise ValueError('completed source native wave required')
    for path, digest in source_manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('source wave drift: '+path)
    # Explicit allowlist: neither native observations, force nor future actual q
    # or actual object trajectories are extracted as target labels.
    with np.load(args.evaluation/'trajectory.npz', allow_pickle=False) as stream:
        data = {key: stream[key].copy() for key in ('q', 'dq', 'hand', 'obj', 'velocity')}
    if any(len(value) != 543 or value.shape[1] != 16 for value in data.values()):
        raise ValueError('complete 16-row measured source wave required')
    sources = list((TASK/'src/trajectory_policy').glob('*.py'))+[Path(__file__),
        args.evaluation/'manifest.json', args.evaluation/'trajectory.npz',
        ROOT/'src/task/consequence-evaluator/src/consequence_evaluator/proposal_history.py']
    sources += [ROOT/'src/task/consequence-evaluator/src/consequence_evaluator'/(name+'.py') for name in
        ('reference_tracking', 'tau_tracking', 'reset_kinematics', 'retargeter', 'contracts', 'reference_motion')]
    hashes.update({str(path.resolve()): sha(path) for path in sources})
    record = dict(status='RUNNING', schema=SCHEMA, history_schema=HISTORY_SCHEMA,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256=hashes, physical_gpu=args.gpu, gpu_before=before, seed=295,
        updates=2500, batch_size=512, width=512, optimizer='Adam', learning_rate=.0003,
        offline_future_labels='Only hand-derived geometry q; never actual future robot/object state.',
        deployment='H328 only; no clock/phase/reference/tactile or reference actor.')
    args.output.mkdir(parents=True)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('manifest.json', record)
    started = time.monotonic()
    def deadline(signum, frame):
        raise TimeoutError('actor initializer exceeded240s')
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(240)
    try:
        roles = np.asarray(source_manifest['roles'])
        validation_rows = np.asarray([np.flatnonzero(roles == role)[-1] for role in sorted(set(roles))])
        train_rows = np.setdiff1d(np.arange(16), validation_rows)
        histories, labels = [], []
        for tick in range(542):
            past = np.maximum(tick+np.arange(-3, 1), 0)
            histories.append(measured_history(*(data[key][past] for key in ('obj', 'hand', 'q', 'dq', 'velocity'))))
            future = geometry['q'][np.minimum(tick+np.arange(1, 25), 542)]
            labels.append(DenseDecoder.encode(np.broadcast_to(future, (16, 24, 18)), data['q'][tick], data['obj'][tick]))
        h, c = np.stack(histories), np.stack(labels)
        if not np.isfinite(c).all():
            raise ValueError('finite hand-derived labels required')
        train_h, train_c = (value[:, train_rows].reshape(-1, value.shape[-1]) for value in (h, c))
        val_h, val_c = (value[:, validation_rows].reshape(-1, value.shape[-1]) for value in (h, c))
        actor = HistoryActor(512).cuda()
        with torch.no_grad():
            for key, value in dict(history_mean=train_h.mean(0), history_scale=train_h.std(0).clip(.001),
                                   action_mean=train_c.mean(0), action_scale=train_c.std(0).clip(.1)).items():
                getattr(actor, key).copy_(torch.as_tensor(value, device='cuda'))
        actor.log_std.requires_grad_(False)
        x, y, vx, vy = [torch.as_tensor(value, device='cuda') for value in (train_h, train_c, val_h, val_c)]
        optimizer = torch.optim.Adam(actor.mean_net.parameters(), lr=.0003)
        weights = torch.as_tensor([1.]*8+[.25]*16, device='cuda')[None, :, None]
        def loss(prediction, target):
            error = (prediction-target).reshape(-1, 24, 12)
            delta = torch.diff(error[:, :, :6], dim=1)*30
            velocity = torch.cat((delta[:, :1], (delta[:, :-1]+delta[:, 1:])/2, delta[:, -1:]), dim=1)
            ff = error[:, :8, :6]+.1*velocity[:, :8]
            return ((error*weights)**2).mean()+.1*(ff**2).mean()
        with torch.no_grad():
            mean_baseline = float(loss(actor.action_mean.expand_as(vy), vy))
        best, best_step, best_state = float('inf'), None, None
        fit_started = time.monotonic()
        for step in range(1, 2501):
            ids = torch.randint(len(x), (512,), device='cuda')
            objective = loss(actor(x[ids]), y[ids])
            if not torch.isfinite(objective):
                raise FloatingPointError('nonfinite BC objective')
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(actor.mean_net.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % 250 == 0:
                with torch.no_grad():
                    val = float(loss(actor(vx), vy))
                if val < best:
                    best, best_step = val, step
                    best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
                state = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                monitor = dict(update=step, train_loss=float(objective), validation_loss=val,
                    elapsed_s=time.monotonic()-started, eta_s=(time.monotonic()-fit_started)/step*(2500-step), gpu=state)
                print(json.dumps(monitor), flush=True)
                with (args.output/'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(monitor)+'\n')
        actor.load_state_dict(best_state)
        actor.eval()
        packet = dict(schema=SCHEMA, history_schema=HISTORY_SCHEMA, width=512, model=best_state,
            selected_update=best_step, train_rows=train_rows.tolist(), validation_rows=validation_rows.tolist(),
            input_sha256=hashes, git_commit=record['git_commit'])
        torch.save(packet, args.output/'best.pt')
        ticks = np.arange(0, 542, 8)
        with torch.no_grad():
            predicted = actor(torch.as_tensor(h[ticks][:, validation_rows].reshape(-1, 328), device='cuda')).cpu().numpy()
            decoder = DenseDecoder(urdf, 'cuda')
            current = data['q'][ticks][:, validation_rows].reshape(-1, 18)
            obj = data['obj'][ticks][:, validation_rows].reshape(-1, 4, 4)
            decoded = decoder.decode(predicted, current, obj, 1/30)
        q, hand, velocity = (decoded[key].cpu().numpy() for key in ('q', 'hand', 'velocity'))
        index = np.minimum(ticks[:, None]+np.arange(1, 25), 542)
        target_hand = np.repeat(geometry['fitted_points'][index][:, None], 4, axis=1).reshape(-1, 24, 11, 3)
        error = np.linalg.norm(hand-target_hand, axis=-1)
        np.savez_compressed(args.output/'coverage.npz', tick=ticks, rows=validation_rows,
            history=h[ticks][:, validation_rows], c=predicted, q=q, hand=hand, velocity=velocity)
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('source/input drift during fit')
        result = dict(status='UNCLEAR', neural_training_completed=True, native_evaluation_required=True,
            best_update=best_step, validation_loss=best, train_mean_validation_loss=mean_baseline,
            improvement_over_mean=1-best/mean_baseline, train_rows=train_rows.tolist(), validation_rows=validation_rows.tolist(),
            train_samples=len(train_h), validation_samples=len(val_h), prefix8_hand_3d_rms_mm=float(np.sqrt((error[:, :8]**2).mean())*1000),
            prefix8_palm_max_mm=float(error[:, :8, 0].max()*1000), elapsed_s=time.monotonic()-started,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            claim='Supervised same-motion row holdout initializer only; not policy success, RL or Cm benefit.')
        write('result.json', result)
        record.update(status='COMPLETED', checkpoint_sha256=sha(args.output/'best.pt'))
        write('manifest.json', record)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        write('manifest.json', record)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    main()
