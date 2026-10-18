"""Bounded UniDex BC using actual frozen-WAM conditions at teacher states.

Record IDs are explicit; train/dev are disjoint raw episodes. This is a
within-TRAIN development split, not the global held-out object evaluation.
No oracle future is fed to the scheduler, WAM, or executor.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from wam_rl_policy.goal_scheduler import OrderedGoalScheduler
from wam_rl_policy.scheduled_executor import ScheduledV4ExecutorInputs
from wam_rl_policy.wam_v4_runtime import V4Runtime
from train_closed_loop_student import build_model


def records_for_split(manifest, train_ids, dev_ids):
    if not train_ids or not dev_ids or set(train_ids) & set(dev_ids):
        raise ValueError('nonempty disjoint record IDs required')
    groups = []
    for ids in (train_ids, dev_ids):
        rows = [manifest['records'][i] for i in ids]
        if any(r['split'] != 'train' for r in rows):
            raise ValueError('use TRAIN records only; this is internal development')
        groups.append({r['raw_episode'] for r in rows})
    if groups[0] & groups[1]:
        raise ValueError('raw episodes must be disjoint between BC train and dev')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'wam', 'dense_manifest', 'output'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    p.add_argument('--train-records', type=int, nargs='+', required=True)
    p.add_argument('--dev-records', type=int, nargs='+', required=True)
    p.add_argument('--rows-per-record', type=int, default=32)
    p.add_argument('--fit-steps', type=int, default=1000)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('fresh output required')
    if not (1 <= args.rows_per_record <= 256 and 0 <= args.fit_steps <= 3000
            and 0 < args.seconds <= 1800 and len(args.train_records+args.dev_records) <= 16):
        raise ValueError('bounded records, rows, updates and runtime required')
    if shutil.disk_usage(args.output.parent).free < 20*2**30:
        raise RuntimeError('free space below20GiB')
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for real WAM inference and BC')
    manifest = json.loads((args.cache/'manifest.json').read_text())
    records_for_split(manifest, args.train_records, args.dev_records)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    features, labels, groups, train_mask, ticks_out = [], [], [], [], []
    for record_id in args.train_records+args.dev_records:
        if time.monotonic()-started > args.seconds:
            raise TimeoutError('preparation budget exhausted')
        row = manifest['records'][record_id]
        with np.load(args.cache/row['path'], allow_pickle=False) as values:
            data = {k:values[k] for k in values.files}
        poses = data['poses']
        if not np.array_equal(data['observation'][1:, 167:191], data['action']):
            raise ValueError('next observation previous-action channels must match canonical labels')
        if not np.allclose(data['goal'], data['goal'][0], atol=1e-7):
            raise ValueError('task goal must be decision-known and fixed within episode')
        objects = np.concatenate((poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()), -1)
        runtime = V4Runtime(args.wam, [row], args.cache, dense_manifest=args.dense_manifest,
                            device=f'cuda:{args.gpu}')
        target = poses[0].copy(); target[:3, 3] = data['goal'][0]
        scheduler = OrderedGoalScheduler([poses[0], target], [0., 2.], runtime.geometry[0][2],
                                         interaction_points=runtime.geometry[0][0])
        bridge = ScheduledV4ExecutorInputs(runtime, scheduler)
        end = min(len(data['action']), int(row['unsafe_onset']))
        if end <= 6 or not np.allclose(np.diff(data['time']), 1/60, atol=1e-6):
            raise ValueError('continuous labelled60Hz history required')
        selected = set(np.linspace(6, end-1, min(args.rows_per_record, end-6), dtype=int).tolist())
        for tick in range(end):
            if time.monotonic()-started > args.seconds:
                raise TimeoutError('preparation budget exhausted')
            if tick not in selected:
                scheduler.update(poses[tick], float(data['time'][tick]), hand_points=data['hand'][tick])
                continue
            obs = data['observation'][tick]
            p2 = np.concatenate([obs[a:b] for a,b in ((0,44),(66,131),(161,191))])
            result = bridge.observation(objects[tick-6:tick+1], data['hand'][tick-6:tick+1],
                data['time'][tick-6:tick+1], poses[tick], p2)
            action = data['action'][tick]
            if action.shape != (24,) or not np.isfinite(action).all() or np.abs(action).max()>1.00001:
                raise ValueError('canonical24 action label required')
            features.append(result['observation']); labels.append(action)
            groups.append(record_id); ticks_out.append(tick); train_mask.append(record_id in args.train_records)
        del runtime, bridge
        print(json.dumps(dict(record=record_id, rows=len(features), elapsed=time.monotonic()-started)), flush=True)
    x, y, is_train = np.stack(features), np.stack(labels), np.asarray(train_mask, bool)
    np.savez_compressed(args.output/'data.npz', observation=x, action=y,
                        record_id=groups, tick=ticks_out, train=is_train)
    report = dict(status='PREPARED', seed=args.seed, wam=str(args.wam), cache=str(args.cache),
        train_records=args.train_records, dev_records=args.dev_records,
        rows=int(len(x)), train_rows=int(is_train.sum()), dev_rows=int((~is_train).sum()),
        split='raw-episode-disjoint internal TRAIN development; no global DEV claim',
        conditioning='measured H4 + task-goal scheduler + frozen WAM; no GT tau',
        startup_limitation='teacher histories; native six-zero startup is not covered by this packet',
        python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.gpu), fit_steps=args.fit_steps)
    if args.fit_steps:
        mean=x[is_train].mean(0); std=x[is_train].std(0); std=np.where(std<1e-6,1.,std)
        device=f'cuda:{args.gpu}'
        xt=torch.tensor((x-mean)/std,device=device); yt=torch.tensor(y,device=device)
        mask=torch.tensor(is_train,device=device)
        model=build_model(403,256).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        for step in range(args.fit_steps):
            if time.monotonic()-started > args.seconds: raise TimeoutError('BC budget exhausted')
            opt.zero_grad(); loss=torch.nn.functional.smooth_l1_loss(model(xt[mask]),yt[mask],beta=.1)
            loss.backward();opt.step()
        with torch.no_grad():
            prediction=model(xt)
            report.update(status='FITTED',train_l1=float((prediction[mask]-yt[mask]).abs().mean()),
                dev_l1=float((prediction[~mask]-yt[~mask]).abs().mean()),
                dev_constant_l1=float((yt[~mask]-yt[mask].mean(0)).abs().mean()),
                dev_clipped_fraction=float((prediction[~mask].abs()>1).float().mean()))
        torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,variant='p2_tau',
            hidden=256,input_dim=403,seed=args.seed,conditioning='scheduled-wam-v4',
            wam=str(args.wam),train_records=args.train_records,dev_records=args.dev_records),args.output/'model.pt')
    report.update(elapsed_seconds=time.monotonic()-started,
                  peak_mib=torch.cuda.max_memory_allocated(args.gpu)/2**20)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
