"""Fit and audit an ACT-like proposal on native deployment episodes.

This is an engineering-only, leave-one-launch-out Probe.  The inputs are the
teacher arm from two same-process serial-cluster packets, not evaluator
labels.  A proposal is fitted on one launch's ``history[t] -> action[t:t+24]``
windows and evaluated on the other launch.  The tool deliberately never
starts Isaac Gym and never writes a training-eligible dataset.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.action_chunk import (  # noqa: E402
    ACTION_CHUNK_SCHEMA,
    ActionChunkBatch,
    HistoryStandardizer,
    NativeActionChunkProposal,
    chunk_metrics,
)
from consequence_evaluator.contracts import K  # noqa: E402
from consequence_evaluator.data import sha  # noqa: E402


SERIAL_SCHEMA = 'ref2dex.consequence-gate1.gpu-serial-cluster.v1'


def _within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _write(path, value):
    temporary = Path(path).with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _gpu_snapshot(gpu):
    try:
        text = subprocess.check_output([
            'nvidia-smi', '-i', str(gpu),
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True,
            stderr=subprocess.DEVNULL).strip()
        values = [int(value.strip()) for value in text.split(',')]
        if len(values) == 3:
            return dict(utilization_gpu_pct=values[0], memory_used_mib=values[1],
                        memory_total_mib=values[2])
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return None


def _load_packet(path):
    path = Path(path).resolve()
    if not path.exists() or not _within(path, ROOT / 'outputs/consequence-evaluator'):
        raise ValueError('serial packet must be an existing task-owned output')
    with path.open('rb') as stream:
        packet = pickle.load(stream)
    if (not isinstance(packet, dict) or packet.get('schema') != SERIAL_SCHEMA
            or packet.get('engineering_only') is not True
            or packet.get('training_allowed') is not False
            or packet.get('serial_cluster') is not True):
        raise ValueError('engineering serial-cluster packet required')
    roles = list(packet.get('roles') or [])
    expected_roles = ['reactive_teacher', 'frozen_zero_1', 'positive_1',
                      'frozen_zero_2', 'negative_1', 'frozen_zero_3',
                      'negative_2', 'frozen_zero_4', 'positive_2', 'frozen_zero_5']
    if (roles != expected_roles or packet.get('teacher_role') != 'reactive_teacher'
            or packet.get('execution_order') != expected_roles
            or packet.get('control_prefix_exact') is not True
            or not isinstance(packet.get('schedule_sha256'), str)):
        raise ValueError('packet must expose reactive_teacher as role zero')
    backend = packet.get('source_backend')
    if (not isinstance(backend, dict) or backend.get('name') != 'gpu_physx_gpu_pipeline'
            or backend.get('pipeline') != 'gpu' or backend.get('physx_use_gpu') is not True
            or backend.get('tensor_device') != 'cuda:0' or backend.get('actor_device') != 'cuda:0'):
        raise ValueError('native GPU PhysX/GPU tensor backend required')
    if not isinstance(packet.get('teacher_action_sha256'), str):
        raise ValueError('teacher action provenance hash required')
    if packet.get('query_tick') != 48 or packet.get('horizon') != K or packet.get('steps') != 72:
        raise ValueError('deployment Probe requires the fixed 48+24, 72-step contract')
    history = np.asarray(packet.get('history'))
    actions = np.asarray(packet.get('actions'))
    if history.ndim != 3 or actions.ndim != 3 or history.shape[0] != len(roles):
        raise ValueError('serial packet history/action rank mismatch')
    teacher_history = history[0]
    teacher_actions = actions[0]
    if teacher_history.shape != (73, 1442) or teacher_actions.shape != (72, 18):
        raise ValueError('unexpected native deployment teacher shape')
    if (not np.isfinite(teacher_history).all() or not np.isfinite(teacher_actions).all()
            or np.abs(teacher_actions).max() > 1 + 1e-6):
        raise ValueError('deployment teacher arrays must be finite native controls')
    starts = np.arange(0, len(teacher_actions) - K + 1, 4, dtype=np.int64)
    return dict(path=path, packet=packet, history=teacher_history.astype('float32'),
                actions=teacher_actions.astype('float32'), starts=starts)


def _batch(record):
    history = record['history'][record['starts']]
    action = np.stack([record['actions'][tick:tick + K] for tick in record['starts']]).astype('float32')
    episode = np.asarray([str(record['path'])] * len(history))
    return ActionChunkBatch(history, action, episode, record['starts'].copy())


def _empty_like(batch):
    return ActionChunkBatch(np.empty((0, batch.history.shape[1]), dtype='float32'),
                            np.empty((0, K, 18), dtype='float32'),
                            np.empty((0,), dtype='<U1'), np.empty((0,), dtype=np.int64))


def _metrics(model, batch, standardizer, device, mean_target):
    model.eval()
    if len(batch.history) == 0:
        return None
    values = []
    with torch.no_grad():
        for start in range(0, len(batch.history), 512):
            history = torch.from_numpy(standardizer.transform(batch.history[start:start + 512])).to(device)
            values.append(model(history).cpu())
    prediction = torch.cat(values)
    result = chunk_metrics(prediction, torch.from_numpy(batch.action), mean_target)
    result.update(windows=int(len(batch.history)), episodes=int(len(set(batch.episode.tolist()))))
    return result


def _load_reference(path, batch, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != ACTION_CHUNK_SCHEMA or payload.get('chunk') != K
            or payload.get('action_dim') != 18 or payload.get('history_dim') != 1442):
        raise ValueError('reference action-chunk checkpoint contract mismatch')
    standardizer_payload = payload.get('history_standardizer')
    if not isinstance(standardizer_payload, dict):
        raise ValueError('reference checkpoint has no history standardizer')
    standardizer = HistoryStandardizer(
        np.asarray(standardizer_payload['mean'], dtype='float32'),
        np.asarray(standardizer_payload['scale'], dtype='float32'),
        standardizer_payload.get('clip'))
    model = NativeActionChunkProposal(1442, width=int(payload['width']),
                                      layers=int(payload['layers'])).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model, standardizer


def _fit(train, valid, *, device, steps, seed, width, layers):
    standardizer = HistoryStandardizer.fit(train.history)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = NativeActionChunkProposal(1442, width=width, layers=layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_history = torch.from_numpy(standardizer.transform(train.history)).to(device)
    train_target = torch.from_numpy(train.action).to(device)
    mean_target = train_target.mean(dim=0)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best = float('inf')
    events = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(len(train_history),
                                (min(16, len(train_history)),), generator=generator)
        prediction = model(train_history[indices])
        loss = F.mse_loss(prediction, train_target[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            # Model selection may inspect the training launch only. Looking at
            # the held-out launch here would leak the LOO decision into the
            # fitted proposal; held-out metrics are computed once after fit.
            train_metrics = _metrics(model, train, standardizer, device, mean_target.cpu())
            chosen = train_metrics['mse'] if train_metrics is not None else float(loss.detach().cpu())
            if chosen < best:
                best = chosen
                best_state = copy.deepcopy(model.state_dict())
            events.append(dict(step=step, loss=float(loss.detach().cpu()),
                               train_mse=None if train_metrics is None else train_metrics['mse'],
                               gpu=_gpu_snapshot(device.index or 0)))
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state, strict=True)
    return model, standardizer, mean_target.cpu(), events, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--reference', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if len(args.packet) != 2:
        parser.error('exactly two launch packets are required')
    if (args.output.exists() or not 1 <= args.steps <= 5000 or not 0 <= args.gpu <= 7
            or args.width <= 0 or args.width % 4 or not 1 <= args.layers <= 8):
        parser.error('invalid bounded deployment-fit configuration')
    if not _within(args.output, ROOT / 'outputs/consequence-evaluator'):
        parser.error('output must remain under task-owned outputs')
    records = [_load_packet(path) for path in args.packet]
    hashes = [sha(record['path']) for record in records]
    if len(set(hashes)) != len(hashes):
        parser.error('launch packets must be distinct')
    identity = [record['packet'].get('replay_identity') for record in records]
    if any(not isinstance(value, dict) for value in identity):
        parser.error('serial packet replay identity required')
    if identity[0] != identity[1]:
        parser.error('launch packets must share physics/controller/backend identity')
    if records[0]['packet'].get('schedule_sha256') != records[1]['packet'].get('schedule_sha256'):
        parser.error('launch packets must share the frozen arm schedule')
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    reference_paths = [path.resolve() for path in args.reference]
    for path in reference_paths:
        if not path.exists() or not _within(path, ROOT / 'outputs/consequence-evaluator'):
            parser.error('reference checkpoint must be a task-owned existing output')
    batches = [_batch(record) for record in records]
    results = []
    pairs = ((batches[0], batches[1]), (batches[1], batches[0]))
    for fold, (train, valid) in enumerate(pairs, start=0):
        model, standardizer, mean_target, events, elapsed = _fit(
            train, valid, device=device, steps=args.steps, seed=args.seed + fold,
            width=args.width, layers=args.layers)
        fitted = _metrics(model, valid, standardizer, device, mean_target)
        train_metrics = _metrics(model, train, standardizer, device, mean_target)
        references = {}
        for path in reference_paths:
            ref_model, ref_standardizer = _load_reference(path, valid, device)
            references[str(path)] = dict(checkpoint_sha256=sha(path),
                                         metrics=_metrics(ref_model, valid, ref_standardizer,
                                                          device, mean_target))
        results.append(dict(fold=fold, fold_seed=args.seed + fold,
                            train_packet=str(train.episode[0]),
                            valid_packet=str(valid.episode[0]),
                            train=train_metrics, valid=fitted, references=references,
                            standardizer=standardizer.as_dict(), events=events,
                            elapsed_s=elapsed, checkpoint_sha256=None))
    args.output.mkdir(parents=True)
    manifest = dict(schema='ref2dex.consequence-evaluator.deployment-act-fit.v1',
                    status='COMPLETED', engineering_only=True, training_allowed=False,
                    experiment_id='P-20261009-act-deployment-conditioned',
                    task='consequence-evaluator', git_commit=subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    packets=[dict(path=str(record['path']), sha256=hashes[index],
                                  seed=record['packet']['seed'],
                                  teacher_role='reactive_teacher', windows=int(len(batch.history)))
                             for index, (record, batch) in enumerate(zip(records, batches))],
                    replay_identities=identity, device=str(device), seed=args.seed,
                    script_sha256=sha(Path(__file__).resolve()),
                    optimizer=dict(name='AdamW', learning_rate=1e-3,
                                   weight_decay=1e-4, batch_size=16),
                    model=dict(width=args.width, layers=args.layers, chunk=K, action_dim=18),
                    stride=4, steps=args.steps, references=[dict(path=str(path), sha256=sha(path))
                                                             for path in reference_paths],
                    folds=results,
                    limitation=('deployment-distribution/action-space Probe only; two launch-level '
                                'episodes, no simulator, GT Value, evaluator, or utility claim'))
    _write(args.output / 'metrics.json', dict(folds=results, status='COMPLETED'))
    _write(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(status='COMPLETED', output=str(args.output), folds=results),
                     allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
