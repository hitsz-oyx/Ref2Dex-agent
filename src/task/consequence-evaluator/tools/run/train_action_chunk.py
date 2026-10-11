"""Train an engineering-only native 24-step action-chunk proposal.

The command never writes evaluator labels.  It is a small imitation probe for
the frozen GPU behavior harness: ``H_t`` is read once and the model emits all
24 raw native controls without reading a future observation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.action_chunk import (
    ACTION_CHUNK_SCHEMA, EXECUTED_ACTION_SEMANTICS, HistoryStandardizer,
    NativeActionChunkProposal, ActionChunkBatch, NATIVE_RMS_EPSILON, chunk_metrics,
    load_action_chunk_batches)
from consequence_evaluator.contracts import K
from consequence_evaluator.data import sha


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


def _batch_slice(batch, mask):
    mask = np.asarray(mask, dtype=bool)
    return ActionChunkBatch(batch.history[mask], batch.action[mask],
                            batch.episode[mask], batch.tick[mask])


def _nonempty(batch):
    return len(batch.history) > 0


def _load_frozen_rms(path, history_dim):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    stats = payload.get('running_mean_std') if isinstance(payload, dict) else None
    if not isinstance(stats, dict) or 'running_mean' not in stats or 'running_var' not in stats:
        raise ValueError('checkpoint has no running_mean_std contract')
    mean = np.asarray(stats['running_mean'], dtype='float32')
    variance = np.asarray(stats['running_var'], dtype='float32')
    if mean.shape != (history_dim,) or variance.shape != (history_dim,):
        raise ValueError('frozen checkpoint RMS dimension does not match history')
    return HistoryStandardizer.from_running_stats(mean, variance)


def _gpu_snapshot(gpu):
    try:
        text = subprocess.check_output([
            'nvidia-smi', '-i', str(gpu),
            '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL).strip()
        values = [int(value.strip()) for value in text.split(',')]
        if len(values) == 3:
            return dict(utilization_gpu_pct=values[0], memory_used_mib=values[1],
                        memory_total_mib=values[2])
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return None


def _metrics(model, batch, standardizer, device, mean_target):
    if not _nonempty(batch):
        return None
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(batch.history), 512):
            h = torch.from_numpy(standardizer.transform(batch.history[start:start + 512])).to(device)
            predictions.append(model(h).cpu())
    prediction = torch.cat(predictions)
    target = torch.from_numpy(batch.action)
    metrics = chunk_metrics(prediction, target, mean_target)
    metrics['windows'] = int(len(batch.history))
    metrics['episodes'] = int(len(set(batch.episode.tolist())))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--quality', action='append', default=None)
    parser.add_argument('--expert', action='append')
    parser.add_argument('--task', action='append')
    parser.add_argument('--motion', action='append')
    parser.add_argument('--clean-only', action='store_true',
                        help='select only metadata-clean episodes with zero, fully-known residual plans')
    parser.add_argument('--history-rms', type=Path,
                        help='optional native checkpoint whose running_mean_std is frozen')
    parser.add_argument('--allow-audit-only', action='store_true',
                        help='explicitly mark an audit-only source as engineering-only')
    parser.add_argument('--holdout-episodes', type=int, default=0,
                        help='move the lexicographically last train episodes to an engineering val split')
    args = parser.parse_args()
    args.data = args.data.resolve(); args.output = args.output.resolve()
    owned = ROOT / 'outputs/consequence-evaluator'
    if not _within(args.data, owned) or not _within(args.output, owned):
        parser.error('data/output must remain under task-owned outputs')
    if args.history_rms is not None:
        args.history_rms = args.history_rms.resolve()
        if not _within(args.history_rms, owned) or not args.history_rms.exists():
            parser.error('history RMS checkpoint does not exist')
    if (args.output.exists() or not 30 <= args.seconds <= 900 or args.steps < 1
            or not 1 <= args.batch_size <= 512 or args.stride < 1
            or not 1 <= args.width <= 1024 or args.width % 4 != 0 or not 1 <= args.layers <= 8
            or args.learning_rate <= 0 or not 0 <= args.gpu <= 7
            or args.holdout_episodes < 0):
        parser.error('invalid bounded action-chunk training budget or existing output')
    if args.allow_audit_only and args.output.name.find('engineering') < 0:
        parser.error('audit-only source outputs must include engineering in the run id')
    if args.clean_only and args.history_rms is None:
        parser.error('clean-only native deployment fit requires --history-rms')
    if args.clean_only and (not args.expert or len(args.expert) != 1):
        parser.error('clean-only native deployment fit requires exactly one --expert')
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    qualities = tuple(args.quality or ('expert_success',))
    batches, source_manifest = load_action_chunk_batches(
        args.data, qualities=qualities, stride=args.stride,
        experts=args.expert, tasks=args.task, motions=args.motion,
        allow_audit_only=args.allow_audit_only, clean_only=args.clean_only)
    history_rms_sha256 = None
    history_rms_source_match = None
    if args.history_rms is not None:
        history_rms_sha256 = sha(args.history_rms)
        selected_expert = args.expert[0] if args.expert and len(args.expert) == 1 else None
        source_hashes = {
            str(value) for path, value in source_manifest.get('sources', {}).items()
            if isinstance(value, str) and selected_expert is not None
            and ('/' + selected_expert + '/') in str(path) and str(path).endswith('.pth')}
        history_rms_source_match = history_rms_sha256 in source_hashes
        if args.clean_only and not history_rms_source_match:
            parser.error('clean-only source does not contain the supplied history RMS for the selected expert')
    train = batches.get('train')
    if train is None or not _nonempty(train):
        parser.error('selected source has no train action chunks')
    val = batches.get('val')
    if args.holdout_episodes:
        if args.holdout_episodes >= len(set(train.episode.tolist())):
            parser.error('holdout must leave at least one train episode')
        names = sorted(set(train.episode.tolist()))
        holdout = set(names[-args.holdout_episodes:])
        val = _batch_slice(train, np.asarray([e in holdout for e in train.episode]))
        train = _batch_slice(train, np.asarray([e not in holdout for e in train.episode]))
    if val is None:
        val = ActionChunkBatch(np.empty((0, train.history.shape[1]), dtype='float32'),
                               np.empty((0, K, 18), dtype='float32'),
                               np.empty((0,), dtype='<U1'), np.empty((0,), dtype=np.int64))
    if not _nonempty(train):
        parser.error('holdout removed all training episodes')
    standardizer = (_load_frozen_rms(args.history_rms, train.history.shape[-1])
                    if args.history_rms is not None else HistoryStandardizer.fit(train.history))
    test = batches.get('test')
    if test is None:
        test = ActionChunkBatch(np.empty((0, train.history.shape[1]), dtype='float32'),
                                np.empty((0, K, 18), dtype='float32'),
                                np.empty((0,), dtype='<U1'), np.empty((0,), dtype=np.int64))
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    model = NativeActionChunkProposal(train.history.shape[-1], width=args.width, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_history = torch.from_numpy(standardizer.transform(train.history))
    train_target = torch.from_numpy(train.action)
    mean_target = train_target.mean(dim=0)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.monotonic(); best_state = None; best_value = float('inf'); events = []
    last_report = started
    for step in range(1, args.steps + 1):
        elapsed = time.monotonic() - started
        if elapsed >= args.seconds:
            break
        model.train()
        indices = torch.randint(len(train_history), (min(args.batch_size, len(train_history)),), generator=generator)
        history = train_history[indices].to(device)
        target = train_target[indices].to(device)
        prediction = model(history)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            val_metrics = _metrics(model, val, standardizer, device, mean_target)
            chosen = val_metrics['mse'] if val_metrics is not None else float(loss.detach().cpu())
            if chosen < best_value:
                best_value = chosen; best_state = copy.deepcopy(model.state_dict())
            now = time.monotonic(); rate = step / max(now - started, 1e-6)
            event = dict(step=step, elapsed_s=now - started, loss=float(loss.detach().cpu()),
                         val_mse=None if val_metrics is None else val_metrics['mse'],
                         steps_per_s=rate, eta_s=max(args.steps - step, 0) / max(rate, 1e-6),
                         gpu=_gpu_snapshot(args.gpu) if device.type == 'cuda' else None)
            events.append(event); print(json.dumps(event), flush=True); last_report = now
        if time.monotonic() - last_report > 30:
            event = dict(step=step, elapsed_s=time.monotonic() - started,
                         steps_per_s=step / max(time.monotonic() - started, 1e-6),
                         gpu=_gpu_snapshot(args.gpu) if device.type == 'cuda' else None)
            events.append(event); print(json.dumps(event), flush=True); last_report = time.monotonic()
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    metrics = dict(train=_metrics(model, train, standardizer, device, mean_target),
                   val=_metrics(model, val, standardizer, device, mean_target),
                   test=_metrics(model, test, standardizer, device, mean_target))
    args.output.mkdir(parents=True)
    checkpoint = args.output / 'action_chunk.pt'
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(dict(schema=ACTION_CHUNK_SCHEMA, executed_action_semantics=EXECUTED_ACTION_SEMANTICS,
                    history_dim=int(train.history.shape[-1]), chunk=K, action_dim=18,
                    width=args.width, layers=args.layers, state_dict=cpu_state,
                    history_standardizer=standardizer.as_dict(),
                    history_normalization_epsilon=NATIVE_RMS_EPSILON), checkpoint)
    selected_episodes = sorted(set(train.episode.tolist()) | set(val.episode.tolist()) |
                               (set(test.episode.tolist()) if _nonempty(test) else set()))
    source_files = {}
    for record in source_manifest.get('episodes', []):
        if record.get('episode') in selected_episodes and record.get('sha256'):
            source_files[str(args.data / record['path'])] = record['sha256']
    manifest = dict(schema=ACTION_CHUNK_SCHEMA, status='COMPLETED', training_allowed=False,
                    engineering_only=True, run_id=args.output.name, experiment_id='P-20261009-act-native-chunk',
                    task='consequence-evaluator', git_commit=subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_manifest=str(args.data / 'manifest.json'), source_manifest_sha256=sha(args.data / 'manifest.json'),
                    source_files=source_files, source_training_allowed=source_manifest.get('training_allowed'),
                    allow_audit_only=bool(args.allow_audit_only), qualities=list(qualities),
                    clean_only=bool(args.clean_only), expert_filter=args.expert,
                    task_filter=args.task, motion_filter=args.motion,
                    selected_episodes=selected_episodes, train_episodes=sorted(set(train.episode.tolist())),
                    val_episodes=sorted(set(val.episode.tolist())), test_episodes=sorted(set(test.episode.tolist())),
                    stride=args.stride, chunk=K, action_dim=18, executed_action_semantics=EXECUTED_ACTION_SEMANTICS,
                    history_normalization='frozen_checkpoint_running_mean_std' if args.history_rms else 'train_episode_standardizer',
                    history_normalization_epsilon=NATIVE_RMS_EPSILON,
                    history_rms=None if args.history_rms is None else str(args.history_rms),
                    history_rms_sha256=history_rms_sha256,
                    history_rms_source_match=history_rms_source_match,
                    history_rms_expert=(args.expert[0] if args.expert and len(args.expert) == 1 else None),
                    history_dim=int(train.history.shape[-1]), model=dict(width=args.width, layers=args.layers),
                    device=str(device), seed=args.seed, requested_steps=args.steps,
                    completed_steps=len(events) and events[-1].get('step', 0), elapsed_s=time.monotonic() - started,
                    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), metrics=metrics,
                    events=events, limitation='offline imitation/action-space evidence only; no native behavior or GT value claim')
    _write(args.output / 'metrics.json', metrics)
    _write(args.output / 'manifest.json', manifest)
    print(json.dumps(dict(status='COMPLETED', output=str(args.output), metrics=metrics,
                          elapsed_s=manifest['elapsed_s']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
