"""Bounded train-only TCC alignment Probe; not evaluator/policy training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]; ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import is_within
from consequence_evaluator.data import sha
from consequence_evaluator.reference_progress import FeatureScale, trajectory_features
from consequence_evaluator.temporal_phase import ENCODER_SCHEMA, TemporalPhaseEncoder, context_clips
from consequence_evaluator.value_outcomes import RAW_SCHEMA, task_trace
from consequence_evaluator.xirl_tcc_loss import compute_tcc_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--seconds', type=int, default=300)
    args = parser.parse_args(); out = args.output.resolve()
    if (out.exists() or not is_within(out, ROOT / 'outputs/consequence-evaluator')
            or not 10 <= args.updates <= 2000 or not 30 <= args.seconds <= 900 or not 0 <= args.gpu <= 7):
        parser.error('fresh owned output, <=2000updates/900s and GPU0..7 required')
    used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True).strip())
    if used > 256:
        raise RuntimeError('selected GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    torch.set_num_threads(1); torch.manual_seed(281); np.random.seed(281)
    started = time.monotonic(); frozen = {str(Path(__file__).resolve()): sha(__file__)}
    for name in ('temporal_phase.py', 'xirl_tcc_loss.py', 'reference_progress.py', 'xirl_alignment.py', 'value_outcomes.py', 'data.py', 'contracts.py'):
        path = TASK / 'src/consequence_evaluator' / name; frozen[str(path)] = sha(path)
    source = args.source.resolve(); reference = args.reference.resolve()
    raw_path = source / 'manifest.json'; ref_path = reference / 'manifest.json'
    raw = json.loads(raw_path.read_text()); meta = json.loads(ref_path.read_text())
    if raw.get('schema') != RAW_SCHEMA or raw.get('status') != 'COMPLETED' or raw.get('seed') != 230:
        raise ValueError('fixed independent nominal train230 source required')
    ref_file = reference / 'reference.npz'
    frozen.update(raw['sources']); frozen.update(meta['sources'])
    frozen.update({str(raw_path): sha(raw_path), str(ref_path): sha(ref_path), str(ref_file): meta['reference_sha256']})
    if any(sha(p) != h for p, h in frozen.items()):
        raise ValueError('fixed alignment train/reference sources drifted')
    with np.load(ref_file, allow_pickle=False) as f:
        features = trajectory_features(f['object_pose'], f['hand_keypoints'], f['timestamps'])
    scaler = FeatureScale([features]); views = [features]; selected = []
    for record in raw['episodes']:
        if len(selected) >= 8:
            break
        if record['split'] != 'train' or record['assigned_phase'] != 'clean':
            continue
        paths = [source / record['path'], source / record['diagnostics']]
        for path, key in zip(paths, ('sha256', 'diagnostics_sha256')):
            if not is_within(path, source) or sha(path) != record[key]:
                raise ValueError('train packet drift/path escape')
            frozen[str(path)] = record[key]
        with np.load(paths[0], allow_pickle=False) as f, np.load(paths[1], allow_pickle=False) as d:
            packet = {key: f[key] for key in f.files}; diagnostic = {key: d[key] for key in d.files}
        # Old weak success is used ONLY to qualify train demonstration views;
        # it never enters the encoder, progress anchor or new Y labels.
        if not task_trace(packet, diagnostic)['task_success']:
            continue
        views.append(trajectory_features(packet['object_pose'], packet['hand_keypoints'], packet['timestamps']))
        selected.append(record['episode'])
    if len(selected) != 8:
        raise ValueError('eight independent weak-success train views required')
    clips = [torch.as_tensor(context_clips(scaler.apply(v)), dtype=torch.float32, device='cuda:0') for v in views]
    encoder = TemporalPhaseEncoder().cuda(); optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(281)
    out.mkdir(parents=True)
    manifest = dict(schema=ENCODER_SCHEMA, status='RUNNING', run_id=out.name, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, seed=281, updates=args.updates, seconds_budget=args.seconds,
        role='temporal alignment encoder only; not evaluator or actor', sources=frozen,
        reference=str(reference), reference_sha256=meta['reference_sha256'],
        standardizer=scaler.dictionary(), training_demonstrations=selected, training_source_seed=230,
        evaluation_group='independent preserved train261; never fitted or used for checkpoint selection',
        loss='upstream deterministic TCC regression_mse; normalized cycle-back indices only in train loss',
        input='8 causal90D geometry frames; no actual time/index/phase/actions/outcomes/reference clock',
        prototype='MLP720/128/128/64 GELU, unit embedding, matching temperature0.1',
        training_allowed=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    log = out / 'loss.jsonl'; losses = []
    try:
        for step in range(1, args.updates + 1):
            if time.monotonic() - started > args.seconds:
                raise TimeoutError('fixed TCC Probe deadline')
            other = int(rng.integers(1, len(clips)))
            # Always include original R, with one train-only physical view.
            ids = [np.sort(rng.choice(len(clips[v]), 256, replace=False)) for v in (0, other)]
            embeddings = torch.stack([encoder(clips[v][torch.as_tensor(i, device='cuda:0')]) for v, i in zip((0, other), ids)])
            loss = compute_tcc_loss(embeddings, torch.as_tensor(np.stack(ids), device='cuda:0'),
                torch.tensor([len(clips[0]), len(clips[other])], device='cuda:0'),
                stochastic_matching=False, normalize_embeddings=True, loss_type='regression_mse',
                similarity_type='l2', temperature=.1, normalize_indices=True)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite TCC loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
            if step % 100 == 0 or step == 1:
                elapsed = time.monotonic() - started
                row = dict(update=step, loss_mean=float(np.mean(losses[-100:])), elapsed_s=elapsed,
                           eta_s=elapsed / step * (args.updates - step),
                           gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                with log.open('a') as stream: stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
        if any(sha(p) != h for p, h in frozen.items()):
            raise ValueError('TCC source drift')
        checkpoint = out / ('encoder_step_%04d.pt' % args.updates)
        torch.save(dict(schema=ENCODER_SCHEMA, model=encoder.cpu().state_dict(),
                        standardizer=scaler.dictionary()), checkpoint)
        manifest.update(status='COMPLETED', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                        final_loss_mean=float(np.mean(losses[-100:])))
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        manifest['elapsed_s'] = time.monotonic() - started
        manifest['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
