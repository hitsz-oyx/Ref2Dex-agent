#!/usr/bin/env python3
"""Exploratory cached-condition retrieval; oracle goals are NOT deployable inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from audit_native_curation import masks


def phase(metric):
    d0, df, end = metric[:, 4], metric[:, 5], metric[:, 6]
    label = np.full(len(metric), 3, np.int64)
    label[end <= 50] = 2
    label[(d0 <= 50) & (end > 100)] = 1
    label[(d0 > 50) & (df <= 50)] = 0
    return np.eye(4, dtype=np.float32)[label]


def normalized_pair(train, query):
    mean, scale = train.mean(0), np.maximum(train.std(0), 1e-3)
    divisor = scale * np.sqrt(train.shape[1])
    return ((train - mean) / divisor).astype(np.float32), ((query - mean) / divisor).astype(np.float32)


def shuffle_present(array, presence, rng):
    out = array.copy()
    for row in np.unique(presence, axis=0):
        ids = np.flatnonzero((presence == row).all(-1))
        out[ids] = array[rng.permutation(ids)]
    return out


def errors(pred, target, valid):
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    result = {}
    for name, stop in [('first8', 8), ('all24', 24)]:
        v = valid[:, :stop]
        result[name + '_mm'] = ((error[:, :stop] * v).sum((1, 2, 3)) / v.sum((1, 2, 3)) * 1000).cpu().numpy()
    return result


def grouped(values, metric):
    groups = dict(all=np.ones(len(metric), bool), **masks(metric))
    return {name: dict(n=int(mask.sum()), **{key: float(v[mask].mean()) for key, v in values.items()})
            for name, mask in groups.items() if mask.any()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=299)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Use an idle GPU for batched feature-distance calculations')
    data = {}
    for split in ('train', 'val'):
        with np.load(args.root / 'model-r1' / (split + '.npz')) as archive:
            data[split] = {key: archive[key] for key in archive.files}
    result = dict(status='RUNNING', code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  seed=args.seed, k=5, sources={}, oracle_only=True,
                  metric='k5 mean-trajectory EPE, not the frozen neural model EPE',
                  phase_names=['far_to_near', 'near_to_far', 'near_at_end', 'other'])
    saved = {}
    rng = np.random.default_rng(args.seed)
    for sid, source in enumerate(('oakink2', 'grab', 'arctic')):
        tr = {key: value[data['train']['source'] == sid] for key, value in data['train'].items()}
        source_result = {}
        for split in ('train', 'val'):
            query = {key: value[data[split]['source'] == sid] for key, value in data[split].items()}
            feature_pairs = [normalized_pair(tr[key], query[key])
                             for key in ('scene_embedding', 'goal_embedding', 'hand_embedding')]
            train_base = np.concatenate([pair[0] for pair in feature_pairs], -1)
            query_base = np.concatenate([pair[1] for pair in feature_pairs], -1)
            distances = torch.cdist(torch.tensor(query_base, device='cuda'),
                                    torch.tensor(train_base, device='cuda')).square()
            allowed = (query['sequence'][:, None] != tr['sequence'][None])
            allowed &= (query['presence'][:, None] == tr['presence'][None]).all(-1)
            allowed = torch.tensor(allowed, device='cuda')
            if (allowed.sum(-1) < 5).any():
                raise ValueError('Fewer than five eligible cross-sequence neighbors')
            target = torch.tensor(query['target'], device='cuda')
            train_target = torch.tensor(tr['target'], device='cuda')
            valid = torch.tensor(query['valid'], device='cuda')
            train_valid = torch.tensor(tr['valid'], device='cuda')
            condition_pairs = dict(
                phase=(phase(tr['metrics']), phase(query['metrics'])),
                wrist=(tr['target'][:, -1, :, 0].reshape(len(tr['index']), -1),
                       query['target'][:, -1, :, 0].reshape(len(query['index']), -1)),
                hand22=(tr['target'][:, -1].reshape(len(tr['index']), -1),
                        query['target'][:, -1].reshape(len(query['index']), -1)),
            )
            variants = {'base': distances}
            for key, (left, right) in condition_pairs.items():
                lnorm, rnorm = normalized_pair(left, right)
                variants['oracle_' + key] = distances + torch.cdist(
                    torch.tensor(rnorm, device='cuda'), torch.tensor(lnorm, device='cuda')).square()
                ls = shuffle_present(lnorm, tr['presence'], rng)
                rs = shuffle_present(rnorm, query['presence'], rng)
                variants['shuffled_' + key] = distances + torch.cdist(
                    torch.tensor(rs, device='cuda'), torch.tensor(ls, device='cuda')).square()
            reports = {}
            for variant, distance in variants.items():
                nn = distance.masked_fill(~allowed, float('inf')).topk(5, largest=False).indices
                neighbor_valid = train_valid[nn]
                if not (neighbor_valid == valid[:, None]).all():
                    raise ValueError('Matching hand presence did not ensure matching target masks')
                pred = train_target[nn].mean(1)
                values = errors(pred, target, valid)
                reports[variant] = grouped(values, query['metrics'])
                for key, value in values.items():
                    saved[f'{source}/{split}/{variant}/{key}'] = value
                saved[f'{source}/{split}/{variant}/neighbors'] = tr['index'][nn.cpu().numpy()]
            fraction = torch.arange(1, 25, device='cuda')[None, :, None, None, None] / 24
            endpoint_line = target[:, -1:] * fraction
            wrist_line = target[:, -1:, :, :1] * fraction
            for name, pred in [('oracle_hand_linear', endpoint_line), ('oracle_wrist_linear', wrist_line.expand_as(target)),
                               ('zero_motion', torch.zeros_like(target))]:
                reports[name] = grouped(errors(pred, target, valid), query['metrics'])
            source_result[split] = dict(n=len(query['index']), variants=reports,
                                        phase_counts=phase(query['metrics']).sum(0).astype(int).tolist())
            print(json.dumps(dict(source=source, split=split, elapsed_s=time.time() - started)), flush=True)
            if time.time() - started > 300:
                raise TimeoutError('Five-minute budget')
        result['sources'][source] = source_result
    result.update(status='COMPLETED', elapsed_s=time.time() - started,
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  limitations='Oracle future-derived inputs; kNN metric choice and representation aliasing; not architecture or deployment validation')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output / 'window_results.npz', **saved)
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_s', 'gpu_peak_reserved_bytes')}), flush=True)


if __name__ == '__main__':
    main()
