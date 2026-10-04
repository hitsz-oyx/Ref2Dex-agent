#!/usr/bin/env python3
"""Decision diagnostic: E horizon versus E representation, fixed G capacity."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run'))
from probe_pointflow_g import (ROOT, sha256, pose_effect, subset_rows, _split,
                              _standardize, fit_bridge, evaluate, comparison)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'tmp/e260_all4_h16_histfix.pt')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for repeated model computation')
    source = torch.load(args.input, map_location='cpu', weights_only=False, mmap=True)
    idx = subset_rows(source, 64)
    raw = {'_namespace': source.get('source_namespace', torch.zeros_like(source['source_run']))[idx],
           '_run': source['source_run'][idx], '_episode': source['episode_id'][idx]}
    train, test, train_groups, test_groups = _split(raw, 20261004)
    keys = torch.stack([raw[k] for k in ('_namespace', '_run', '_episode')], -1)
    history = _standardize(torch.cat([source[k][idx].float() for k in
                            ('history_state', 'history_previous_action', 'history_context', 'history_progress')], -1), train)
    target = source['return_to_go'][idx].float()
    effect = source['effect'][idx, :8].float()
    pose = _standardize(pose_effect(effect), train)
    full = _standardize(effect, train)
    arms = {'H': torch.zeros(len(idx), 8, 30)}
    for k in (1, 4, 8):
        value = torch.zeros(len(idx), 8, 30)
        value[:, :k, :6] = pose[:, :k]
        arms['GT_pose_K%d' % k] = value
    arms['GT_full13_K8'] = F.pad(full, (0, 17))
    results = {}
    for name, value in arms.items():
        model, ym, ys, losses = fit_bridge(history, value, target, train, device, 16, 512, 201)
        result, _ = evaluate(model, history, value, target, test, keys, test_groups, device, ym, ys, 512)
        result['training_loss'] = losses
        results[name] = result
        print(json.dumps({'arm': name, 'mae': result['episode_balanced_mae']}), flush=True)
    contrasts = {name: comparison(results['H'], result, 201) for name, result in results.items() if name != 'H'}
    test_keys = keys[test]
    means = torch.tensor([target[test][(test_keys == torch.tensor(g)).all(-1)].abs().mean() for g in test_groups])
    largest = int(means.argmax())
    for name in contrasts:
        h = torch.tensor(results['H']['test_group_mae'])
        delta = h - torch.tensor(results[name]['test_group_mae'])
        keep = torch.arange(len(h)) != largest
        contrasts[name].update({'positive_episode_count': int((delta > 0).sum()),
                               'largest_target_episode': test_groups[largest],
                               'relative_gain_excluding_largest_target_episode': float(delta[keep].mean() / h[keep].mean())})
    promising = [name for name, cmp in contrasts.items() if cmp['relative_mae_reduction'] >= .05
                 and cmp['relative_gain_excluding_largest_target_episode'] > 0
                 and cmp['positive_episode_count'] >= 12]
    import subprocess
    report = {'schema': 'ref2dex.gt_effect_contract_probe.v1',
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'input': str(args.input), 'input_sha256': sha256(args.input), 'script_sha256': sha256(__file__),
              'run_id': args.output_dir.name, 'seed': 201, 'split_identifier': 20261004,
              'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
              'train_groups': train_groups, 'test_groups': test_groups,
              'arms': results, 'contrasts': contrasts,
              'status': 'PROMISING' if promising else 'UNPROMISING', 'promising_contracts': promising,
              'elapsed_seconds': time.monotonic() - started,
              'decision_rule': '>=5% gain, >=12/22 improving episodes, positive gain excluding largest-target episode; proceed only to smallest positive pose horizon; full13-only signal requires twist-aware E contract',
              'limitations': ['single seed and four source runs; contract selection is exploratory',
                              'all use same 8-step 30D padded branch, identical capacity/initialization/order',
                              'missing horizons/channels are normalized training-mean constants',
                              'E is future GT oracle; not deployment or policy utility']}
    (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'contrasts': contrasts}), flush=True)


if __name__ == '__main__':
    main()
