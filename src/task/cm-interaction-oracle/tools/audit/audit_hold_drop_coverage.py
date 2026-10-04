#!/usr/bin/env python3
"""Audit existing episode labels before choosing a task-relevant G target."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run'))
from probe_pointflow_g import ROOT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'tmp/e260_all4_h16_histfix.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.input, map_location='cpu', weights_only=False, mmap=True)
    summaries = {}
    summary_hashes = []
    for info in source['metadata']['runs']:
        path = ROOT / info['run_dir'] / 'results.json'
        report = json.loads(path.read_text())
        summary_hashes.append({'path': str(path), 'sha256': sha256(path)})
        for row in report['per_episode']:
            ep = int(row['episode_id'])
            if ep in summaries:
                raise ValueError('ambiguous episode identity across summaries')
            summaries[ep] = row
    keys = torch.stack((source['source_run'], source['episode_id']), -1)
    episodes = []
    for key in torch.unique(keys, dim=0):
        mask = (keys == key).all(-1)
        aux = source['episode_auxiliary'][mask]
        if not torch.equal(aux, aux[0].expand_as(aux)):
            raise ValueError('episode-final labels vary within an episode')
        run, ep = map(int, key)
        summary = summaries[ep]
        label = aux[0]
        if bool(label[0]) != bool(summary['stable_success']) or bool(label[1]) != bool(summary['drop_after_success']):
            raise ValueError('assembled success/drop summary mismatch')
        if abs(float(label[2]) - float(summary['max_hold_seconds'])) > 1e-5:
            raise ValueError('assembled max hold summary mismatch')
        episodes.append({'source_run': run, 'episode_id': ep, 'success': bool(label[0]),
                         'drop_after_success': bool(label[1]), 'max_hold_seconds': float(label[2]),
                         'mean_abs_window_rtg': float(source['return_to_go'][mask].abs().mean()),
                         'rows': int(mask.sum())})
    largest = max(episodes, key=lambda row: row['mean_abs_window_rtg'])
    report = {'schema': 'ref2dex.hold_drop_target_coverage.v1',
              'input_sha256': sha256(args.input), 'script_sha256': sha256(__file__),
              'summary_files': summary_hashes, 'device': 'CPU: label joins and statistics only',
              'episode_count': len(episodes), 'success_count': sum(row['success'] for row in episodes),
              'drop_after_success_count': sum(row['drop_after_success'] for row in episodes),
              'success_without_recorded_later_drop_count': sum(row['success'] and not row['drop_after_success'] for row in episodes),
              'nonzero_hold_count': sum(row['max_hold_seconds'] > 0 for row in episodes),
              'largest_mean_abs_window_rtg_episode': largest, 'episodes': episodes,
              'label_semantics': {'stable_success': 'ever 45 consecutive held steps, 1.5s at 30Hz',
                                  'drop_after_success': 'later height loss or >=6 contact-lost steps after first reaching primary success',
                                  'success_without_drop': 'derived conjunction over recorded episode only; not a new formal success metric'},
              'decision': 'do not immediately fit rare binary success/drop; inspect existing continuous hold and later-drop time series before choosing a new G fit',
              'limitations': ['coverage of this source_e260 four-run dataset only, not the accepted six-expert baseline',
                              'final labels audit does not reconstruct temporal ordering or validate the final evaluation protocol']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('episodes', 'summary_files')}, indent=2))


if __name__ == '__main__':
    main()
