#!/usr/bin/env python3
"""Label/support qualification; CPU file processing, no model computation."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_paired_evaluator_resolution import sha


def qualify(source, additional=None, additional_audit=None):
    import torch
    from analyze_executable_contact_opportunity import split_group
    from src.task.CmResidual.contact_to_lift_macro import windows
    torch.set_num_threads(2)
    parts, episodes, groups, buckets, hashes = [], [], [], [], {}
    if additional:
        if additional_audit is None:
            raise ValueError('additional source audit required')
        audit = json.loads(additional_audit.read_text())
        if audit['run_status'] != 'COMPLETED' or audit['run_manifest_sha256'] != sha(additional / 'run_manifest.json'):
            raise ValueError('additional source audit drift')
    for index, directory in enumerate([source] + ([additional] if additional else [])):
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        if manifest['run_status'] != 'COMPLETED' or manifest['smoke_only']:
            raise ValueError('completed source required')
        if any(sha(Path(k)) != v for k, v in manifest['input_sha256'].items()):
            raise ValueError('source drift')
        for phase in manifest['phases']:
            path = Path(phase['directory']) / 'records.pt'
            if phase['run_status'] != 'COMPLETED' or sha(path) != phase['result']['record_sha256']:
                raise ValueError('record drift')
            b = torch.load(path, weights_only=False, map_location='cpu')
            local = [split_group(int(i), int(j)) for i, j in zip(b['motion_id'], b['start_frame'])]
            if index and any(v >= 70 for v in local):
                raise ValueError('additional source leaked into held groups')
            parts.append(windows(b)); episodes += b['episode_id']
            groups += [f'{int(i)}/{int(j)}' for i, j in zip(b['motion_id'], b['start_frame'])]
            buckets += local
            hashes[str(path.resolve())] = sha(path)
    data = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    bucket = torch.tensor(buckets)
    splits = {'fit': bucket < 50, 'cal': (bucket >= 50) & (bucket < 70), 'held': bucket >= 70}
    def support(mask):
        selected = mask.tolist()
        return dict(windows=int(mask.sum()), episodes=len({e for e, take in zip(episodes, selected) if take}),
                    initial_groups=len({g for g, take in zip(groups, selected) if take}))
    counts = {}
    for name, mask in splits.items():
        early = mask & data['early']
        counts[name] = dict(all=support(mask), early=support(early),
                            positive_lift=support(early & data['target'][:, 32].bool()),
                            negative_lift=support(early & ~data['target'][:, 32].bool()))
    held = counts['held']; early = held['early']
    adequate = early['windows'] >= 300 and early['episodes'] >= 24 and early['initial_groups'] >= 8
    for label in ['positive_lift', 'negative_lift']:
        s = held[label]
        adequate &= s['windows'] >= 24 and s['episodes'] >= 8 and s['initial_groups'] >= 4
        adequate &= counts['cal'][label]['windows'] >= 12
    return data, splits, counts, bool(adequate), hashes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional', type=Path); p.add_argument('--additional-audit', type=Path)
    args = p.parse_args()
    if args.output.exists(): raise ValueError('unique output required')
    begin = time.monotonic()
    _, _, counts, adequate, hashes = qualify(args.source, args.additional, args.additional_audit)
    result = dict(experiment_id='P-20261002-contact-to-lift-macro', status='COMPLETED',
                  supervision_adequate=adequate, counts=counts, input_sha256=hashes,
                  elapsed_seconds=time.monotonic()-begin, device_reason='CPU label/support tally; no neural computation')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2))
