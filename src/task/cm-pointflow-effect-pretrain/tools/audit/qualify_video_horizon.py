#!/usr/bin/env python3
"""Audit physical-horizon support and publish a non-copying, shorter Probe view."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_points import sample_window


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--horizon', type=int, choices=(8,), default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.source / 'manifest.json').read_text())
    if original['status'] not in ('QUALIFIED_VIDEO_PROBE_ONLY', 'INSUFFICIENT_LONG_HORIZON_SUPPORT'):
        raise ValueError('source preparation failed')
    counts, clips, entries = {'train': 0, 'dev': 0}, [], []
    for e in original['sequences']:
        if sha(args.source / e['file']) != e['sha256']:
            raise ValueError('source pack drift')
        total = np.zeros(24, 'int64')
        with np.load(args.source / e['file'], allow_pickle=False) as z:
            for block in e['window_starts']:
                s = sample_window(z['points'][block], z['valid'][block], z['kind'][block], z['timestamps'][block], 0)
                total += s['target_valid'][s['point_kind'] == 1].sum(0)
        qualified = int(total[args.horizon - 1]) >= 16
        counts[e['split']] += qualified
        clips.append(dict(scene=e['scene'], split=e['split'], object_support_by_horizon=total.tolist(),
                          qualified_for_macro=qualified))
        (args.output / e['file']).symlink_to((args.source / e['file']).resolve())
        entries.append(e)
    manifest = dict(original, schema='ref2dex.video-point-horizon-view.v1',
        status='QUALIFIED_VIDEO_PROBE_ONLY' if min(counts.values()) >= 2 else 'INSUFFICIENT_HORIZON_SUPPORT',
        protocol=dict(original['protocol'], supervised_horizon=args.horizon),
        source_manifest_sha256=sha(args.source / 'manifest.json'), source_root=str(args.source.resolve()),
        qualification=dict(min_points_per_clip=16, min_clips_per_split=2, supported_clips=counts, clips=clips),
        sequences=entries, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        qualifier_script_sha256=sha(Path(__file__)))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(manifest['qualification'], indent=2))


if __name__ == '__main__':
    main()
