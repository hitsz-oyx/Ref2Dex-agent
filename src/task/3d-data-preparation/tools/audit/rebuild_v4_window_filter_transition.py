"""Reclassify only the old blocked v4 windows under the current classifier.

The old filter manifests remain valid for the old run.  A clean new run that
uses the `moving_object_noncontact` class needs the blocked indices rechecked:
the new class retains non-contact moving-object windows at its own ratio
instead of treating them as 10%-thinned idle.  This tool never edits the old
manifest or source payloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from pose_mesh_data.clean_data import CleanPoseMeshWindows
from pose_mesh_data.dense_surface import apply_surface_overlay
from pose_mesh_data.v4_interaction import V4SamplingPolicy, retain_window
from build_v4_window_filter_fast import _main3, _native


def transition_blocked(blocked, classify, policy):
    """Return restored global indices and the rechecked kind counts."""
    restored, counts = [], Counter()
    for index in blocked:
        kind, identity = classify(int(index))
        counts[kind] += 1
        if retain_window(kind, identity, policy):
            restored.append(int(index))
    return restored, dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main3-root', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'dev'), required=True)
    p.add_argument('--old-filter', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--surface-manifest', type=Path)
    p.add_argument('--surface-sha256')
    p.add_argument('--static-keep-ratio', type=float, default=.10)
    p.add_argument('--approach-keep-ratio', type=float, default=.35)
    p.add_argument('--moving-object-keep-ratio', type=float, default=.35)
    p.add_argument('--probe-limit', type=int,
                   help='recheck only the first N blocked indices; write no filter')
    p.add_argument('--seconds', type=float, default=6 * 3600)
    a = p.parse_args()
    if bool(a.surface_manifest) != bool(a.surface_sha256):
        raise ValueError('dense filter geometry requires manifest and SHA-256')
    if a.probe_limit is not None and a.output is not None:
        p.error('probe mode writes no filter; omit --output')
    if a.probe_limit is None and a.output is None:
        p.error('full transition requires --output')
    old_path = a.old_filter.resolve()
    old_bytes = old_path.read_bytes()
    old = json.loads(old_bytes)
    if old.get('schema') != 'ref2dex.wam-v4-window-filter.v1' or old.get('split') != a.split:
        raise ValueError('v4 split window filter required')
    policy_values = dict(old['policy'])
    policy_values.pop('schema', None)
    policy_values.update(static_keep_ratio=a.static_keep_ratio,
                         approach_keep_ratio=a.approach_keep_ratio,
                         moving_object_keep_ratio=a.moving_object_keep_ratio)
    policy = V4SamplingPolicy(**policy_values)
    blocked = [int(i) for i in old['blocked_indices']]
    if len(blocked) != old['blocked_windows'] or any(
            i < 0 or i >= old['total_windows'] or (j and i <= blocked[j-1])
            for j, i in enumerate(blocked)):
        raise ValueError('sorted in-range old blocked indices required')

    from oakink_wm.multisource import MixedWindows
    main3 = MixedWindows(str(a.main3_root), 'train' if a.split == 'train' else 'val')
    native = CleanPoseMeshWindows(str(a.native_root), a.split, engineering=True, cache_size=8)
    if a.surface_manifest:
        apply_surface_overlay(native.geometry, a.surface_manifest, a.surface_sha256)
    main_length = len(main3)
    if old['total_windows'] != main_length + len(native):
        raise ValueError('filter total does not match current sources')

    def classify(index):
        return ((_main3(main3, index, policy) if index < main_length
                 else _native(native, index - main_length, policy)))

    scanned = blocked if a.probe_limit is None else blocked[:a.probe_limit]
    started = time.monotonic()
    restored, counts = [], Counter()
    for position, index in enumerate(scanned):
        if position and position % 10000 == 0:
            if time.monotonic() - started > a.seconds:
                raise TimeoutError('transition scan deadline')
            print(f'rechecked={position}/{len(scanned)} restored={len(restored)}', flush=True)
        kind, identity = classify(index)
        counts[kind] += 1
        if retain_window(kind, identity, policy):
            restored.append(index)
    report = dict(schema='ref2dex.wam-v4-filter-transition.v1', split=a.split,
                  parent_filter=str(old_path), parent_sha256=hashlib.sha256(old_bytes).hexdigest(),
                  scanned=len(scanned), restored=len(restored), restored_counts=dict(counts),
                  probe=a.probe_limit is not None, elapsed_s=time.monotonic() - started,
                  note=('Only old blocked windows are rechecked. Old kept windows stay kept '
                        'because contact is always retained and the approach/idle ratios are '
                        'unchanged; the new moving-object class can only restore blocks.'))
    if a.probe_limit is not None:
        print(json.dumps(report, indent=2))
        return
    restored_set = set(restored)
    remaining = [i for i in blocked if i not in restored_set]
    result = dict(old, policy=policy.__dict__, blocked_indices=remaining,
                  blocked_windows=len(remaining),
                  kept_windows=old['total_windows'] - len(remaining),
                  transition=report)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(output=str(a.output), **report), indent=2))


if __name__ == '__main__':
    main()
