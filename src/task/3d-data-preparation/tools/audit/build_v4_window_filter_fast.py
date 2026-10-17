"""Fast v4 window filter over frozen source metadata.

This deliberately avoids constructing full PTv3 samples. It reads only the
target canonical points, target pose transport, and semantic hand trajectories
needed by the v4 interaction-field labeler, producing the same deterministic
window classes as the training adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from pose_mesh_data.clean_data import CleanPoseMeshWindows
from pose_mesh_data.v4_interaction import (V4SamplingPolicy, classify_window, interaction_field,
                                            resampled_subset, retain_window)


def _classify(hand, hand_valid, obj_future, policy):
    target, _ = interaction_field(hand, hand_valid, obj_future, policy)
    return classify_window(hand, hand_valid, obj_future, target, policy)


def _main3(dataset, index, policy):
    source_id = int(np.searchsorted(dataset.offsets, index, side='right') - 1)
    source = dataset.sources[source_id]
    local = int(index - dataset.offsets[source_id])
    seq_id, anchor, tick, _ = map(int, source.rows[local])
    sequence = source.sequence(source.sequences[seq_id])
    anchor_pose = sequence['poses'][tick, anchor]
    world_to_anchor = np.linalg.inv(anchor_pose)
    valid_now = np.repeat(np.asarray(sequence['hand_valid'][tick], bool)[:, None], 11, axis=1)
    hand_valid = np.repeat(np.asarray(sequence['hand_valid'][tick + 1:tick + 25], bool)[:, :, None], 11, axis=2) & valid_now[None]
    hand = np.asarray(sequence['hand'][tick + 1:tick + 25], np.float32).reshape(24, 2, 11, 3)
    hand = hand @ world_to_anchor[:3, :3].T + world_to_anchor[:3, 3]
    points = source.canonical(sequence['objects'][anchor])[0]
    query, _ = resampled_subset(points, policy.interaction_points, name='main3 interaction queries')
    poses = sequence['poses'][tick + 1:tick + 25, anchor]
    relative = np.einsum('ij,tjk->tik', world_to_anchor, poses)
    obj_future = np.einsum('tij,qj->tqi', relative[:, :3, :3], query) + relative[:, None, :3, 3]
    kind = _classify(hand, hand_valid, obj_future, policy)
    return kind, f'{source_id}:{seq_id}:{anchor}:{tick}'


def _native(dataset, index, policy):
    seq_id, tick = dataset.index[index]
    row = dataset.meta['sequences'][seq_id]
    sequence = dataset._sequence(seq_id)
    anchor = int(row['target_slot'])
    world_to_anchor = np.linalg.inv(sequence['object_poses'][tick, anchor])
    valid_now = np.asarray(sequence['hand_valid'][tick - 3:tick + 1].all(0), bool)
    hand_valid = np.asarray(sequence['hand_valid'][tick + 1:tick + 25], bool) & valid_now[None]
    hand = np.asarray(sequence['hand'][tick + 1:tick + 25], np.float32)
    hand = hand @ world_to_anchor[:3, :3].T + world_to_anchor[:3, 3]
    points = dataset.geometry[row['objects'][anchor]]['surface']
    query, _ = resampled_subset(points, policy.interaction_points, name='native interaction queries')
    poses = sequence['object_poses'][tick + 1:tick + 25, anchor]
    relative = np.einsum('ij,tjk->tik', world_to_anchor, poses)
    obj_future = np.einsum('tij,qj->tqi', relative[:, :3, :3], query) + relative[:, None, :3, 3]
    kind = _classify(hand, hand_valid, obj_future, policy)
    return kind, f'{row["source_id"]}:{seq_id}:{tick}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main3-root', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'dev'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--stop-index', type=int)
    p.add_argument('--static-keep-ratio', type=float, default=.10)
    p.add_argument('--approach-keep-ratio', type=float, default=.35)
    a = p.parse_args()
    policy = V4SamplingPolicy(static_keep_ratio=a.static_keep_ratio, approach_keep_ratio=a.approach_keep_ratio)
    from oakink_wm.multisource import MixedWindows
    main3 = MixedWindows(str(a.main3_root), 'train' if a.split == 'train' else 'val')
    native = CleanPoseMeshWindows(str(a.native_root), a.split, engineering=True, cache_size=8)
    counts, blocked = {}, []
    total = len(main3) + len(native)
    start = max(0, a.start_index)
    stop = total if a.stop_index is None else min(total, a.stop_index)
    if not 0 <= start <= stop <= total:
        raise ValueError('bounded global index range required')
    for i in range(start, stop):
        kind, identity = (_main3(main3, i, policy) if i < len(main3)
                          else _native(native, i - len(main3), policy))
        counts[kind] = counts.get(kind, 0) + 1
        if not retain_window(kind, identity, policy):
            blocked.append(i)
        if (i - start + 1) % 10000 == 0:
            print(f'processed={i + 1}/{total} blocked={len(blocked)}', flush=True)
    result = dict(schema='ref2dex.wam-v4-window-filter.v1', split=a.split,
                  policy=policy.__dict__, total_windows=total, scan_start=start, scan_stop=stop,
                  kept_windows=(stop - start) - len(blocked), blocked_windows=len(blocked), counts=counts,
                  blocked_indices=blocked,
                  source_roots=[str(a.main3_root.resolve()), str(native.root)])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
