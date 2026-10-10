#!/usr/bin/env python3
"""Bounded inventory and hand-window qualification of already-local EPIC data.

Only small metadata members are read from tar archives. This never extracts
videos, creates a training manifest, or changes the original split authorities.
"""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from convert_epic_scene_flow import WINDOW, load_contact_hand, source_clock, source_splits, sha


def audit(root):
    metadata = root / 'metadata'
    inventory = {}
    local_scenes = set()
    for archive in sorted((root / 'shard').glob('*.tar')):
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.name.endswith(('/spatracker.npz', '/action.mp4')):
                    local_scenes.add(Path(member.name).parent.name)
                if not member.name.endswith('/action.meta.json'):
                    continue
                if not member.isfile() or member.size > 65536:
                    raise ValueError('unexpected scene metadata member')
                scene = Path(member.name).parent.name
                if scene in inventory:
                    raise ValueError('duplicate scene in shard inventory')
                inventory[scene] = dict(metadata=json.load(tar.extractfile(member)),
                                       archive=str(archive.resolve()), member=member.name)
    pairs = []
    for contact in sorted((root / 'epic_contact').glob('*.npz')):
        prefix = '_'.join(contact.stem.split('_')[:2])
        with np.load(contact, allow_pickle=True) as z:
            frame_range = [int(z['_frame_num'].min()), int(z['_frame_num'].max())]
        qualities = []
        # Membership in the quality table determines Contact split, never the
        # ObjectForesight split or the output directory chosen by the operator.
        import csv
        for path in sorted(metadata.glob('epic_contact_*_frame_quality.csv')):
            with path.open() as f:
                if any(row['clip'] == contact.stem for row in csv.DictReader(f)):
                    qualities.append(path)
        if len(qualities) != 1:
            raise ValueError('Contact clip has missing or conflicting split authorities')
        quality = qualities[0]
        for scene, entry in sorted(inventory.items()):
            meta = entry['metadata']
            if not scene.startswith(prefix + '_') or max(frame_range[0], meta['start_frame']) >= min(frame_range[1], meta['stop_frame']):
                continue
            start, fps = source_clock(meta, meta['stop_frame'] - meta['start_frame'])
            splits = source_splits(scene, quality, metadata / 'train.txt', metadata / 'val.txt')
            offsets = np.arange(0, meta['stop_frame'] - start - 2 * (WINDOW - 1), 2)
            qualified = []
            records = []
            for offset in offsets:
                targets = int(offset) + 2 * np.arange(WINDOW)
                _, valid, _, _, hand = load_contact_hand(contact, targets, start,
                                                         quality_csv=quality, source_fps=fps)
                sides = valid.all(0)
                if sides.any():
                    qualified.append(dict(local_start=int(offset), sides=sides.tolist()))
                records.append(dict(local_start=int(offset), valid_frames=valid.sum(0).tolist()))
            pairs.append(dict(scene=scene, contact=str(contact.resolve()),
                              contact_sha256=sha(contact), metadata=meta,
                              splits=splits, sampled_fps=fps / 2,
                              hand_window_candidates=len(offsets),
                              hand_qualified_windows=len(qualified), qualified=qualified,
                              window_records=records,
                              overlap_semantics='source video/time overlap only; object/semantic pairing not qualified',
                              first_window_hand_audit=None if not len(offsets) else
                              load_contact_hand(contact, 2 * np.arange(WINDOW), start,
                                                quality_csv=quality, source_fps=fps)[4]))
    return dict(schema='ref2dex.epic-video-readiness.v1', status='CANDIDATE_ONLY',
                training_allowed=False, local_scene_count=len(local_scenes | set(inventory)),
                scene_clock_metadata_count=len(inventory),
                scenes_missing_clock_metadata=sorted(local_scenes - set(inventory)),
                local_contact_clip_count=len(list((root / 'epic_contact').glob('*.npz'))),
                overlap_pair_count=len(pairs),
                hand_qualified_windows=sum(p['hand_qualified_windows'] for p in pairs),
                hand_train_qualified_windows=sum(p['hand_qualified_windows'] for p in pairs
                                                  if p['splits']['train_eligible']),
                scenes=inventory, pairs=pairs,
                protocol=dict(window=WINDOW, source_stride=2, max_label_gap_source_frames=3,
                              all_11_joints_required=True, high_confidence_required=True,
                              verified_clip_required=True, extrapolation_allowed=False),
                limitations=['hand qualification is necessary but insufficient for scene-flow training',
                             'time-overlap candidates do not establish same-object pairing',
                             'inventory metadata does not qualify all camera/scene tracks',
                             'clock resampling and scene decoder adapter remain separate gates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    result = audit(args.data_root)
    result['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['script_sha256'] = sha(Path(__file__))
    result['converter_sha256'] = sha(Path(__file__).with_name('convert_epic_scene_flow.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('scenes', 'pairs')}, indent=2))


if __name__ == '__main__':
    main()
