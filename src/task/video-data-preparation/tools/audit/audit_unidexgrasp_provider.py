#!/usr/bin/env python3
"""Check actual compact trajectories, current-only scene inputs and shared collate."""
import argparse
import json
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
for path in (TASK/'src', TASK.parent/'ptv3-shared-wam/src', ROOT/'tmp/video-data-preparation-deps'):
    sys.path.insert(0, str(path))

import numpy as np
import torch
from video_data_preparation.unidexgrasp_data import UniDexGraspWindows, forecasting_sample
from video_data_preparation.sample_contract import FIXED_FIELDS
from ptv3_shared_wam.egodex_data import forecasting_collate
from ptv3_shared_wam.data import observed_inputs


def main(root):
    begin = time.monotonic()
    denied = False
    try:
        UniDexGraspWindows(root, 'train')
    except ValueError:
        denied = True
    if not denied:
        raise ValueError('provisional training access was not blocked')
    datasets = {s: UniDexGraspWindows(root, s, engineering=True) for s in ('train', 'dev')}
    checked = set()
    samples_checked = 0
    causal_fields = ('xyz', 'features', 'scene_object', 'object_valid', 'scene_points',
                     'start_hand', 'start_hand_history', 'start_hand_valid', 'start_object_points')
    for split, dataset in datasets.items():
        for i in range(len(dataset)):
            sample = dataset[i]
            for key, (shape, kind) in FIXED_FIELDS.items():
                value = np.asarray(sample[key])
                if value.shape != shape or value.dtype.kind != kind:
                    raise ValueError('forecast field mismatch: '+key)
            if sample['features'].shape != (len(sample['xyz']), 18) or not (sample['features'][:, 14] == 0).all():
                raise ValueError('current non-hand scene contract failed')
            if int(sample['source_id']) != 4 or sample['training_allowed']:
                raise ValueError('source/admission identity changed')
            if not np.allclose(sample['goal_object_points']-sample['start_object_points'],
                               sample['scene_target'][0, -1], atol=2e-6):
                raise ValueError('endpoint is not the same-ID scene label')
            if sample['start_hand_valid'][1].any() or sample['trajectory_valid'][:, 1].any():
                raise ValueError('missing left hand was fabricated')
            for key, value in sample.items():
                if np.asarray(value).dtype.kind == 'f' and not np.isfinite(value).all():
                    raise ValueError('nonfinite actual sample: '+key)
            trajectory_id, tick = dataset.index[i]
            trajectory = dataset._trajectory(trajectory_id)
            code = str(trajectory['object_code'])
            if code not in checked:
                changed = {k: v.copy() for k, v in trajectory.items()}
                changed['hand'][tick+1:] += 1000
                changed['object'][tick+1:, :3] += 500
                altered = forecasting_sample(changed, dataset.geometry[code], tick, trajectory_id)
                if any(not np.array_equal(sample[k], altered[k]) for k in causal_fields):
                    raise ValueError('future leaked into current scene/hand fields')
                if np.array_equal(sample['scene_target'], altered['scene_target']):
                    raise ValueError('future intervention did not affect actual labels')
                checked.add(code)
            samples_checked += 1
    # Exercise actual spawned workers and the existing Shared-WAM tensor boundary.
    joined = torch.utils.data.ConcatDataset(list(datasets.values()))
    loader = torch.utils.data.DataLoader(joined, batch_size=4, num_workers=2,
        multiprocessing_context='spawn', collate_fn=forecasting_collate)
    batches = []
    iterator = iter(loader)
    try:
        for _ in range(2):
            batch = next(iterator)
            current = observed_inputs(batch)
            if any(not torch.isfinite(v).all() for v in current.values()):
                raise ValueError('nonfinite current tensors at shared collate')
            if batch['source_id'].unique().tolist() != [4] or 'native_action' in batch:
                raise ValueError('native robot actions entered geometry provider')
            batches.append({key: list(value.shape) for key, value in batch.items()})
    finally:
        del iterator, loader
    summary = dict(status='COMPLETE', profile=datasets['train'].meta['profile'],
                   windows={s: len(d) for s, d in datasets.items()}, actual_samples_checked=samples_checked,
                   object_disjoint_splits=True, future_intervention_objects=sorted(checked),
                   source_id=4, training_access_blocked=denied, training_admitted=0,
                   shared_spawn_worker_batches=batches, elapsed_s=time.monotonic()-begin,
                   limitations=['provisional full CAD with palm-root ShadowHand semantics',
                                'not a shared model forward/backward or training admission',
                                'official cumulative success is separate from endpoint distance'])
    (root/'provider-audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
