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
from ptv3_shared_wam.loading import initialize_loader_worker, project_ipc_directory


def proximity_record(trajectory, row):
    """Describe target proximity in recorded states; this is not a grasp label.

    The official endpoint is at raw frame199. Compact30Hz stops at frame198;
    keep these clocks separate and never filter samples implicitly.
    """
    times = np.asarray(trajectory['time'])
    goal = np.asarray(trajectory['goal'])
    objects = np.asarray(trajectory['object'])
    if goal.shape != (len(times), 3) or not np.isfinite(goal).all():
        raise ValueError('finite same-execution target positions required')
    distance = np.linalg.norm(goal - objects[:, :3], axis=-1)
    near = distance <= .05
    tail_start = int(np.searchsorted(times, times[-1] - .5 - 1e-7))
    if len(times) < 16 or times[-1] - times[tail_start] < .5 - 1e-7:
        raise ValueError('half-second endpoint history required')
    last = len(near)
    while last > 0 and near[last - 1]:
        last -= 1
    return dict(path=row['path'], object_code=row['object_code'], split=row['split'],
                official_success=row['official_success'],
                official_endpoint_within_5cm=row['endpoint_within_5cm'],
                official_endpoint_time_s=float(trajectory['terminal_time']),
                sampled_endpoint_time_s=float(times[-1]),
                sampled_endpoint_distance_m=float(distance[-1]),
                sampled_endpoint_within_5cm=bool(near[-1]),
                sampled_tail_half_second_within_5cm=bool(near[tail_start:].all()),
                sampled_tail_duration_s=float(times[-1]-times[tail_start]),
                sampled_final_proximity_duration_s=float(times[-1]-times[last]) if last < len(times) else 0.,
                official_hit_then_sampled_endpoint_outside=bool(row['official_success'] and not near[-1]))


def main(root, output=None):
    output = output or root/'provider-audit.json'
    if output.exists():
        raise FileExistsError('retain previous audit; choose a new --output')
    begin = time.monotonic()
    torch.set_num_threads(2)
    project_ipc_directory()
    denied = False
    try:
        UniDexGraspWindows(root, 'train')
    except ValueError:
        denied = True
    if not denied:
        raise ValueError('provisional training access was not blocked')
    datasets = {s: UniDexGraspWindows(root, s, engineering=True) for s in ('train', 'dev')}
    checked = set()
    proximity = []
    samples_checked = 0
    causal_fields = ('xyz', 'features', 'scene_object', 'object_valid', 'scene_points',
                     'start_hand', 'start_hand_history', 'start_hand_valid', 'start_object_points')
    for split, dataset in datasets.items():
        for trajectory_id in sorted({i for i, _ in dataset.index}):
            row = dataset.meta['trajectories'][trajectory_id]
            proximity.append(proximity_record(dataset._trajectory(trajectory_id), row))
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
        multiprocessing_context='spawn', collate_fn=forecasting_collate,
        worker_init_fn=initialize_loader_worker)
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
                   proximity_by_split={s: dict(trajectories=sum(r['split']==s for r in proximity),
                       **{key: sum(r[key] for r in proximity if r['split']==s) for key in (
                           'official_success', 'official_endpoint_within_5cm', 'sampled_endpoint_within_5cm',
                           'sampled_tail_half_second_within_5cm', 'official_hit_then_sampled_endpoint_outside')})
                       for s in datasets},
                   trajectory_proximity=proximity,
                   limitations=['provisional full CAD with palm-root ShadowHand semantics',
                                'not a shared model forward/backward or training admission',
                                'target proximity, including half-second retention, is not stable grasp evidence'])
    output.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'trajectory_proximity'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='new report path; existing reports are never overwritten')
    args = parser.parse_args()
    main(args.run.resolve(), args.output)
