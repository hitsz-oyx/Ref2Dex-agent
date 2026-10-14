"""Audit every native window, shared collate, and optional UniDex equivalence."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]
for path in [TASK/'src', REPO/'src/task/ptv3-shared-wam/src', REPO/'src/task/video-data-preparation/src']:
    sys.path.insert(0, str(path))
from pose_mesh_data import PoseMeshWindows
from ptv3_shared_wam.egodex_data import forecasting_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--compare-unidex', type=Path)
    args = parser.parse_args()
    reports = {}
    for split in ('train', 'dev'):
        dataset = PoseMeshWindows(args.root, split, engineering=True)
        old = None
        if args.compare_unidex:
            from video_data_preparation.unidexgrasp_data import UniDexGraspWindows
            old = UniDexGraspWindows(args.compare_unidex, split, engineering=True)
            assert len(old) == len(dataset)
        sources, objects, valid, maximum = set(), set(), 0, 0.
        for i in range(len(dataset)):
            sample = dataset[i]
            for key in ('xyz', 'features', 'start_hand', 'start_hand_history', 'trajectory_target',
                        'scene_points', 'scene_target', 'goal_object_points'):
                if not np.isfinite(sample[key]).all():
                    raise ValueError('nonfinite '+key)
            if old is not None:
                reference = old[i]
                for key in ('xyz', 'features', 'start_hand', 'start_hand_history', 'trajectory_target',
                            'scene_target', 'goal_object_points'):
                    delta = float(abs(sample[key]-reference[key]).max())
                    maximum = max(maximum, delta)
                    np.testing.assert_allclose(sample[key], reference[key], atol=2e-4, rtol=1e-5)
            sources.add(int(sample['source_id'])); objects.add(len(sample['scene_points']))
            valid += int(sample['trajectory_valid'].sum())
        batch = forecasting_collate([dataset[0], dataset[-1]])
        reports[split] = dict(windows=len(dataset), source_ids=sorted(sources), object_counts=sorted(objects),
            valid_hand_points=valid, shared_collate_scene_shape=list(batch['scene_target'].shape),
            comparison_max_abs=maximum if old else None)
    report = dict(status='ENGINEERING_PASS', training_allowed=False, splits=reports,
                  limits=['No model training or efficacy claim.', 'Native target/profile visual review pending.'])
    (args.root/'provider-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
