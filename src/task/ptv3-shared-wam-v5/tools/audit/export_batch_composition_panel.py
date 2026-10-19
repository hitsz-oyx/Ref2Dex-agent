"""Export a bounded real-data panel and immutable checkpoint copy for GPU audit."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v5.data import V5Windows
from ptv3_shared_wam_v5.storage import read_json


def grid_depth(sample, config):
    coords = np.concatenate((sample['xyz'][sample['point_valid']],
        sample['start_hand'][sample['start_hand_valid']],
        sample['dense_hand'][sample['dense_hand_valid']]))
    grid = np.floor((coords - np.asarray(config['voxel_origin_m'], np.float32)) / config['voxel_m'])
    assert np.isfinite(grid).all() and grid.min() >= 0 and grid.max() < 65536
    return int(int(grid.max()) + 1).bit_length()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'recipe', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=177)
    p.add_argument('--git-commit', required=True)
    a = p.parse_args()
    if shutil.disk_usage(a.output.parent).free < 20 * 2**30:
        raise ValueError('20GiB free-space floor')
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    config = read_json(a.run / 'config.json')
    train = V5Windows(a.recipe / 'manifest.json', 'train')
    dev = V5Windows(a.recipe / 'manifest.json', 'dev')
    rng = np.random.default_rng(a.seed)
    targets, scanned, sentinel, highest = [], [], None, -1
    for sid in train.meta['source_ids']:
        dataset = dev if str(sid) in dev.pools else train
        pool = dataset.pools[str(sid)]
        candidates = []
        for index in rng.choice(pool, min(16, len(pool)), replace=False):
            sample = dataset[int(index)]
            depth = grid_depth(sample, config)
            row = dict(source_id=sid, split=dataset.split, index=int(index),
                sample_id=sample['sample_id'].tolist(), depth=depth)
            scanned.append(row)
            candidates.append((depth, sample, row))
            if depth > highest:
                highest, sentinel = depth, sample
        selected = min(candidates, key=lambda x: x[0])
        targets.append(selected[1])
    panel = targets + [train[i] for i in train.draw(128-len(targets), a.seed, 0, 0)]
    payload = {k: np.stack([s[k] for s in panel]) for k in panel[0]}
    payload.update({'sentinel__' + k: v for k, v in sentinel.items()})
    np.savez_compressed(a.output / 'panel.npz', **payload)
    shutil.copyfile(a.checkpoint, a.output / 'checkpoint.pt')
    shutil.copyfile(a.recipe / 'stats.json', a.output / 'stats.json')
    (a.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    result = dict(status='COMPLETED', git_commit=a.git_commit, seed=a.seed,
        checkpoint=str(a.checkpoint), checkpoint_bytes=a.checkpoint.stat().st_size,
        training_git_commit='6334e1ea', target_count=len(targets), panel_size=len(panel),
        target_ids=[s['sample_id'].tolist() for s in targets],
        target_sources=[int(s['source_id']) for s in targets],
        target_depths=[grid_depth(s, config) for s in targets],
        panel_depths=[grid_depth(s, config) for s in panel],
        sentinel_id=sentinel['sample_id'].tolist(), sentinel_depth=highest,
        scan=scanned, elapsed_s=time.monotonic()-started,
        limitation='Depth-stress panel: minimum-depth target from16/source, not an accuracy estimator; source8 uses TRAIN because inherited DEV is empty.')
    (a.output / 'export.json').write_text(json.dumps(result, indent=2) + '\n')
    for path in a.output.iterdir():
        path.chmod(0o664)
    print(json.dumps({k: v for k, v in result.items() if k != 'scan'}), flush=True)


if __name__ == '__main__':
    main()
