"""Prepare paired small/full controls and commands; never launch GPU work."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from shadow_finetune import paths
from shadow_finetune.data import ShadowWindows
from shadow_finetune.scale_controls import SCHEMA, normalizer_arrays, stamp, verify_nested, window_key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small-data', type=Path, required=True)
    p.add_argument('--full-data', type=Path, required=True)
    p.add_argument('--native-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent', type=Path, default=paths.PARENT)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=214)
    p.add_argument('--dev-seed', type=int, default=215)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or paths.ROOT/'outputs/shadow-finetune' not in out.parents:
        raise ValueError('fresh Task output required')
    if not 1 <= a.steps <= 2000 or not 1 <= a.batch_size <= 256:
        raise ValueError('bounded Probe: <=2000 updates and batch<=256')
    small = ShadowWindows(a.small_data, 'train')
    full = ShadowWindows(a.full_data, 'train')
    if len(full) <= len(small):
        raise ValueError('full TRAIN must be larger')
    verify_nested(dict(small.meta, _root=str(small.root)), dict(full.meta, _root=str(full.root)))
    train_objects = {r['object_code'] for r in full.records}
    dev = ShadowWindows(a.full_data, 'dev')
    if train_objects & {r['object_code'] for r in dev.records}:
        raise ValueError('TRAIN/DEV object overlap')
    panel = json.loads(a.native_panel.read_text())
    for row in panel['objects']:
        if row['object_code'] not in {r['object_code'] for r in full.meta['records'] if r['split'] == row['split']}:
            raise ValueError('native panel split or object missing')
    validation = np.random.default_rng(a.dev_seed).choice(len(dev), min(64, len(dev)), replace=False)
    # Both arms receive statistics fit ONLY from the small TRAIN set, not full TRAIN or DEV.
    normalizer = normalizer_arrays(small.normalization())
    settings = dict(parent=str(a.parent.resolve()), seed=a.seed, updates=a.steps, batch_size=a.batch_size)
    inputs = [a.small_data/'manifest.json', a.full_data/'manifest.json', a.parent, a.native_panel]
    report = dict(schema=SCHEMA, status='PREPARED',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=paths.ROOT, text=True).strip(),
                  settings=settings, training_caches=dict(small=str(a.small_data.resolve()), full=str(a.full_data.resolve())),
                  validation_cache=str(a.full_data.resolve()), validation_ids=validation.tolist(),
                  validation_keys=[window_key(dev, i) for i in validation], dev_seed=a.dev_seed,
                  normalization_source=str(a.small_data.resolve()),
                  normalization={k: v.tolist() for k, v in normalizer.items()},
                  native_panel=str(a.native_panel.resolve()), native_reset_seed=216,
                  input_stamps=[dict(path=str(f.resolve()), stamp=stamp(f)) for f in inputs],
                  counts=dict(small_train=len(small), full_train=len(full), common_dev=len(dev)),
                  comparison='fixed-compute exposure; change TRAIN support only; no claim of full-data convergence',
                  execution='prepared only; GPU authorization/resource check required at actual launch')
    out.mkdir(parents=True)
    (out/'controls.json').write_text(json.dumps(report, indent=2)+'\n')
    commands = {}
    for arm, cache in report['training_caches'].items():
        train = [sys.executable, str(TASK/'tools/run/train.py'), '--data', cache,
                         '--controls', str(out/'controls.json'), '--parent', settings['parent'],
                         '--seed', str(a.seed), '--steps', str(a.steps), '--batch-size', str(a.batch_size),
                         '--output', str(out/('train-'+arm)), '--gpu', '6']
        native = [sys.executable, str(TASK/'tools/run/evaluate.py'), '--cache', str(a.full_data.resolve()),
                  '--checkpoint', str(out/('train-'+arm)/('step%d.pt' % a.steps)),
                  '--object-manifest', str(a.native_panel.resolve()), '--seed', '216', '--envs', '16',
                  '--output', str(out/('native-final-'+arm)), '--gpu', '6']
        commands[arm] = dict(train=train, evaluate_final=native)
    (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED', counts=report['counts'], output=str(out), gpu_started=False)))


if __name__ == '__main__':
    main()
