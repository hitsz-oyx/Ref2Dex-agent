"""Freeze the independently audited local subset; retain pretrained output scales."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.contracts import DATA_SCHEMA
from ptv3_shared_wam_v6.events import EVENT_ENCODING
from ptv3_shared_wam_v6.local_subset import LOCAL_SCOPE, index_identity
from ptv3_shared_wam_v6.recipe import validate_recipe
from ptv3_shared_wam_v6.data import V6Windows
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.contracts import STATS_SCHEMA


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-root', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--git-commit', required=True)
    a = parser.parse_args()
    if shutil.disk_usage(a.output.parent).free < 25*2**30:
        raise ValueError('reserve >=20GiB plus preparation/training checkpoint headroom')
    root = a.prepared_root.resolve()
    audit = json.loads((root/'integrity-audit.json').read_text())
    original = root/'data/recipe/manifest.json'
    parent = json.loads(original.read_text())
    if (audit.get('status') != 'PASS' or audit.get('checked_sequences') != len(parent['sequences'])
            or len(parent['sequences']) != 1177 or parent.get('training_allowed') is not False):
        raise ValueError('independently verified 1177-sequence preparation required')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if saved['step'] != 23758:
        raise ValueError('audited local step23758 initialization required')
    a.output.mkdir(exist_ok=False)
    out = a.output.resolve()
    (out/'parent').mkdir()
    for split in ('train', 'dev'):
        for field, suffix in (('indices', ''), ('strata', '-strata')):
            target = out/'parent'/(split+suffix+'.npy')
            shutil.copyfile(parent[field][split], target)
            parent[field][split] = str(target)
    train = np.load(parent['indices']['train'])
    sequence_sources = np.array([r['source_id'] for r in parent['sequences']])
    counts = {str(s): int((sequence_sources[train[:, 0]] == s).sum()) for s in parent['source_ids']}
    parent.update(training_allowed=True, local_subset_scope=LOCAL_SCOPE,
                  source_probabilities={s: n/len(train) for s, n in counts.items()},
                  subset_sampling='source weight proportional to admitted TRAIN windows; original within-source strata',
                  source_engineering_manifest=str(original), source_engineering_identity=metadata_identity(original),
                  git_commit=a.git_commit)
    # Keep engineering_subset=True and pending_sequences=38349 visible. This is
    # admission of this fixed subset, never promotion to full-nine-source data.
    write(out/'parent/manifest.json', parent)
    events = json.loads((root/'data/events/manifest.json').read_text())
    events.update(parent_manifest_identity=metadata_identity(out/'parent/manifest.json'),
                  training_allowed=True, local_subset_scope=LOCAL_SCOPE)
    write(out/'events.json', events)
    stats = dict(schema=STATS_SCHEMA, split='train', units='m', fps=30, history=4, horizon=24,
                 hand_points=21, source_ids=parent['source_ids'],
                 method='checkpoint-normalizer-buffers; inherited TRAIN scales; no moments refit',
                 initialization_checkpoint=str(a.checkpoint.resolve()), source_step=saved['step'],
                 source_data_identity=saved['data_identity'],
                 limitation='Reconstituted numerical scales, not the unavailable original stats JSON or its admission metadata.')
    for name in ('hand', 'scene'):
        stats[name] = {k: saved['model']['normalizer.'+name+'_'+k].tolist() for k in ('mean', 'std')}
    write(out/'stats.json', stats)
    recipe = dict(schema=DATA_SCHEMA, status='FROZEN', training_allowed=True,
                  local_subset_scope=LOCAL_SCOPE, engineering_subset=True,
                  initialization_checkpoint=str(a.checkpoint.resolve()),
                  event_encoding=EVENT_ENCODING, event_dropout=.15, source_ids=parent['source_ids'],
                  windows={s: len(np.load(parent['indices'][s])) for s in ('train', 'dev')},
                  index_identities={s: index_identity(parent['indices'][s]) for s in ('train', 'dev')},
                  strata_identities={s: index_identity(parent['strata'][s]) for s in ('train', 'dev')},
                  git_commit=a.git_commit,
                  scope='Independent GRAB-dominant local adaptation; not strict remote continuation',
                  primary_source=1,
                  limitation='Existing teacher-forced tau/contact objective; no new coupling loss; not a free-rollout cure or robot success claim.')
    for key, path in (('parent_manifest', out/'parent/manifest.json'), ('event_manifest', out/'events.json'), ('stats', out/'stats.json')):
        recipe[key] = str(path)
        recipe[key+'_identity'] = metadata_identity(path)
    write(out/'manifest.json', recipe)
    config = dict(saved['config'], local_subset_scope=LOCAL_SCOPE,
                  seed=170, validation_seed=171, batch_size=64, updates=2000,
                  geometry_lr=1e-5, shared_lr=1e-5, warmup_updates=50,
                  max_seconds=14400, artifact_budget_bytes=8*2**30,
                  checkpoint_keep_recent=2, checkpoint_interval=200, checkpoint_milestone_interval=0,
                  validation_samples=32, validation_microbatch_size=4, validation_interval=200,
                  validation_initial_step=8, validation_before_training=True, log_interval=10)
    write(out/'config.json', config)
    validate_recipe(out/'manifest.json', out/'stats.json')
    checked = 0
    for split in ('train', 'dev'):
        ds = V6Windows(out/'manifest.json', split)
        for seq_id in np.unique(ds.index[:, 0]):
            indices = np.flatnonzero(ds.index[:, 0] == seq_id)
            for index in set([int(indices[0]), int(indices[-1])]):
                row = ds[index]
                for name, value in row.items():
                    if np.asarray(value).dtype.kind == 'f' and not np.isfinite(value).all():
                        raise ValueError('nonfinite actual V6 reader output: '+name)
                checked += 1
    write(out/'preflight.json', dict(status='PASS', checked_reader_windows=checked,
          sequence_split_overlap=0, sequences=len(parent['sequences']), windows=recipe['windows'],
          source_train_windows=counts, initialization_step=saved['step'],
          artifact_budget_gib=8, max_seconds=14400, max_updates=2000,
          original_remote_continuation=False, git_commit=a.git_commit))
    print(json.dumps(dict(status='PASS', output=str(out), checked_reader_windows=checked,
                         windows=recipe['windows'], source_train_windows=counts)), flush=True)


if __name__ == '__main__':
    main()
