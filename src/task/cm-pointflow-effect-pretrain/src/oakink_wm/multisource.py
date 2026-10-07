"""Source-balanced temporal datasets with explicit per-source category mapping."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .data import Windows

SOURCE_NAMES = ('oakink2', 'grab', 'arctic', 'contactpose')
SOURCE_WEIGHTS = (.5, .2, .2, .1)
MAIN_SOURCE_NAMES = SOURCE_NAMES[:3]
MAIN_SOURCE_WEIGHTS = (5/9, 2/9, 2/9)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class SourceWindows(Windows):
    def __init__(self, descriptor, split):
        self.root = Path(descriptor['root'])
        manifest = self.root/'processed/manifest.json'
        if sha(manifest) != descriptor['manifest_sha256']:
            raise ValueError('source manifest drift: '+descriptor['name'])
        self.meta = json.loads(manifest.read_text())
        if (self.meta['status'] != 'COMPLETED' or self.meta['fps'] != 30
                or self.meta['history'] != 4 or self.meta['horizon'] != 24):
            raise ValueError('source temporal contract mismatch')
        self.kind = descriptor['kind']
        if self.kind == 'native' and (not self.meta.get('training_allowed')
                or self.meta.get('schema') != 'ref2dex.native-wm30.v1'
                or self.meta.get('units') != 'm' or self.meta.get('hand_order') != ['right', 'left']):
            raise ValueError('native source is not eligible for training')
        index = descriptor['indices'][split]
        if sha(index['path']) != index['sha256']:
            raise ValueError('mixed index drift')
        self.rows = np.load(index['path']).copy()
        self.sequences = self.meta['sequences']
        if self.rows.ndim != 2 or self.rows.shape[1] != 4 or self.rows.dtype.kind not in 'iu':
            raise ValueError('invalid source index')
        if len(self.rows) and (self.rows[:, 0].min() < 0 or self.rows[:, 0].max() >= len(self.sequences)):
            raise ValueError('source sequence index out of bounds')
        if self.kind == 'native':
            if not np.isin(self.rows[:, 3], [1, 2]).all():
                raise ValueError('unknown native motion category')
            # Native1=moving,2=near-static; OakInk0=moving. No program labels invented.
            self.rows[:, 3] -= 1
        elif self.kind != 'oakink2' or not np.isin(self.rows[:, 3], [0, 1, 2]).all():
            raise ValueError('unknown source/category contract')
        self.groups = [np.flatnonzero(self.rows[:, 3] == k) for k in range(3)]

    @lru_cache(maxsize=32)
    def sequence(self, seq):
        data = dict(super().sequence(seq))
        if self.kind == 'native':
            timestamps = np.load(self.root/'processed/sequences'/seq/'timestamps.npy', mmap_mode='r')
            if not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), 1/30, atol=1e-5, rtol=0):
                raise ValueError('native timestamps are not contiguous 30Hz')
            data['frame_ids'] = 4*np.arange(len(timestamps), dtype=np.int64)
        return data


class MixedWindows(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.meta = json.loads((self.root/'processed/manifest.json').read_text())
        if self.meta.get('schema') != 'pointworld-multisource.wm30.v1' or self.meta.get('status') != 'COMPLETED':
            raise ValueError('mixed corpus not frozen')
        if (self.meta.get('fps'), self.meta.get('history'), self.meta.get('horizon')) != (30, 4, 24):
            raise ValueError('mixed temporal contract mismatch')
        names = tuple(d['name'] for d in self.meta['sources'])
        if names not in (SOURCE_NAMES, MAIN_SOURCE_NAMES):
            raise ValueError('expected registered four-source or main three-source corpus')
        if names == MAIN_SOURCE_NAMES and self.meta.get('supervision') != 'measured_dynamic_hand_main':
            raise ValueError('three-source main supervision must exclude rigid-transport auxiliary')
        self.sources = [SourceWindows(d, split) for d in self.meta['sources']]
        self.offsets = np.cumsum([0]+[len(s) for s in self.sources])
        self.sequence_offsets = np.cumsum([0]+[len(s.sequences) for s in self.sources])
        self.source_weights = np.array([d['weight'] for d in self.meta['sources']], dtype=float)
        if not np.isfinite(self.source_weights).all() or (self.source_weights <= 0).any():
            raise ValueError('invalid source probabilities')
        self.source_weights /= self.source_weights.sum()
        if split in ('train', 'val') and any(not len(s) or not len(s.groups[0]) for s in self.sources):
            raise ValueError('every fitted/evaluated source needs real moving windows')
        rows = []
        for source, seq_offset in zip(self.sources, self.sequence_offsets):
            r = source.rows.copy(); r[:, 0] += seq_offset; rows.append(r)
        self.rows = np.concatenate(rows)
        self.groups = [np.flatnonzero(self.rows[:, 3] == k) for k in range(3)]
        if not len(self): raise ValueError('empty mixed split')

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, item):
        if not 0 <= item < len(self): raise IndexError(item)
        source_id = int(np.searchsorted(self.offsets, item, side='right')-1)
        sample = self.sources[source_id][int(item-self.offsets[source_id])]
        sample['sample_id'][0] += self.sequence_offsets[source_id]
        return sample


def source_draw(source, count, rng):
    # Renormalize only over actual strata; a missing program/background stratum
    # must not be fabricated to satisfy the old OakInk sampler.
    probabilities = np.array([.6, .2, .2])*np.array([bool(len(g)) for g in source.groups])
    if probabilities.sum() == 0: raise ValueError('empty source')
    categories = rng.choice(3, count, p=probabilities/probabilities.sum())
    out = np.empty(count, dtype=np.int64)
    for k, pool in enumerate(source.groups):
        chosen = categories == k
        if chosen.any(): out[chosen] = rng.choice(pool, int(chosen.sum()))
    return out


def mixed_indices(dataset, count, seed, equal_sources=False):
    rng = np.random.default_rng(seed)
    n_sources = len(dataset.sources)
    if equal_sources:
        if count % n_sources: raise ValueError('validation count must divide into equal source panels')
        sources = np.repeat(np.arange(n_sources), count//n_sources)
        rng.shuffle(sources)
    else:
        sources = rng.choice(n_sources, count, p=dataset.source_weights)
    out = np.empty(count, dtype=np.int64)
    for source_id, source in enumerate(dataset.sources):
        mask = sources == source_id
        if mask.any(): out[mask] = dataset.offsets[source_id]+source_draw(source, int(mask.sum()), rng)
    return out


def validate_pretrained(state, model, config, identity, expected_model_sources):
    """New mixed corpus, same coordinate/model/normalization contract; model only."""
    previous = state['identity']
    parent_data_matches = state['dataset_hash'] == identity['normalization_source_manifest_sha256']
    if previous.get('mixed_data'):
        old = previous.get('source_manifest', {})
        new = identity.get('source_manifest', {})
        old_sources = {d['name']: d for d in old.get('sources', [])}
        new_names = tuple(d['name'] for d in new.get('sources', []))
        parent_data_matches = (
            state['dataset_hash'] == previous.get('dataset_hash')
            and previous.get('normalization_source_manifest_sha256') == identity['normalization_source_manifest_sha256']
            and tuple(d['name'] for d in old.get('sources', [])) in (SOURCE_NAMES, MAIN_SOURCE_NAMES)
            and new_names == MAIN_SOURCE_NAMES
            and old.get('normalization_stats_sha256') == new.get('normalization_stats_sha256')
            and all(d['name'] in old_sources and d['kind'] == old_sources[d['name']]['kind']
                    and d['manifest_sha256'] == old_sources[d['name']]['manifest_sha256']
                    and all(d['indices'][split]['sha256'] == old_sources[d['name']]['indices'][split]['sha256']
                            for split in ('train', 'val', 'test')) for d in new.get('sources', [])))
    if (not parent_data_matches
            or previous['stats_sha256'] != identity['stats_sha256'] or previous['arm'] != 'action'
            or identity['arm'] != 'action' or previous['vendor_sources'] != identity['vendor_sources']
            or any(previous['implementation_sources'].get(k) != v for k, v in expected_model_sources.items())):
        raise ValueError('mixed pretraining model/normalization/provenance mismatch')
    recipe = {'seed', 'microbatch', 'accumulation', 'updates', 'learning_rate', 'warmup_updates',
              'learning_rate_schedule', 'validation_samples', 'validation_interval', 'checkpoint_interval',
              'group_seconds', 'workers', 'validation_seed', 'natural_validation_seed', 'validation_microbatch'}
    for key in set(config) | set(state['config']):
        if key not in recipe and config.get(key) != state['config'].get(key):
            raise ValueError('mixed initialization changes model/data semantics: '+key)
    model.load_state_dict(state['model'], strict=True)
    return dict(parent_step=state['step'], weights_only=True, optimizer_reset=True,
                schedule_reset=True, draw_reset=True, mixed_data=True,
                normalization='preserve parent input/output statistics; physical loss scales may be separate train-only statistics')
