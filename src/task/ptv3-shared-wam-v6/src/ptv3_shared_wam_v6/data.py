"""Admitted v6 events over unchanged v5 windows; deterministic condition dropout."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from . import paths
from ptv3_shared_wam_v5.data import V5Windows, collate
from .recipe import validate_recipe
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.storage import read_json, load_array
from ptv3_shared_wam_v5.event_labels import EVENT_LABEL_SCHEMA


class V6Windows(V5Windows):
    def __init__(self, manifest, split, cache_size=8):
        self.recipe_path = Path(manifest).resolve()
        self.recipe = validate_recipe(self.recipe_path)
        event_manifest = self.recipe['event_manifest']
        event_dropout = self.recipe['event_dropout']
        super().__init__(self.recipe['parent_manifest'], split, cache_size)
        self.event_manifest = Path(event_manifest).resolve()
        events = read_json(self.event_manifest)
        if (events.get('schema') != EVENT_LABEL_SCHEMA or events.get('status') != 'COMPLETED'
                or events.get('parent_manifest_identity') != metadata_identity(self.path)
                or len(events['sequences']) != len(self.meta['sequences'])):
            raise ValueError('complete event annotations bound to this frozen parent required')
        if not 0 <= event_dropout < 1:
            raise ValueError('bounded event condition dropout required')
        self.event_dropout = float(event_dropout)
        self.event_rows = {r['key']: r for r in events['sequences']}
        if set(self.event_rows) != {r['key'] for r in self.meta['sequences']}:
            raise ValueError('event sequence keys must match parent exactly')
        self.event_cache = OrderedDict()

    def __getstate__(self):
        return dict(manifest=str(self.recipe_path), split=self.split, cache_size=self.cache_size)

    def __setstate__(self, state): self.__init__(**state)

    def __getitem__(self, index):
        index, drop = index if isinstance(index, tuple) else (int(index), False)
        sample = super().__getitem__(index)
        seq_id, slot, tick = map(int, self.index[int(index)])
        parent = self.meta['sequences'][seq_id]; key = parent['key']
        if key not in self.event_cache:
            row = self.event_rows[key]; folder = Path(row['path'])
            meta = read_json(folder/'meta.json')
            if (metadata_identity(folder/'meta.json') != row['event_meta_identity']
                    or meta['contact_meta_identity'] != parent['contact_meta_identity']
                    or meta['objects'] != parent['objects']):
                raise ValueError('event annotation order/upstream changed')
            arrays = {k: load_array(folder/(k+'.npy'), mmap_mode='r')
                      for k in ('event_type', 'event_delta_steps', 'event_valid')}
            if any(list(v.shape) != meta['shape'] for v in arrays.values()):
                raise ValueError('event array shape changed')
            self.event_cache[key] = arrays
            while len(self.event_cache) > self.cache_size: self.event_cache.popitem(last=False)
        self.event_cache.move_to_end(key); values = self.event_cache[key]
        valid = bool(values['event_valid'][tick, slot]) and not drop
        sample.update(event_type=np.int64(values['event_type'][tick, slot] if valid else 0),
                      event_delta_steps=np.float32(values['event_delta_steps'][tick, slot] if valid else 0),
                      event_valid=np.bool_(valid), event_hz=np.int64(30))
        return sample

    def draw(self, count, seed, step=0, rank=0):
        indices = super().draw(count, seed, step, rank)
        # Independent stream: adding dropout must not perturb original source,
        # stratum or window choices. No DataLoader-worker RNG state to recover.
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank, 19281]))
        return [(i, bool(drop)) for i, drop in zip(indices, rng.random(count) < self.event_dropout)]
