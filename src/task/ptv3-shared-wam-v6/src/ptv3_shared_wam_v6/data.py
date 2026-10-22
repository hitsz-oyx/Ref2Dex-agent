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
from .events import apply_event_input_policy, validate_event_input_policy
from .goal_policy import apply_goal_input_policy, validate_goal_input_policy, UNKNOWN_GOAL_POLICY


class V6Windows(V5Windows):
    def __init__(self, manifest, split, cache_size=8, geometry_gt_object=False,
                 event_input_policy=None, curated_manifest=None, goal_input_policy=None):
        self.goal_input_policy=validate_goal_input_policy(goal_input_policy)
        self.event_input_policy=validate_event_input_policy(event_input_policy)
        self.geometry_gt_object = bool(geometry_gt_object)
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
        self.curated_manifest=None
        if curated_manifest is not None:
            from .events import UNKNOWN_EVENT_POLICY
            if self.event_input_policy!=UNKNOWN_EVENT_POLICY:
                raise ValueError('curated tau reader requires the frozen unknown-event policy')
            from .curated_training import bind_curated_index
            bind_curated_index(self,curated_manifest)

    def __getstate__(self):
        return dict(manifest=str(self.recipe_path), split=self.split, cache_size=self.cache_size,
                    geometry_gt_object=self.geometry_gt_object,event_input_policy=self.event_input_policy,
                    goal_input_policy=self.goal_input_policy,
                    curated_manifest=str(self.curated_manifest) if self.curated_manifest else None)

    def __setstate__(self, state): self.__init__(**state)

    def __getitem__(self, index):
        if self.curated_manifest is not None and self.goal_input_policy != UNKNOWN_GOAL_POLICY:
            # There is no certified scheduler cache bound to this reader yet.
            # Index/sampler preflight is available without consuming an update;
            # never expose the inherited future endpoint as a waypoint input.
            raise ValueError('scheduler waypoint cache binding absent for curated tau; '
                             'refusing realized-endpoint fallback before training')
        index, drop = index if isinstance(index, tuple) else (int(index), False)
        sample = super().__getitem__(index)
        sample = apply_goal_input_policy(sample, self.goal_input_policy)
        seq_id, slot, tick = map(int, self.index[int(index)])
        if self.geometry_gt_object:
            from .geometry_gt_object import dense_object_future_gt
            sequence, _ = self._sequence(seq_id)
            sample['geometry_object_future_gt'] = dense_object_future_gt(sample, sequence, tick, slot)
        if self.event_input_policy is not None:
            sample=apply_event_input_policy(sample,self.event_input_policy)
            sample.update(event_raw_invalid=np.bool_(True),event_condition_dropout=np.bool_(False))
            return sample
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
        raw_valid=bool(values['event_valid'][tick,slot])
        valid = raw_valid and not drop
        sample.update(event_type=np.int64(values['event_type'][tick, slot] if valid else 0),
                      event_delta_steps=np.float32(values['event_delta_steps'][tick, slot] if valid else 0),
                      event_valid=np.bool_(valid), event_hz=np.int64(30),
                      event_raw_invalid=np.bool_(not raw_valid),event_condition_dropout=np.bool_(drop))
        return sample

    def draw(self, count, seed, step=0, rank=0):
        if self.curated_manifest is not None:
            from .curated_training import draw_curated
            return draw_curated(self,count,seed,step,rank)
        indices = super().draw(count, seed, step, rank)
        if self.event_input_policy is not None:
            return [(i,False) for i in indices]
        # Independent stream: adding dropout must not perturb original source,
        # stratum or window choices. No DataLoader-worker RNG state to recover.
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank, 19281]))
        return [(i, bool(drop)) for i, drop in zip(indices, rng.random(count) < self.event_dropout)]
