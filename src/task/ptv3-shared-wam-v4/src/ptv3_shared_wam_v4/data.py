"""Nine-source v4 reader with a target-only 512/32/128 tensor boundary."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader

from pose_mesh_data.v4_data import V4Main3Windows, V4NativeWindows, v4_collate
from pose_mesh_data.native_recipe import source_pools
from .paths import SOURCE_IDS
from .quality_admission import excluded_indices
from oakink_wm.multisource import source_draw


class NineWindows(Dataset):
    def __init__(self, main3, native, split, recipe=None):
        self.split = split
        self.main = V4Main3Windows(main3, 'train' if split == 'train' else 'val')
        self.native = V4NativeWindows(native, split, engineering=True)
        self.pools = source_pools(self.native.base)
        self.counts = {str(i): len(s) for i, s in enumerate(self.main.base.sources)}
        self.counts.update({s: len(p) for s, p in self.pools.items()})
        self.blocked = np.zeros(len(self.main) + len(self.native), bool)
        excluded = excluded_indices(recipe, split, len(self.blocked))
        self.blocked[excluded] = True
        if recipe and recipe.get('window_filter_manifest'):
            path = Path(recipe['window_filter_manifest'])
            manifest = __import__('json').loads(path.read_text())
            if (manifest.get('schema') != 'ref2dex.wam-v4-window-filter.v1'
                    or manifest.get('split') != split
                    or manifest.get('total_windows') != len(self.blocked)
                    or manifest.get('source_roots') != [str(Path(main3).resolve()), str(self.native.base.root)]):
                raise ValueError('v4 window filter identity mismatch')
            filtered = np.asarray(manifest.get('blocked_indices', []), dtype=np.int64)
            if (filtered.ndim != 1 or (len(filtered) and
                    (filtered[0] < 0 or filtered[-1] >= len(self.blocked) or (np.diff(filtered) <= 0).any()))):
                raise ValueError('sorted unique v4 window filter required')
            self.blocked[filtered] = True
        if len(excluded):
            for sid, source in enumerate(self.main.base.sources):
                offset = self.main.base.offsets[sid]
                mask = self.blocked[offset:offset + len(source)]
                source.groups = [g[~mask[g]] for g in source.groups]
                self.counts[str(sid)] -= int(mask.sum())
            for sid, pool in self.pools.items():
                self.pools[sid] = pool[~self.blocked[len(self.main) + pool]]
                self.counts[sid] = len(self.pools[sid])
            if any(n <= 0 for n in self.counts.values()):
                raise ValueError('quality filter emptied a source')
        if split == 'train':
            if set(map(int, self.counts)) != set(SOURCE_IDS):
                raise ValueError('all nine verified TRAIN sources required')
            if recipe is not None:
                if (recipe['source_counts'] != self.counts
                        or recipe['main3_root'] != str(Path(main3).resolve())
                        or recipe['native_root'] != str(self.native.base.root)):
                    raise ValueError('nine-source recipe/corpus mismatch')
                if recipe['native_index_stamp'] != list(self.native.base._index_stamp):
                    raise ValueError('native frozen index changed')
                if recipe['main3_identity'] != self.main.base.meta.get('normalization_source_manifest_sha256'):
                    raise ValueError('main-three identity changed')
        self.recipe = recipe

    def __len__(self):
        return len(self.main) + len(self.native)

    def __getitem__(self, index):
        if self.blocked[int(index)]:
            raise ValueError('quality-quarantined window')
        if index < len(self.main):
            sample = self.main[int(index)]
            provenance = dict(source_id=int(sample['source_id']), sample_id=np.asarray(sample['sample_id']).tolist(),
                              root=str(self.main.base.root), split=self.split, profile='main3-target-v4.v1')
            return sample, provenance, None
        sample = dict(self.native[int(index) - len(self.main)])
        provenance = sample.pop('provenance')
        return sample, provenance, sample

    def draw_source(self, sid, count, rng):
        if int(sid) < 3:
            return self.main.base.offsets[int(sid)] + source_draw(self.main.base.sources[int(sid)], count, rng)
        return len(self.main) + rng.choice(self.pools[sid], count)

    def draw(self, count, seed, step=0, rank=0):
        if self.split != 'train' or self.recipe is None:
            raise ValueError('TRAIN recipe required for optimizer sampling')
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank]))
        sources = list(self.recipe['source_probabilities'])
        selected = rng.choice(sources, count, p=[self.recipe['source_probabilities'][s] for s in sources])
        result = np.empty(count, np.int64)
        for sid in sources:
            mask = selected == sid
            if mask.any():
                result[mask] = self.draw_source(sid, int(mask.sum()), rng)
        return result.tolist()

    def panel(self, per_source, seed):
        rng = np.random.default_rng(seed)
        return {sid: self.draw_source(sid, per_source, rng).tolist() for sid in sorted(self.counts, key=int)}


def collate(samples):
    return dict(model_inputs=v4_collate([s[0] for s in samples]), provenance=[s[1] for s in samples])


class Steps:
    def __init__(self, dataset, count, seed, rank, first, last):
        self.dataset, self.count, self.seed, self.rank = dataset, count, seed, rank
        self.first, self.last = first, last

    def __iter__(self):
        for step in range(self.first, self.last + 1):
            yield self.dataset.draw(self.count, self.seed, step, self.rank)

    def __len__(self):
        return max(0, self.last - self.first + 1)


def initialize_worker(worker_id):
    import torch
    from torch.utils.data import get_worker_info
    from ptv3_shared_wam.loading import project_ipc_directory
    torch.set_num_threads(1)
    project_ipc_directory()
    dataset = get_worker_info().dataset
    for source in dataset.main.base.sources:
        source.sequence = lru_cache(maxsize=16)(source.sequence)


def loader(dataset, config, rank, first, last, workers=2):
    import torch
    from ptv3_shared_wam.loading import project_ipc_directory
    project_ipc_directory()
    kwargs = dict(dataset=dataset, batch_sampler=Steps(dataset, config['batch_size'], config['seed'], rank, first, last),
                  collate_fn=collate, num_workers=workers, pin_memory=True,
                  generator=torch.Generator().manual_seed(config['seed'] * 1009 + rank))
    if workers:
        kwargs.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2,
                      worker_init_fn=initialize_worker, timeout=180)
    return DataLoader(**kwargs)
