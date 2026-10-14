"""Whole-sequence robot windows from freshly captured multi-object RL executions."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from .data import point_labels
from .robot import features, ACTIVE, actual_consequence_sample, chunk_windows

CORPUS_SCHEMA = 'ref2dex.rl660-corpus.v1'


def tracking_windows(data, limit=.1):
    """Require current and all 24 future measured object positions near the plan."""
    error = np.linalg.norm(data['obj'][:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
    count = len(error)-24
    peak = np.maximum.reduce([error[offset:offset+count] for offset in range(25)])
    return np.flatnonzero(np.isfinite(peak) & (peak<=limit))


class RLRobotWindows(torch.utils.data.Dataset):
    def __init__(self, source, split, point_count=32):
        self.root = Path(source).resolve()
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if (self.manifest.get('schema') != CORPUS_SCHEMA or self.manifest.get('status') != 'COMPLETED'
                or not self.manifest.get('training_allowed') or self.manifest.get('fps') != 30
                or split not in ('train','val')):
            raise ValueError('frozen audited whole-sequence RL corpus required')
        all_records = self.manifest['records']
        names = [r['sequence'] for r in all_records]
        if len(names) != len(set(names)) or len(names) != self.manifest['num_sequences']:
            raise ValueError('duplicate or incomplete sequence membership')
        self.records = [r for r in all_records if r['split'] == split]
        if not self.records:
            raise ValueError('nonempty train/dev sequence panels required')
        self.split,self.point_count = split,point_count
        self.controller = self.manifest['native_controller']
        self.limit = self.manifest.get('max_reference_error_m',.1)
        self.keys,self.ticks_by_sequence,self.pools = [],[],[]
        for seq,record in enumerate(self.records):
            ticks = record.get('admitted_ticks')
            if ticks is None:
                ticks = tracking_windows(self.sequence(seq)[0],self.limit)
            ticks = np.asarray(ticks,dtype=np.int64)
            if (np.any(np.diff(ticks)<=0) or (len(ticks) and
                    (ticks.min()<0 or ticks.max()>=record['frames']-24))):
                raise ValueError('invalid admitted window indices')
            start = len(self.keys)
            self.keys.extend((int(tick),seq) for tick in ticks)
            self.ticks_by_sequence.append(ticks)
            self.pools.append(np.arange(start,len(self.keys)))
        if not self.keys:
            raise ValueError('no tracking-qualified windows in sequence split')
        self.phase_pools = []
        for startup in (True,False):
            pools = [pool[(ticks<16) if startup else (ticks>=16)]
                     for ticks,pool in zip(self.ticks_by_sequence,self.pools)]
            self.phase_pools.append([p for p in pools if len(p)] or [p for p in self.pools if len(p)])
        if any(r['frames'] <= 40 for r in self.records):
            raise ValueError('complete startup and future windows required')

    def __len__(self):
        return len(self.keys)

    @lru_cache(maxsize=16)
    def sequence(self, seq):
        record = self.records[seq]
        path = (self.root/record['path']).resolve()
        if self.root not in path.parents:
            raise ValueError('sequence artifact path escapes corpus')
        with np.load(path/'trajectory.npz',allow_pickle=False) as f:
            data = {k:f[k].copy() for k in f.files}
        length = record['frames']
        shapes = dict(q=(length,18),dq=(length,18),hand=(length,11,3),obj=(length,4,4),
            velocity=(length,6),reference_obj=(length,4,4),action=(length-1,18),
            applied=(length-1,18),pd_targets=(length-1,18))
        if (any(data[k].shape != shape or not np.isfinite(data[k]).all() for k,shape in shapes.items())
                or not np.array_equal(data['action'],data['applied'])
                or np.abs(data['applied']).max() > 1.000001):
            raise ValueError('same-execution complete measured states/applied commands required')
        if 'admitted_ticks' in record and not np.array_equal(
                record['admitted_ticks'],tracking_windows(data,self.limit)):
            raise ValueError('frozen window quality indices differ from measured execution')
        with np.load(path/'canonical.npz',allow_pickle=False) as f:
            canonical = {k:f[k].copy() for k in f.files}
        return data,canonical

    def __getitem__(self, index):
        tick,seq = self.keys[index]
        data,canonical = self.sequence(seq)
        sample = actual_consequence_sample(data['obj'][:,None],data['hand'][:,None],tick,0,
            data['reference_obj'][tick+24],canonical,sample_id=index)
        future = sample['action'].reshape(24,2,11,9)[...,:3]
        sample['trajectory_target'] = future-sample['start_hand'][None]
        sample['trajectory_valid'] = sample['action_valid'].reshape(24,2,11)
        state,tau = features(*(data[k][tick] for k in ('q','dq','hand','obj','velocity')),
                             data['hand'][tick+np.arange(1,25)])
        sample.update(point_labels(sample,self.point_count))
        sample.update(native_state=state,native_tau=tau,
            native_action=data['applied'][tick+np.arange(8)][:,list(ACTIVE)].astype(np.float32),
            has_action=np.bool_(True),source_id=np.int64(3))
        return sample

    def sample_indices(self, rng, batch_size):
        """Uniform sequence sampling; half startup and half later, as robot V0."""
        indices = []
        for row in range(batch_size):
            eligible = self.phase_pools[0 if row < batch_size//2 else 1]
            pool = eligible[int(rng.integers(len(eligible)))]
            indices.append(int(rng.choice(pool)))
        return indices

    def normalization(self):
        if self.split != 'train':
            raise ValueError('retargeter normalizers use TRAIN sequences only')
        arrays = [[],[],[]]
        for seq in range(len(self.records)):
            ticks = self.ticks_by_sequence[seq]
            if not len(ticks):
                continue
            data,_ = self.sequence(seq)
            expanded = {k:data[k][:,None] for k in ('q','dq','hand','obj','velocity','action','applied')}
            for values,array in zip(arrays,chunk_windows(expanded,[0])):
                values.append(array[ticks])
        return [np.concatenate(values) for values in arrays]
