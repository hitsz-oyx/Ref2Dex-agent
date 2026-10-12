"""Admitted actual robot futures mixed with the existing three-domain corpus."""
import json
from pathlib import Path

import numpy as np

from .retargeter_data import ACTIVE, features
from .wam_consequence import actual_consequence_sample


class EffectData:
    def __init__(self, root, main):
        self.main = main
        owner = Path(root) / 'outputs/trajectory-policy'
        fit = owner / 'wam-v2-control-fit-20261012-r1'
        gt = owner / 'wam-v2-control-GT-20261012-r1'
        if not json.loads((gt/'result.json').read_text())['GT_execution_gate']:
            raise ValueError('qualified Chunk-R required')
        m = json.loads((fit/'manifest.json').read_text())
        self.source = Path(m['source'])
        recovery = json.loads((Path(m['recovery_aggregation'])/'manifest.json').read_text())
        self.paths = {'nominal': self.source}
        self.takeovers = {}
        for i, entry in enumerate(recovery['admissions']):
            name = 'recovery%d' % i
            self.paths[name] = Path(entry['native'])
            self.takeovers[name] = int(entry['takeover'])
        self.data = {}
        for name, path in self.paths.items():
            with np.load(path/'trajectory.npz', allow_pickle=False) as f:
                self.data[name] = {k: f[k].copy() for k in f.files}
        nominal = self.data['nominal']
        if (not json.loads((self.source/'manifest.json').read_text()).get('training_allowed')
                or not np.array_equal(nominal['action'], nominal['applied'])):
            raise ValueError('actual applied nominal teacher controls required')
        self.controller = json.loads((self.source/'manifest.json').read_text())['native_controller']
        with np.load(owner/'competent-reference-20261011-r1/reference.npz', allow_pickle=False) as f:
            self.reference = f['obj'].copy()
        with np.load(Path(root)/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz', allow_pickle=False) as f:
            self.canonical = {k: f[k].copy() for k in f.files}

    def __getitem__(self, key):
        kind, tick, role = key
        if kind == 'main':
            sample = dict(self.main[int(tick)])
            sample.update(has_pd=False, native_state=np.zeros(87, np.float32),
                          native_action=np.zeros((8, 12), np.float32))
            return sample
        tick, role = int(tick), int(role)
        d = self.data[kind]
        if kind == 'nominal':
            if not 0 <= role < 28 or not 0 <= tick <= 518:
                raise ValueError('nominal teacher window outside admitted range')
        elif (role != 3 or not self.takeovers[kind] <= tick <= 104
              or not d['expert_applied'][tick:tick+24, 1].all()
              or d['expert_clipped'][tick:tick+24, 1].any()):
            raise ValueError('recovery requires own actual unclipped expert future24')
        sample = actual_consequence_sample(d['obj'], d['hand'], tick, role,
                    self.reference[min(tick+24, 542)], self.canonical, sample_id=tick*32+role)
        # This source is already admitted for ACTOR expert futures. The generic
        # consequence helper deliberately grants no such admission by itself.
        future = sample['action'].reshape(24, 2, 11, 9)[..., :3]
        sample['trajectory_target'] = future - sample['start_hand'][None]
        sample['trajectory_valid'] = sample['action_valid'].reshape(24, 2, 11) & sample['start_hand_valid'][None]
        sample['trajectory_target'][~sample['trajectory_valid']] = 0
        state, _ = features(*(d[k][tick, role] for k in ('q', 'dq', 'hand', 'obj', 'velocity')),
                             d['hand'][tick+np.arange(1, 25), role])
        sample.update(has_pd=True, native_state=state,
                      native_action=d['applied'][tick+np.arange(8), role][:, list(ACTIVE)].astype(np.float32),
                      source_id=np.int64(3 if kind == 'nominal' else 4+int(kind[-1])))
        return sample


def effect_collate(samples):
    import torch
    from goal_point_wam.data import goal_collate
    batch = goal_collate(samples)
    for key in ('has_pd', 'native_state', 'native_action'):
        batch[key] = torch.as_tensor(np.stack([s[key] for s in samples]))
    return batch


class EffectBatches:
    """Stateless resumable per-rank draws: exactly60/25/15 per100 windows."""
    def __init__(self, mixed, takeovers, rank, first, last, seed=296):
        self.mixed, self.takeovers = mixed, takeovers
        self.rank, self.first, self.last, self.seed = rank, first, last, seed

    def __iter__(self):
        from oakink_wm.multisource import source_draw
        for step in range(self.first, self.last+1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, self.rank]))
            sources = rng.choice(3, 60, p=self.mixed.source_weights)
            ids = np.empty(60, np.int64)
            for sid, source in enumerate(self.mixed.sources):
                mask = sources == sid
                if mask.any():
                    ids[mask] = self.mixed.offsets[sid]+source_draw(source, int(mask.sum()), rng)
            keys = [('main', int(i), 0) for i in ids]
            keys += [('nominal', int(t), int(e)) for t, e in
                     zip(np.r_[rng.integers(0, 16, 13), rng.integers(16, 519, 12)], rng.integers(0, 16, 25))]
            for name, takeover in self.takeovers.items():
                keys += [(name, int(t), 3) for t in rng.integers(takeover, 105, 5)]
            yield keys

    def __len__(self):
        return self.last-self.first+1
