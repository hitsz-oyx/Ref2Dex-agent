"""Only new C-final actual takeover futures supervise robot control context."""
import json
from pathlib import Path

import numpy as np

from .wam_effect_data import EffectData, effect_collate
from .robot_context import control_context


class ControlData(EffectData):
    def __init__(self, root, main, recovery_paths):
        super().__init__(root, main)
        self.paths = {'nominal': self.paths['nominal']}
        self.data = {'nominal': self.data['nominal']}
        self.takeovers = {}
        for i, path in enumerate(recovery_paths):
            path = Path(path).resolve()
            m = json.loads((path/'manifest.json').read_text())
            audit = json.loads((path/'root_recovery_audit.json').read_text())
            if m['status'] != 'COMPLETED' or not audit['pass'] or m['controls'] != 128:
                raise ValueError('audited actual128-control C-state takeover required')
            identity = m.get('effect_identity', {})
            expected_c = Path(root)/'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'
            if (not identity.get('applied') or identity.get('global_step') != 50500
                    or Path(identity.get('checkpoint','')).resolve() != expected_c.resolve()
                    or int(m['takeover_tick']) != (32,40,48)[i]):
                raise ValueError('old student recovery cannot replace current C states')
            name = 'recovery%d' % i
            self.paths[name], self.takeovers[name] = path, int(m['takeover_tick'])
            with np.load(path/'trajectory.npz', allow_pickle=False) as f:
                self.data[name] = {k: f[k].copy() for k in f.files}
            d = self.data[name]
            takeover = self.takeovers[name]
            if (not np.array_equal(d['applied'][takeover:,3], d['expert_action'][takeover:,1])
                    or not np.array_equal(d['action'],d['applied'])):
                raise ValueError('recovery futures must follow actually applied own expert commands')
        self.valid_ticks = {}
        for name, takeover in self.takeovers.items():
            d = self.data[name]
            self.valid_ticks[name] = [t for t in range(takeover, 105)
                if d['expert_applied'][t:t+24, 1].all() and not d['expert_clipped'][t:t+24, 1].any()]
            if not self.valid_ticks[name]:
                raise ValueError('no unclipped actual expert future24 for '+name)

    def __getitem__(self, key):
        sample = super().__getitem__(key)
        kind, tick, role = key
        if kind == 'main':
            prop, goal = np.zeros(48, np.float32), np.zeros(27, np.float32)
        else:
            d, tick, role = self.data[kind], int(tick), int(role)
            previous = d['applied'][tick-1, role] if tick else np.zeros(18, np.float32)
            prop, goal = control_context(d['q'][tick, role], d['dq'][tick, role], previous,
                d['obj'][tick, role], self.reference[np.minimum(tick+np.array([8,16,24]), 542)])
        sample.update(proprio=prop, robot_goal=goal, robot_mask=sample['has_pd'])
        return sample


def control_collate(samples):
    import torch
    batch = effect_collate(samples)
    for key in ('proprio', 'robot_goal', 'robot_mask'):
        batch[key] = torch.as_tensor(np.stack([s[key] for s in samples]))
    return batch
