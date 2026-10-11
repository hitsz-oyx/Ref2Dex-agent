"""Autonomous query scheduling shared by the native competent-PD runner."""
from pathlib import Path

import numpy as np
import torch
import json

from trajectory_policy.competent_actor import actor_input, load_actor, shifted_plan, world_plan


class ActorExecution:
    def __init__(self, directory, device, period=8):
        self.device = device
        if period not in (1, 8):
            raise ValueError('one or eight control query period required')
        self.period = period
        self.reference = None
        manifest = json.loads((Path(directory) / 'manifest.json').read_text())
        if manifest.get('reference'):
            from trajectory_policy.reference_actor import load_reference_actor
            load_actor_model = load_reference_actor
            with np.load(manifest['reference'], allow_pickle=False) as f:
                self.reference = {k: f[k].copy() for k in f.files}
        else:
            load_actor_model = load_actor
        self.models = {name: load_actor_model(torch.load(Path(directory) / (name + '.pt'),
                       map_location=device), device) for name in ('tau_only', 'joint')}
        self.plan = None
        self.records = dict(query_tick=[], high_input=[], high_tau=[], plan_world=[])

    def future(self, tick, measured, goal):
        # measured holds only already recorded states, including current tick.
        if tick % self.period == 0:
            past = np.maximum(tick + np.arange(-3, 1), 0)
            inputs = actor_input(*(np.stack([measured[k][i][8:24] for i in past])
                                 for k in ('obj', 'hand', 'q', 'dq', 'velocity')), goal[8:24])
            if self.reference is not None:
                from trajectory_policy.reference_actor import reference_input
                inputs = reference_input(*(np.stack([measured[k][i][8:24] for i in past])
                         for k in ('obj', 'hand', 'q', 'dq', 'velocity')), self.reference, np.full(16, tick))
            predictions = np.concatenate([self.models[name](torch.as_tensor(inputs[i * 8:(i + 1) * 8],
                          device=self.device)).cpu().numpy() for i, name in enumerate(('tau_only', 'joint'))])
            current_hand = measured['hand'][-1][8:24]
            current_rotation = measured['obj'][-1][8:24, :3, :3]
            self.plan = world_plan(predictions, current_hand, current_rotation)
            for key, value in dict(query_tick=tick, high_input=inputs, high_tau=predictions,
                                   plan_world=self.plan).items():
                self.records[key].append(np.asarray(value).copy())
        return shifted_plan(self.plan, tick % self.period)
