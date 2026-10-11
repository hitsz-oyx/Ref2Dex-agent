"""Autonomous query scheduling shared by the native competent-PD runner."""
from pathlib import Path

import numpy as np
import torch

from trajectory_policy.competent_actor import actor_input, load_actor, shifted_plan, world_plan


class ActorExecution:
    def __init__(self, directory, device):
        self.device = device
        self.models = {name: load_actor(torch.load(Path(directory) / (name + '.pt'),
                       map_location=device), device) for name in ('tau_only', 'joint')}
        self.plan = None
        self.records = dict(query_tick=[], high_input=[], high_tau=[], plan_world=[])

    def future(self, tick, measured, goal):
        # measured holds only already recorded states, including current tick.
        if tick % 8 == 0:
            past = np.maximum(tick + np.arange(-3, 1), 0)
            inputs = actor_input(*(np.stack([measured[k][i][8:24] for i in past])
                                 for k in ('obj', 'hand', 'q', 'dq', 'velocity')), goal[8:24])
            predictions = np.concatenate([self.models[name](torch.as_tensor(inputs[i * 8:(i + 1) * 8],
                          device=self.device)).cpu().numpy() for i, name in enumerate(('tau_only', 'joint'))])
            current_hand = measured['hand'][-1][8:24]
            current_rotation = measured['obj'][-1][8:24, :3, :3]
            self.plan = world_plan(predictions, current_hand, current_rotation)
            for key, value in dict(query_tick=tick, high_input=inputs, high_tau=predictions,
                                   plan_world=self.plan).items():
                self.records[key].append(np.asarray(value).copy())
        return shifted_plan(self.plan, tick % 8)
