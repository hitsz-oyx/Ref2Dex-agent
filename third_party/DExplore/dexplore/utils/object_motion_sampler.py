"""Cycle all reference motions without changing an environment's object mesh."""
import torch


class ObjectMotionSampler:
    def __init__(self, motion_objects, num_envs, device):
        names = sorted(set(motion_objects))
        if not names or num_envs < len(names):
            raise ValueError("object sampling needs at least one environment per object")
        ids = [names.index(name) for name in motion_objects]
        self.motion_objects = torch.tensor(ids, device=device, dtype=torch.long)
        self.env_objects = torch.arange(num_envs, device=device) % len(names)
        groups = [[i for i, obj in enumerate(ids) if obj == k] for k in range(len(names))]
        self.counts = torch.tensor([len(group) for group in groups], device=device)
        self.members = torch.zeros((len(names), max(map(len, groups))), device=device,
                                   dtype=torch.long)
        for k, group in enumerate(groups):
            self.members[k, :len(group)] = torch.tensor(group, device=device)
        self.cursor = torch.zeros(len(names), device=device, dtype=torch.long)
        self.visits = torch.zeros(len(ids), device=device, dtype=torch.long)
        self.initial = self.members[self.env_objects, 0]

    def sample(self, env_ids):
        objects = self.env_objects[env_ids]
        result = torch.empty_like(env_ids)
        for obj in objects.unique():
            mask = objects == obj
            count = int(mask.sum())
            slots = (torch.arange(count, device=env_ids.device) + self.cursor[obj]) % self.counts[obj]
            result[mask] = self.members[obj, slots]
            self.cursor[obj] += count
        self.visits.index_add_(0, result, torch.ones_like(result))
        return result
