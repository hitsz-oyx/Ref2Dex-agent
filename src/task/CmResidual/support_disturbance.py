"""Fixed finite mechanical disturbance/recovery cells, not a learned policy."""
import torch


def cells(motion, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed+13000)
    result = torch.full((len(motion),), -1, dtype=torch.long)
    for mo in range(3):
        ids = (motion.cpu() == mo).nonzero().flatten()
        if len(ids) != 256:
            raise ValueError('balanced768 environment design')
        result[ids] = torch.arange(32).repeat_interleave(8)[torch.randperm(256, generator=generator)]
    return result


def recovery_parameters(arm, direction):
    if arm.shape != direction.shape or ((arm < 0) | (arm > 3)).any() or (direction.abs() != 1).any():
        raise ValueError('fixed recovery arms/directions')
    p = torch.zeros((len(arm), 3), device=arm.device)
    p[:,0] = direction*.02*((arm == 1).float()-(arm == 2).float())
    p[:,2] = (arm == 3).float()*.15
    return p
