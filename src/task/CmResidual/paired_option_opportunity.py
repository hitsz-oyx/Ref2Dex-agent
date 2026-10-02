"""Initial-only matched placements and fixed antithetic joint request options."""
import torch


def matched_inputs(motion, assignment, seed):
    device = motion.device
    motion, assignment = motion.cpu(), assignment.cpu()
    generator = torch.Generator(device='cpu').manual_seed(seed + 11000)
    placements = (torch.rand((192, 2), generator=generator) * 2 - 1) * .01
    generator = torch.Generator(device='cpu').manual_seed(seed + 18000)
    options = torch.randn((192, 12), generator=generator)
    offsets = torch.zeros((768, 2))
    raw = torch.zeros((768, 12))
    cluster = torch.full((768,), -1, dtype=torch.long)
    for m in range(3):
        ids = torch.arange(m * 64, (m + 1) * 64)
        for arm in range(4):
            envs = ((motion == m) & (assignment == arm)).nonzero().flatten()
            assert len(envs) == 64
            cluster[envs] = ids
            offsets[envs] = placements[ids]
            if arm in (2, 3):
                raw[envs] = options[ids] * (1 if arm == 2 else -1)
    assert (cluster >= 0).all()
    return offsets.to(device), cluster.to(device), raw.to(device)
