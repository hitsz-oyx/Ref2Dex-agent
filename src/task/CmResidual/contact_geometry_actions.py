"""Executable expert combinations with separate wrist and five-finger weights."""
import torch
from .orientation_anchored_options import orientation_anchored_action

BLOCKS = ((0, 3), (6, 8), (8, 10), (10, 12), (12, 14), (14, 18))
OPTION_NAMES = ('base', 'rotation_cup', 'mixture0', 'mixture1', 'mixture2',
                'mixture3', 'mixture4', 'mixture5')
ALLOCATION_TO_OPTION = (0, 1, 2, 3, 4, 5, 6, 7, 0, 1)


def sample_programs(count, device, generator):
    """Sample before allocation; coefficients remain fixed during feedback."""
    weights = torch.zeros(count, 8, 6, 6, device=device)
    weights[:, 0, :, 4] = 1
    weights[:, 1, :, 1] = 1
    # Sparse, continuous simplex proposals; not claimed to be Dirichlet samples.
    uniform = torch.rand(count, 6, 6, 6, device=device, generator=generator)
    positive = (-uniform.clamp_min(torch.finfo(uniform.dtype).tiny).log()).pow(3)
    weights[:, 2:] = positive / positive.sum(-1, keepdim=True)
    return weights


def feedback_candidates(bank, weights):
    if bank.ndim != 3 or bank.shape[1:] != (6, 18):
        raise ValueError('six expert actions required')
    if weights.shape != (len(bank), 8, 6, 6):
        raise ValueError('eight programmes, six action blocks, six experts required')
    if not torch.isfinite(bank).all() or not torch.isfinite(weights).all():
        raise ValueError('nonfinite candidate input')
    if (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=2e-6):
        raise ValueError('convex expert weights required')
    if (bank.abs() > 1 + 1e-6).any():
        raise ValueError('expert action outside native raw domain')
    result = bank[:, 1, None].expand(-1, 8, -1).clone()
    for block, (start, stop) in enumerate(BLOCKS):
        result[..., start:stop] = torch.einsum('nke,ned->nkd', weights[:, :, block], bank[:, :, start:stop])
    result[:, 0] = bank[:, 4]
    result[:, 1] = bank[:, 1]
    return result


def executable_candidates(bank, weights, anchor, position, offset, scale):
    feedback = feedback_candidates(bank, weights)
    values = [feedback[:, 0]]
    for option in range(1, 8):
        values.append(orientation_anchored_action(feedback[:, option], anchor, position, offset, scale))
    return torch.stack(values, 1)
