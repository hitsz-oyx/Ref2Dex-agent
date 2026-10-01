"""Fixed current-context learners for randomized physical response experiments."""
from __future__ import annotations
import torch
from src.task.CmResidual.actuation_effect import state_features
from src.task.CmResidual.v118_planner import QUERY_LINKS, TIP_LINKS

SEEDS = (611, 612, 613)
NULL_COMMANDS = (7, 9, 11, 13, 16, 17)


def iid_assignment(count, evaluation_seed):
    generator = torch.Generator(device='cpu').manual_seed(9900 + evaluation_seed)
    return torch.randint(7, (count,), generator=generator), torch.full((count, 7), 1 / 7)


@torch.no_grad()
def context(state, reference_action, bridge):
    indices = [QUERY_LINKS.index(name) for name in ('hand_base_link', *TIP_LINKS)]
    parts = []
    for before, action in zip(state.split(128), reference_action.split(128)):
        before = before.to(bridge.device); action = action.to(bridge.device).clone()
        action[:, NULL_COMMANDS] = 0
        geometry = bridge.current(before[:, :18], before[:, 36:49])
        base = state_features(before, geometry.link_poses[:, indices, :3, 3],
                              geometry.object_pose[:, :3, :3])
        parts.append(torch.cat((base, action), -1))
    result = torch.cat(parts)
    if result.shape != (len(state), 81) or not torch.isfinite(result).all():
        raise ValueError('invalid current context')
    return result


def indicators(arm, dtype=torch.float32):
    return torch.nn.functional.one_hot(arm, 7)[:, 1:].to(dtype)


def network(dim, output, seed, device):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, output)).to(device)


def fit(x, y, seed, updates, check):
    # All scales belong to exactly the rows passed here, including nuisance folds.
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-5)
    target_mean = y.mean(0); target_std = y.std(0, unbiased=False).clamp_min(1e-6)
    normalized_x = (x - mean) / std; normalized_y = (y - target_mean) / target_std
    model = network(x.shape[1], y.shape[1], seed, x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    for update in range(updates):
        if update % 100 == 0: check()
        chosen = torch.randint(len(x), (256,), device=x.device)
        loss = (model(normalized_x[chosen]) - normalized_y[chosen]).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        mean=mean.cpu(), std=std.cpu(), target_mean=target_mean.cpu(),
        target_std=target_std.cpu(), seed=seed, updates=updates, fit_rows=len(x),
        input_dim=x.shape[1], output_dim=y.shape[1])


@torch.no_grad()
def predict(x, checkpoint):
    model = network(checkpoint['input_dim'], checkpoint['output_dim'],
                    checkpoint['seed'], x.device).eval()
    model.load_state_dict(checkpoint['model'])
    return model((x - checkpoint['mean'].to(x.device)) / checkpoint['std'].to(x.device)) * \
        checkpoint['target_std'].to(x.device) + checkpoint['target_mean'].to(x.device)


@torch.no_grad()
def factual_contrast(x, checkpoint):
    candidates = []
    for arm in range(1, 7):
        labels = torch.full((len(x),), arm, device=x.device, dtype=torch.long)
        candidates.append(predict(torch.cat((x, indicators(labels, x.dtype)), -1), checkpoint))
    return torch.stack([candidates[2*j] - candidates[2*j+1] for j in range(3)], -1)
