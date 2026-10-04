"""Episode-isolated relative-outcome labels and small matched action critics.

Offline labels use future observations. Critic inputs contain current/history
observations only; recorded multi-step actions are explicitly privileged.
"""
from __future__ import annotations

import random

import torch
from torch import nn


def discounted_returns(reward, gamma):
    result = torch.empty_like(reward)
    carry = 0.0
    for t in range(len(reward) - 1, -1, -1):
        carry = reward[t] + gamma * carry
        result[t] = carry
    return result


def nstep_outcome(reward, times, horizon, gamma):
    """Complete single episode; -1 endpoint means terminal, bootstrap zero."""
    local = torch.zeros(len(times), dtype=reward.dtype)
    endpoint = torch.full((len(times),), -1, dtype=torch.long)
    discount = torch.zeros_like(local)
    for j, t in enumerate(times.tolist()):
        n = min(horizon, len(reward) - t)
        local[j] = (reward[t:t + n] * gamma ** torch.arange(n)).sum()
        if t + n < len(reward):
            endpoint[j] = t + n
            discount[j] = gamma ** n
    return local, endpoint, discount


def episode_split(keys, split_id=20261004):
    keys = sorted(keys)
    random.Random(split_id).shuffle(keys)
    n_test = max(1, round(.2 * len(keys)))
    return sorted(keys[n_test:]), sorted(keys[:n_test])


def stratified_folds(episodes, train_ids, nfold=3):
    """Round-robin whole episodes per motion/noise stratum, deterministic."""
    strata = {}
    for i in train_ids:
        strata.setdefault(episodes[i]["stratum"], []).append(i)
    folds = [[] for _ in range(nfold)]
    for key, ids in sorted(strata.items()):
        random.Random(20261004 + 100 * key[0] + key[1]).shuffle(ids)
        for j, i in enumerate(ids):
            folds[j % nfold].append(i)
    return [sorted(f) for f in folds]


def fit_thresholds(advantage, strata, train, minimum=.05, min_rows=32):
    thresholds = {}
    for key in sorted(set(map(tuple, strata[train].tolist()))):
        mask = train & (strata == torch.tensor(key)).all(-1)
        values = advantage[mask]
        if len(values) >= min_rows:
            lo, hi = torch.quantile(values, torch.tensor([.3, .7])).tolist()
            thresholds[key] = (min(lo, -minimum), max(hi, minimum))
    return thresholds


def label_advantage(advantage, strata, thresholds):
    label = torch.zeros(len(advantage), dtype=torch.long)
    for key, (lo, hi) in thresholds.items():
        mask = (strata == torch.tensor(key)).all(-1)
        label[mask & (advantage < lo)] = -1
        label[mask & (advantage > hi)] = 1
    return label


def permute_within_strata(action, strata, seed):
    result = action.clone()
    rng = torch.Generator().manual_seed(seed)
    for key in sorted(set(map(tuple, strata.tolist()))):
        idx = ((strata == torch.tensor(key)).all(-1)).nonzero().flatten()
        result[idx] = action[idx[torch.randperm(len(idx), generator=rng)]]
    return result


class StateValue(nn.Module):
    def __init__(self, dim, width=48):
        super().__init__()
        self.history = nn.GRU(dim, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, history):
        return self.head(self.history(history)[1][-1]).squeeze(-1)


class ActionCritic(nn.Module):
    def __init__(self, dim, width=48):
        super().__init__()
        self.history = nn.GRU(dim, width, batch_first=True)
        self.action = nn.GRU(18, width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, history, action):
        h = self.history(history)[1][-1]
        a = self.action(action)[1][-1]
        return self.head(torch.cat([h, a], -1)).squeeze(-1)


def normalize_history(history, rows):
    values = history[rows].flatten(0, 1)
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-5)


def fit_value(history, returns, fit_rows, seed, device, epochs=16, batch=512):
    torch.manual_seed(seed)
    model = StateValue(history.shape[-1]).to(device)
    mean, scale = normalize_history(history, fit_rows)
    ymean = returns[fit_rows].mean()
    yscale = returns[fit_rows].std(unbiased=False).clamp_min(1e-5)
    # Equal sampled rows per episode: MC target and normalization stay in raw
    # task-reward units after inversion. No nonlinear value transformation.
    x = ((history[fit_rows] - mean) / scale).to(device)
    y = ((returns[fit_rows] - ymean) / yscale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), device=device)
        total = 0.0
        for ix in order.split(batch):
            loss = (model(x[ix]) - y[ix]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite V loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        losses.append(total / len(x))
    del x, y
    return model.eval(), (mean, scale, ymean, yscale), losses


@torch.no_grad()
def predict_value(model, norm, history, rows, device):
    mean, scale, ymean, yscale = norm
    pred = []
    for ix in rows.split(512):
        x = ((history[ix] - mean) / scale).to(device)
        pred.append((model(x).cpu() * yscale + ymean))
    return torch.cat(pred)


def crossfit_values(data, seed, device, epochs, emit):
    h, mc = data["history"], data["returns"]
    folds = stratified_folds(data["episodes"], data["train_episode_ids"])
    all_train = set(data["train_episode_ids"])
    current = torch.full_like(data["local"], float("nan"))
    future = torch.zeros_like(current)
    test_now, test_future = [], []
    records, weights = [], []
    for fold, excluded in enumerate(folds):
        fit_episodes = sorted(all_train - set(excluded))
        fit_rows = torch.cat([data["episodes"][i]["fit_rows"] for i in fit_episodes])
        model, norm, loss = fit_value(h, mc, fit_rows, seed + fold, device, epochs)
        # Current and endpoint observations for an episode use exactly the same
        # model. That model excludes every observation from the episode.
        mask = torch.tensor([int(i) in excluded for i in data["episode_index"]])
        query = mask.nonzero().flatten()
        current[query] = predict_value(model, norm, h, data["current_rows"][query], device)
        live = query[data["endpoint_rows"][query] >= 0]
        future[live] = predict_value(model, norm, h, data["endpoint_rows"][live], device)
        tq = (~data["train"]).nonzero().flatten()
        tv = torch.zeros(len(tq))
        live_test = data["endpoint_rows"][tq] >= 0
        tv[live_test] = predict_value(model, norm, h, data["endpoint_rows"][tq[live_test]], device)
        test_now.append(predict_value(model, norm, h, data["current_rows"][tq], device))
        test_future.append(tv)
        records.append({"fold": fold, "fit_episode_ids": fit_episodes,
                        "excluded_episode_ids": excluded, "loss": loss})
        weights.append({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "norm": norm})
        emit({"seed": seed, "fold": fold, "fit_episodes": len(fit_episodes),
              "excluded_episodes": len(excluded), "final_loss": loss[-1]})
        del model
    current[~data["train"]] = torch.stack(test_now).mean(0)
    future[~data["train"]] = torch.stack(test_future).mean(0)
    if not torch.isfinite(current).all() or not torch.isfinite(future).all():
        raise RuntimeError("incomplete cross-fit predictions")
    advantage = data["local"] + data["discount"] * future - current
    return advantage, current, future, records, weights


def classification_metrics(score, label):
    positive, negative = label == 1, label == -1
    if not positive.any() or not negative.any():
        return None
    ba = .5 * ((score[positive] >= 0).float().mean() + (score[negative] < 0).float().mean())
    # Exact tie-correct AUC without allocating a quadratic score matrix.
    values, inverse, counts = torch.unique(score, sorted=True, return_inverse=True, return_counts=True)
    del values
    ranks = counts.cumsum(0) - (counts - 1) / 2.0
    npos, nneg = int(positive.sum()), int(negative.sum())
    auc = (ranks[inverse][positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return {"balanced_accuracy": float(ba), "auc": float(auc)}


def fit_critic(data, label, variant, seed, device, epochs=16):
    torch.manual_seed(seed)
    model = ActionCritic(data["history"].shape[-1]).to(device)
    rows = data["current_rows"]
    train = data["train"] & (label != 0)
    mean, scale = normalize_history(data["history"], rows[train])
    x = ((data["history"][rows] - mean) / scale).to(device)
    action = data["action"].clone()
    # Common observed-action normalization. Zero controls represent its mean.
    amean = action[train].flatten(0, 1).mean(0)
    ascale = action[train].flatten(0, 1).std(0, unbiased=False).clamp_min(1e-5)
    action = (action - amean) / ascale
    if variant == "H":
        action.zero_()
    elif variant == "HaK1":
        action[:, -1] = action[:, 0].clone()
        action[:, :-1] = 0
    elif variant != "HaK4_realized":
        raise ValueError(variant)
    a = action.to(device)
    y = (label == 1).float().to(device)
    indices = train.nonzero().flatten().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    # Each (episode,class) gets equal weight; no episode length reward.
    weight = torch.zeros(len(label))
    for ep in data["train_episode_ids"]:
        for cls in [-1, 1]:
            mask = train & (data["episode_index"] == ep) & (label == cls)
            if mask.any():
                weight[mask] = 1 / int(mask.sum())
    weight = (weight / weight[train].mean()).to(device)
    losses = []
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), device=device)]
        total = 0.0
        for ix in order.split(512):
            loss = (nn.functional.binary_cross_entropy_with_logits(model(x[ix], a[ix]), y[ix], reduction="none") * weight[ix]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite critic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        losses.append(total / len(indices))
    model.eval()
    with torch.no_grad():
        score = torch.cat([model(x[ix], a[ix]).cpu() for ix in torch.arange(len(x), device=device).split(512)])
        test = ~data["train"]
        shuffled = action.clone()
        shuffled[test] = permute_within_strata(action[test], data["strata"][test], seed + 1000)
        shuffled = shuffled.to(device)
        swap = torch.cat([model(x[ix], shuffled[ix]).cpu() for ix in torch.arange(len(x), device=device).split(512)])
    result = {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
              "loss": losses, "metrics": classification_metrics(score[test], label[test]),
              "shuffled_test_metrics": classification_metrics(swap[test], label[test]),
              "per_episode": {str(i): classification_metrics(score[data["episode_index"] == i], label[data["episode_index"] == i])
                              for i in data["test_episode_ids"]}}
    return result, score, {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                           "history_norm": (mean, scale), "action_norm": (amean, ascale)}
