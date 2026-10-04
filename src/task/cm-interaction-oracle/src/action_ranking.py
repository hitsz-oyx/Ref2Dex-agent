"""Matched action-sensitive ranking with frozen continuous advantages.

Pairs condition on motion, reference phase and known episode noise; they
come from different episodes, not same-state physical counterfactuals.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from relative_action import ActionCritic, normalize_history, permute_within_strata


def average_ranks(value):
    _, inverse, count = torch.unique(value, sorted=True, return_inverse=True, return_counts=True)
    ranks = count.cumsum(0).double() - (count.double() - 1) / 2
    return ranks[inverse]


def spearman(first, second):
    if len(first) < 3 or first.std() == 0 or second.std() == 0:
        return None
    return float(torch.corrcoef(torch.stack([average_ranks(first), average_ranks(second)]))[0, 1])


def conditional_pairs(strata, episode, subset, seed, per_anchor=4):
    """Partner selection uses observed group keys only, never target values."""
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    for key in sorted(set(map(tuple, strata[subset].tolist()))):
        rows = (subset & (strata == torch.tensor(key)).all(-1)).nonzero().flatten()
        if len(episode[rows].unique()) < 2:
            continue
        for ep in episode[rows].unique().tolist():
            anchors = rows[episode[rows] == ep]
            candidates = rows[episode[rows] != ep]
            anchors = anchors.repeat_interleave(per_anchor)
            partner = candidates[torch.randint(len(candidates), (len(anchors),), generator=generator)]
            chunks.append(torch.stack([anchors, partner], -1))
    if not chunks:
        raise ValueError("no cross-episode pairs in these strata")
    return torch.cat(chunks)


def pair_correct(score, advantage, pairs):
    delta_target = advantage[pairs[:, 0]] - advantage[pairs[:, 1]]
    delta_score = score[pairs[:, 0]] - score[pairs[:, 1]]
    valid = delta_target != 0
    correct = (delta_score * delta_target > 0).float() + .5 * (delta_score == 0).float()
    return correct, valid


def ranking_metrics(score, advantage, data, subset, pairs):
    correct, valid = pair_correct(score, advantage, pairs)
    groups, episodes = {}, {}
    for key in sorted(set(map(tuple, data["strata"][subset].tolist()))):
        rows = subset & (data["strata"] == torch.tensor(key)).all(-1)
        mask = (data["strata"][pairs[:, 0]] == torch.tensor(key)).all(-1) & valid
        groups[str(key)] = {"rows": int(rows.sum()), "pairs": int(mask.sum()),
                            "pair_accuracy": float(correct[mask].mean()) if mask.any() else None,
                            "spearman": spearman(score[rows], advantage[rows])}
    for ep in data["episode_index"][subset].unique().tolist():
        mask = (data["episode_index"][pairs[:, 0]] == ep) & valid
        episodes[str(ep)] = {"pairs": int(mask.sum()), "pair_accuracy": float(correct[mask].mean()) if mask.any() else None}
    average = lambda values: sum(values) / len(values) if values else None
    return {"rows": int(subset.sum()), "pairs": int(valid.sum()), "target_tie_pairs": int((~valid).sum()),
            "pair_accuracy": float(correct[valid].mean()) if valid.any() else None,
            "macro_stratum_pair_accuracy": average([v["pair_accuracy"] for v in groups.values() if v["pair_accuracy"] is not None]),
            "episode_pair_accuracy": average([v["pair_accuracy"] for v in episodes.values() if v["pair_accuracy"] is not None]),
            "global_spearman": spearman(score[subset], advantage[subset]),
            "macro_stratum_spearman": average([v["spearman"] for v in groups.values() if v["spearman"] is not None]),
            "per_stratum": groups, "per_episode": episodes}


def target_reliability(advantages, data, subset, pairs):
    first = advantages[0, pairs[:, 0]] - advantages[0, pairs[:, 1]]
    second = advantages[1, pairs[:, 0]] - advantages[1, pairs[:, 1]]
    valid = (first != 0) & (second != 0)
    agree = (first * second > 0).float()
    groups = []
    for key in sorted(set(map(tuple, data["strata"][subset].tolist()))):
        mask = (data["strata"][pairs[:, 0]] == torch.tensor(key)).all(-1) & valid
        if mask.any():
            groups.append(float(agree[mask].mean()))
    return {"global_spearman_between_v_fits": spearman(advantages[0, subset], advantages[1, subset]),
            "pair_order_agreement": float(agree[valid].mean()),
            "macro_stratum_pair_order_agreement": sum(groups) / len(groups),
            "valid_pairs": int(valid.sum()), "strata_with_pairs": len(groups)}


def prepare_features(data, device):
    rows, train = data["current_rows"], data["train"]
    mean, scale = normalize_history(data["history"], rows[train])
    history = ((data["history"][rows] - mean) / scale).to(device)
    # Only training CURRENT actions determine normalization; K4 uses the same
    # channel statistics. Its later realized actions are privileged diagnostic.
    action = data["action"]
    amean = action[train, 0].mean(0)
    ascale = action[train, 0].std(0, unbiased=False).clamp_min(1e-5)
    return history, (action - amean) / ascale, (mean, scale, amean, ascale)


def action_features(action, variant):
    result = torch.zeros_like(action)
    if variant == "HaK1":
        result[:, -1] = action[:, 0]
    elif variant == "HaK4_realized":
        result = action.clone()
    elif variant != "H":
        raise ValueError(variant)
    return result


def fit_ranking(history, normalized_action, data, advantage, pairs_train, pairs_test,
                variant, seed, device, epochs=16):
    torch.manual_seed(seed)
    model = ActionCritic(history.shape[-1]).to(device)
    initial_hash = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(parameter.detach().cpu().numpy().tobytes())
    action = action_features(normalized_action, variant).to(device)
    pairs = pairs_train.to(device)
    target = advantage.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(pairs), device=device)
        total, count = 0., 0
        for ix in order.split(512):
            pair = pairs[ix]
            left, right = pair[:, 0], pair[:, 1]
            direction = (target[left] - target[right]).sign()
            valid = direction != 0
            if not valid.any():
                continue
            # Single concatenated forward gives the same computation to each
            # pair member. No future observation, outcome, V or A is an input.
            members = torch.cat([left, right])
            scores = model(history[members], action[members])
            difference = scores[:len(left)] - scores[len(left):]
            loss = nn.functional.softplus(-difference[valid] * direction[valid]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite ranking loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            total += float(loss.detach()) * int(valid.sum())
            count += int(valid.sum())
        if count == 0:
            raise ValueError("no training pairs have a nonzero target difference")
        losses.append(total / count)
    model.eval()
    with torch.no_grad():
        chunks = torch.arange(len(history), device=device).split(512)
        score = torch.cat([model(history[ix], action[ix]).cpu() for ix in chunks])
        # Held-out action permutation within motion/phase/noise. Train rows
        # untouched; H zeros remain identical. Permutation seed is pinned.
        swapped = action.cpu()
        test = ~data["train"]
        swapped[test] = permute_within_strata(swapped[test], data["strata"][test], 208)
        swapped = swapped.to(device)
        swapped_score = torch.cat([model(history[ix], swapped[ix]).cpu() for ix in chunks])
    if not torch.isfinite(score).all() or not torch.isfinite(swapped_score).all():
        raise RuntimeError("nonfinite ranking predictions")
    metrics = {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
               "initial_state_sha256": initial_hash.hexdigest(), "loss": losses,
               "train": ranking_metrics(score, advantage, data, data["train"], pairs_train),
               "test": ranking_metrics(score, advantage, data, test, pairs_test),
               "test_action_permuted": ranking_metrics(swapped_score, advantage, data, test, pairs_test)}
    weights = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
    return metrics, score, swapped_score, weights


def compare_arms(arms, scores, advantages, data, pairs_test):
    test = ~data["train"]
    h, ha = arms["H"]["test"], arms["HaK1"]["test"]
    swap = arms["HaK1"]["test_action_permuted"]
    paired = {}
    for ep in data["test_episode_ids"]:
        before = h["per_episode"][str(ep)]["pair_accuracy"]
        after = ha["per_episode"][str(ep)]["pair_accuracy"]
        if before is not None and after is not None:
            paired[str(ep)] = after - before
    differences = torch.tensor(list(paired.values()))
    bootstrap = differences[torch.randint(len(differences), (2000, len(differences)), generator=torch.Generator().manual_seed(207))].mean(1)
    largest = max(data["test_episode_ids"], key=lambda i: abs(data["episodes"][i]["mc_mean"]))
    remaining_pairs = pairs_test[(data["episode_index"][pairs_test] != largest).all(-1)]
    remainder = test & (data["episode_index"] != largest)
    target = advantages.mean(0)
    h_rest = ranking_metrics(scores["H"], target, data, remainder, remaining_pairs)
    ha_rest = ranking_metrics(scores["HaK1"], target, data, remainder, remaining_pairs)
    independent_gains = []
    for adv in advantages:
        hm = ranking_metrics(scores["H"], adv, data, test, pairs_test)
        am = ranking_metrics(scores["HaK1"], adv, data, test, pairs_test)
        independent_gains.append(am["macro_stratum_pair_accuracy"] - hm["macro_stratum_pair_accuracy"])
    gain = ha["macro_stratum_pair_accuracy"] - h["macro_stratum_pair_accuracy"]
    spearman_gain = ha["macro_stratum_spearman"] - h["macro_stratum_spearman"]
    permutation_drop = ha["macro_stratum_pair_accuracy"] - swap["macro_stratum_pair_accuracy"]
    remainder_gain = ha_rest["macro_stratum_pair_accuracy"] - h_rest["macro_stratum_pair_accuracy"]
    gates = {"conditional_pair_gain_3pp": gain >= .03,
             "conditional_spearman_gain_0_03": spearman_gain >= .03,
             "action_permutation_drop_2pp": permutation_drop >= .02,
             "both_fixed_v_targets_positive_gain": all(g > 0 for g in independent_gains),
             "at_least_12_test_episodes_improve": int((differences > 0).sum()) >= 12,
             "excluding_largest_mc_episode_positive": remainder_gain > 0}
    return {"gates": gates, "pass": all(gates.values()), "macro_pair_gain": gain,
            "macro_spearman_gain": spearman_gain, "action_permutation_drop": permutation_drop,
            "per_fixed_v_target_gains": independent_gains, "paired_episodes": len(paired),
            "improving_episodes": int((differences > 0).sum()), "episode_delta": paired,
            "episode_mean_delta_ci95": torch.quantile(bootstrap, torch.tensor([.025, .975])).tolist(),
            "largest_mc_episode_key": data["episodes"][largest]["key"],
            "pairs_excluding_largest_mc_episode": len(remaining_pairs), "gain_excluding_largest_mc_episode": remainder_gain}
