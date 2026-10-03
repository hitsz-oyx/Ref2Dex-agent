#!/usr/bin/env python3
"""Fit the predeclared Gate 1 value-bridge ablations on grouped episodes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn


VARIANTS = {
    "V_H": ["H"],
    "V_HA": ["H", "A"],
    "V_HE": ["H", "E"],
    "V_HI": ["H", "I"],
    "V_HEI": ["H", "E", "I"],
    "V_HAEI": ["H", "A", "E", "I"],
}


class Bridge(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _features(dataset: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {
        "H": torch.cat((dataset["state"].float(), dataset["previous_action"].float(), dataset["context"].float()), dim=1),
        "A": dataset["action"].float(),
        "E": dataset["effect"].float().flatten(1),
        "I": dataset["interaction"].float().flatten(1),
    }


def _split(episode_id: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    episodes = sorted(set(int(x) for x in episode_id.tolist()))
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n_test = max(1, round(len(episodes) * 0.2))
    test_episodes = sorted(episodes[:n_test])
    train_episodes = sorted(episodes[n_test:])
    train = torch.tensor([int(x) in set(train_episodes) for x in episode_id.tolist()], dtype=torch.bool)
    test = ~train
    return train, test, train_episodes, test_episodes


def _standardize(features: torch.Tensor, train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features[train].mean(dim=0)
    scale = features[train].std(dim=0, unbiased=False).clamp_min(1e-6)
    return (features - mean) / scale, mean, scale


def _fit_variant(
    name: str,
    blocks: Dict[str, torch.Tensor],
    target: torch.Tensor,
    aux: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    x = torch.cat([blocks[key] for key in VARIANTS[name]], dim=1)
    x, _, _ = _standardize(x, train)
    y_return = target.float()
    ret_mean = y_return[train].mean()
    ret_scale = y_return[train].std(unbiased=False).clamp_min(1e-6)
    y_return = (y_return - ret_mean) / ret_scale
    aux = aux.float()
    aux_mean = aux[train, 2:].mean(dim=0)
    aux_scale = aux[train, 2:].std(dim=0, unbiased=False).clamp_min(1e-6)
    y_aux = aux.clone()
    y_aux[:, 2:] = (y_aux[:, 2:] - aux_mean) / aux_scale
    model = Bridge(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    for _ in range(epochs):
        model.train()
        permutation = train_idx[torch.randperm(train_idx.numel())]
        for start in range(0, permutation.numel(), batch_size):
            idx = permutation[start:start + batch_size]
            pred = model(x[idx].to(device))
            loss_return = nn.functional.mse_loss(pred[:, 0], y_return[idx].to(device))
            loss_binary = nn.functional.binary_cross_entropy_with_logits(pred[:, 1:3], y_aux[idx, :2].to(device))
            loss_cont = nn.functional.mse_loss(pred[:, 3:], y_aux[idx, 2:].to(device))
            loss = loss_return + 0.2 * loss_binary + 0.2 * loss_cont
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x.to(device)).cpu()
    predicted_return = prediction[:, 0] * ret_scale + ret_mean
    errors = (predicted_return - target.float()).abs()
    test_error = errors[test]
    train_error = errors[train]
    return {
        "name": name,
        "input_dim": int(x.shape[1]),
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_mae": float(train_error.mean()),
        "test_mae": float(test_error.mean()),
        "test_rmse": float(torch.sqrt(((predicted_return[test] - target[test].float()) ** 2).mean())),
        "test_predictions": predicted_return[test],
        "test_targets": target[test].float(),
        "test_episode_id": dataset_episode_id[test],
        "test_noise_std": dataset_noise_std[test],
    }


def _bootstrap_delta(base: Dict[str, object], variant: Dict[str, object], seed: int, repeats: int = 1000) -> Dict[str, float]:
    episodes = sorted(set(int(x) for x in base["test_episode_id"].tolist()))
    base_errors = torch.abs(base["test_predictions"] - base["test_targets"])
    var_errors = torch.abs(variant["test_predictions"] - variant["test_targets"])
    base_by_episode = {e: base_errors[base["test_episode_id"] == e].mean() for e in episodes}
    var_by_episode = {e: var_errors[variant["test_episode_id"] == e].mean() for e in episodes}
    deltas = torch.tensor([float(base_by_episode[e] - var_by_episode[e]) for e in episodes])
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randint(0, len(episodes), (repeats, len(episodes)), generator=generator)
    bootstrap = deltas[sampled].mean(dim=1)
    q = torch.quantile(bootstrap, torch.tensor([0.025, 0.975]))
    base_mae = float(base_errors.mean())
    variant_mae = float(var_errors.mean())
    return {
        "base_mae": base_mae,
        "variant_mae": variant_mae,
        "absolute_mae_reduction": base_mae - variant_mae,
        "relative_mae_reduction": (base_mae - variant_mae) / max(base_mae, 1e-8),
        "episode_bootstrap_delta_ci95": [float(q[0]), float(q[1])],
        "episode_count": len(episodes),
    }


def _noise_stratified(base: Dict[str, object], variant: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    values = sorted(set(float(x) for x in base["test_noise_std"].tolist()))
    result = {}
    for value in values:
        mask = base["test_noise_std"] == value
        base_mae = torch.abs(base["test_predictions"][mask] - base["test_targets"][mask]).mean()
        variant_mae = torch.abs(variant["test_predictions"][mask] - variant["test_targets"][mask]).mean()
        result[str(value)] = {
            "rows": int(mask.sum()),
            "base_mae": float(base_mae),
            "variant_mae": float(variant_mae),
            "absolute_mae_reduction": float(base_mae - variant_mae),
            "relative_mae_reduction": float((base_mae - variant_mae) / base_mae.clamp_min(1e-8)),
        }
    return result


def main() -> None:
    global dataset_episode_id, dataset_noise_std
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    dataset_episode_id = dataset["episode_id"].long()
    dataset_noise_std = dataset["noise_std"].float()
    train, test, train_episodes, test_episodes = _split(dataset_episode_id, args.seed)
    blocks = _features(dataset)
    target = dataset["return_to_go"].float()
    aux = dataset["episode_auxiliary"].float()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    results: Dict[str, object] = {
        "schema": "ref2dex.gate1_value_bridge.v1",
        "input": str(args.input.resolve()),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "split_unit": "episode_id",
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "primary_metric": "held-out episode-cluster MAE of exact return_to_go",
        "variants": {},
    }
    fitted = {}
    for offset, name in enumerate(VARIANTS):
        fitted[name] = _fit_variant(name, blocks, target, aux, train, test, device,
                                     args.epochs, args.batch_size, args.seed + offset)
        metric = {k: v for k, v in fitted[name].items() if not isinstance(v, torch.Tensor)}
        results["variants"][name] = metric
    base = fitted["V_H"]
    for name in VARIANTS:
        if name == "V_H":
            continue
        results.setdefault("comparisons_vs_V_H", {})[name] = _bootstrap_delta(
            base, fitted[name], args.seed + 1000 + list(VARIANTS).index(name))
        results["comparisons_vs_V_H"][name]["noise_stratified"] = _noise_stratified(base, fitted[name])
    results["status"] = "FIT_COMPLETED"
    results["no_policy_training"] = True
    results["no_online_cm"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
