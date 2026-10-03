#!/usr/bin/env python3
"""Fit the corrected temporal Gate 1 value bridge.

The v2 bridge consumes contiguous history and sequence encoders for future
effect/interaction.  It deliberately keeps the v1 six-arm ablation contract so
the implementation repair does not change the scientific comparison.
"""

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


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(sequence)
        return hidden[-1]


class TemporalBridge(nn.Module):
    def __init__(self, blocks: List[str], history_dim: int, effect_dim: int,
                 interaction_dim: int, action_dim: int):
        super().__init__()
        self.blocks = blocks
        if "H" in blocks:
            self.history_encoder = TemporalEncoder(history_dim)
        if "E" in blocks:
            self.effect_encoder = TemporalEncoder(effect_dim)
        if "I" in blocks:
            self.interaction_encoder = TemporalEncoder(interaction_dim)
        fusion_dim = sum(64 if key in ("H", "E", "I") else action_dim for key in blocks)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        for key in self.blocks:
            if key == "H":
                encoded.append(self.history_encoder(blocks[key]))
            elif key == "E":
                encoded.append(self.effect_encoder(blocks[key]))
            elif key == "I":
                encoded.append(self.interaction_encoder(blocks[key]))
            else:
                encoded.append(blocks[key])
        return self.head(torch.cat(encoded, dim=-1))


def _features(dataset: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {
        "H": torch.cat((dataset["history_state"].float(),
                        dataset["history_previous_action"].float(),
                        dataset["history_context"].float(),
                        dataset["history_progress"].float()), dim=-1),
        "A": dataset["action"].float(),
        "E": dataset["effect"].float(),
        "I": dataset["interaction"].float(),
    }


def _split(episode_id: torch.Tensor, source_run: torch.Tensor, seed: int):
    # Episode numbers are only locally unique in a collector run.  Keep the
    # run namespace in the split key so a multi-run dataset cannot leak an
    # episode with the same integer id across train and test.
    groups = sorted(set((int(run), int(ep)) for run, ep in zip(source_run.tolist(), episode_id.tolist())))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * 0.2))
    test_groups = sorted(groups[:n_test])
    train_groups = sorted(groups[n_test:])
    train_set = set(train_groups)
    train = torch.tensor([(int(run), int(ep)) in train_set
                          for run, ep in zip(source_run.tolist(), episode_id.tolist())], dtype=torch.bool)
    return train, ~train, train_groups, test_groups


def _standardize(sequence: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    train_values = sequence[train].reshape(-1, sequence.shape[-1])
    mean = train_values.mean(dim=0)
    scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (sequence - mean.view(*([1] * (sequence.ndim - 1)), -1)) / scale.view(*([1] * (sequence.ndim - 1)), -1)


def _fit_variant(name: str, blocks: Dict[str, torch.Tensor], target: torch.Tensor,
                 aux: torch.Tensor, train: torch.Tensor, test: torch.Tensor,
                 device: torch.device, epochs: int, batch_size: int, seed: int,
                 episode_id: torch.Tensor, noise_std: torch.Tensor) -> Dict[str, object]:
    torch.manual_seed(seed)
    # Standardization is done once outside this function.  Recomputing the
    # large history tensor for every arm made the first v2 sweep CPU-bound.
    y_return = target.float()
    ret_mean = y_return[train].mean()
    ret_scale = y_return[train].std(unbiased=False).clamp_min(1e-6)
    y_return_norm = (y_return - ret_mean) / ret_scale
    aux = aux.float()
    aux_mean = aux[train, 2:].mean(dim=0)
    aux_scale = aux[train, 2:].std(dim=0, unbiased=False).clamp_min(1e-6)
    y_aux = aux.clone()
    y_aux[:, 2:] = (y_aux[:, 2:] - aux_mean) / aux_scale
    model = TemporalBridge(VARIANTS[name], blocks["H"].shape[-1], blocks["E"].shape[-1],
                           blocks["I"].shape[-1], blocks["A"].shape[-1]).to(device)
    device_blocks = {key: blocks[key].to(device) for key in VARIANTS[name]}
    device_return = y_return_norm.to(device)
    device_aux = y_aux.to(device)
    device_train_idx = train.nonzero(as_tuple=False).flatten().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        model.train()
        permutation = device_train_idx[torch.randperm(device_train_idx.numel(), device=device)]
        for start in range(0, permutation.numel(), batch_size):
            idx = permutation[start:start + batch_size]
            batch = {key: device_blocks[key][idx] for key in VARIANTS[name]}
            pred = model(batch)
            loss_return = nn.functional.mse_loss(pred[:, 0], device_return[idx])
            loss_binary = nn.functional.binary_cross_entropy_with_logits(pred[:, 1:3], device_aux[idx, :2])
            loss_cont = nn.functional.mse_loss(pred[:, 3:], device_aux[idx, 2:])
            loss = loss_return + 0.2 * loss_binary + 0.2 * loss_cont
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = []
        for start in range(0, target.shape[0], batch_size):
            idx = torch.arange(start, min(start + batch_size, target.shape[0]))
            batch = {key: device_blocks[key][idx] for key in VARIANTS[name]}
            predictions.append(model(batch)[:, 0].cpu())
        prediction = torch.cat(predictions) * ret_scale + ret_mean
    errors = (prediction - target.float()).abs()
    return {
        "name": name,
        "input_dims": {key: list(blocks[key].shape[1:]) for key in VARIANTS[name]},
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_mae": float(errors[train].mean()),
        "test_mae": float(errors[test].mean()),
        "test_rmse": float(torch.sqrt(((prediction[test] - target[test].float()) ** 2).mean())),
        "test_predictions": prediction[test],
        "test_targets": target[test].float(),
        "test_episode_id": episode_id[test],
        "test_noise_std": noise_std[test],
    }


def _bootstrap_delta(base: Dict[str, object], variant: Dict[str, object], seed: int,
                     repeats: int = 1000) -> Dict[str, object]:
    episodes = sorted(set(int(x) for x in base["test_episode_id"].tolist()))
    base_errors = torch.abs(base["test_predictions"] - base["test_targets"])
    variant_errors = torch.abs(variant["test_predictions"] - variant["test_targets"])
    base_by_episode = {e: base_errors[base["test_episode_id"] == e].mean() for e in episodes}
    variant_by_episode = {e: variant_errors[variant["test_episode_id"] == e].mean() for e in episodes}
    deltas = torch.tensor([float(base_by_episode[e] - variant_by_episode[e]) for e in episodes])
    sampled = torch.randint(0, len(episodes), (repeats, len(episodes)), generator=torch.Generator().manual_seed(seed))
    bootstrap = deltas[sampled].mean(dim=1)
    quantiles = torch.quantile(bootstrap, torch.tensor([0.025, 0.975]))
    base_mae = float(base_errors.mean())
    variant_mae = float(variant_errors.mean())
    return {
        "base_mae": base_mae,
        "variant_mae": variant_mae,
        "absolute_mae_reduction": base_mae - variant_mae,
        "relative_mae_reduction": (base_mae - variant_mae) / max(base_mae, 1e-8),
        "episode_bootstrap_delta_ci95": [float(quantiles[0]), float(quantiles[1])],
        "episode_count": len(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    torch.set_num_threads(args.num_threads)
    episode_id = dataset["episode_id"].long()
    source_run = dataset.get("source_run", torch.zeros_like(episode_id)).long()
    noise_std = dataset["noise_std"].float()
    train, test, train_episodes, test_episodes = _split(episode_id, source_run, args.seed)
    blocks = _features(dataset)
    target = dataset["return_to_go"].float()
    aux = dataset["episode_auxiliary"].float()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    report = {
        "schema": "ref2dex.gate1_value_bridge.v2",
        "input": str(args.input.resolve()),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "split_unit": "(source_run, episode_id)",
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "primary_metric": "held-out episode-cluster MAE of exact return_to_go",
        "temporal_encoder": "GRU(hidden=64) for H/E/I; no sequence flattening",
        "variants": {},
        "comparisons_vs_V_H": {},
    }
    standardized_blocks = {key: _standardize(value, train) for key, value in blocks.items()}
    fitted = {}
    for offset, name in enumerate(VARIANTS):
        fitted[name] = _fit_variant(name, standardized_blocks, target, aux, train, test, device,
                                     args.epochs, args.batch_size, args.seed,
                                     episode_id, noise_std)
        report["variants"][name] = {k: v for k, v in fitted[name].items() if not isinstance(v, torch.Tensor)}
    for offset, name in enumerate(VARIANTS):
        if name != "V_H":
            report["comparisons_vs_V_H"][name] = _bootstrap_delta(
                fitted["V_H"], fitted[name], args.seed + 1000 + offset)
    report["status"] = "FIT_COMPLETED"
    report["no_policy_training"] = True
    report["no_online_cm"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
