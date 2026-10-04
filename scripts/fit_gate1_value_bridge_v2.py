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

FUTURE_ACTION_VARIANTS = {
    "V_HF": ["H", "F"],
    "V_HFEI": ["H", "F", "E", "I"],
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
        if "F" in blocks:
            self.future_action_encoder = TemporalEncoder(action_dim)
        fusion_dim = sum(64 if key in ("H", "E", "I", "F") else action_dim for key in blocks)
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
            elif key == "F":
                encoded.append(self.future_action_encoder(blocks[key]))
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
        **({"F": dataset["future_action"].float()} if "future_action" in dataset else {}),
    }


def _split(episode_id: torch.Tensor, source_namespace: torch.Tensor,
           source_run: torch.Tensor, seed: int):
    # Keep checkpoint namespace and run namespace in the split key.  The
    # former is the actor uncertainty unit; the latter prevents episode-id
    # collisions between repeated runs of one checkpoint.
    groups = sorted(set((int(namespace), int(run), int(ep))
                        for namespace, run, ep in zip(
                            source_namespace.tolist(), source_run.tolist(), episode_id.tolist())))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * 0.2))
    test_groups = sorted(groups[:n_test])
    train_groups = sorted(groups[n_test:])
    train_set = set(train_groups)
    train = torch.tensor([(int(namespace), int(run), int(ep)) in train_set
                          for namespace, run, ep in zip(
                              source_namespace.tolist(), source_run.tolist(), episode_id.tolist())], dtype=torch.bool)
    return train, ~train, train_groups, test_groups


def _namespace_holdout_split(episode_id: torch.Tensor, source_namespace: torch.Tensor,
                             source_run: torch.Tensor, heldout_namespace: int):
    namespaces = sorted(set(int(value) for value in source_namespace.tolist()))
    if heldout_namespace not in namespaces:
        raise ValueError(
            f"heldout source namespace {heldout_namespace} not present; available={namespaces}"
        )
    test = source_namespace == heldout_namespace
    if not bool(test.any()) or bool(test.all()):
        raise ValueError("namespace holdout must leave both train and test rows")
    train = ~test
    train_groups = sorted(set((int(namespace), int(run), int(ep)) for namespace, run, ep in zip(
        source_namespace[train].tolist(), source_run[train].tolist(), episode_id[train].tolist())))
    test_groups = sorted(set((int(namespace), int(run), int(ep)) for namespace, run, ep in zip(
        source_namespace[test].tolist(), source_run[test].tolist(), episode_id[test].tolist())))
    return train, test, train_groups, test_groups


def _standardize(sequence: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    train_values = sequence[train].reshape(-1, sequence.shape[-1])
    mean = train_values.mean(dim=0)
    scale = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (sequence - mean.view(*([1] * (sequence.ndim - 1)), -1)) / scale.view(*([1] * (sequence.ndim - 1)), -1)


def _fit_variant(name: str, blocks: Dict[str, torch.Tensor], target: torch.Tensor,
                 aux: torch.Tensor, train: torch.Tensor, test: torch.Tensor,
                 device: torch.device, epochs: int, batch_size: int, seed: int,
                 episode_id: torch.Tensor, source_namespace: torch.Tensor,
                 source_run: torch.Tensor, noise_std: torch.Tensor,
                 variant_defs: Dict[str, List[str]]) -> Dict[str, object]:
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
    model = TemporalBridge(variant_defs[name], blocks["H"].shape[-1], blocks["E"].shape[-1],
                           blocks["I"].shape[-1], blocks["A"].shape[-1]).to(device)
    device_blocks = {key: blocks[key].to(device) for key in variant_defs[name]}
    device_return = y_return_norm.to(device)
    device_aux = y_aux.to(device)
    device_train_idx = train.nonzero(as_tuple=False).flatten().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    training_history = []
    for epoch in range(epochs):
        model.train()
        loss_sums = torch.zeros(3, device=device)
        permutation = device_train_idx[torch.randperm(device_train_idx.numel(), device=device)]
        for start in range(0, permutation.numel(), batch_size):
            idx = permutation[start:start + batch_size]
            batch = {key: device_blocks[key][idx] for key in variant_defs[name]}
            pred = model(batch)
            loss_return = nn.functional.mse_loss(pred[:, 0], device_return[idx])
            loss_binary = nn.functional.binary_cross_entropy_with_logits(pred[:, 1:3], device_aux[idx, :2])
            loss_cont = nn.functional.mse_loss(pred[:, 3:], device_aux[idx, 2:])
            loss = loss_return + 0.2 * loss_binary + 0.2 * loss_cont
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sums += torch.stack((loss_return, loss_binary, loss_cont)).detach() * idx.numel()
        epoch_losses = (loss_sums / permutation.numel()).cpu().tolist()
        training_history.append(dict(zip(
            ("return_mse_normalized", "binary_bce", "continuous_mse_normalized"), epoch_losses)))
        training_history[-1]["epoch"] = epoch + 1
    model.eval()
    with torch.no_grad():
        predictions = []
        for start in range(0, target.shape[0], batch_size):
            idx = torch.arange(start, min(start + batch_size, target.shape[0]))
            batch = {key: device_blocks[key][idx] for key in variant_defs[name]}
            predictions.append(model(batch)[:, 0].cpu())
        prediction = torch.cat(predictions) * ret_scale + ret_mean
    errors = (prediction - target.float()).abs()
    test_groups = list(zip(source_namespace[test].tolist(), source_run[test].tolist(), episode_id[test].tolist()))
    unique_test_groups = sorted(set(test_groups))
    episode_balanced_mae = float(torch.stack([
        errors[test][torch.tensor([group == g for group in test_groups])].mean()
        for g in unique_test_groups
    ]).mean())
    return {
        "name": name,
        "training_history": training_history,
        "input_dims": {key: list(blocks[key].shape[1:]) for key in variant_defs[name]},
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_mae": float(errors[train].mean()),
        "test_mae": float(errors[test].mean()),
        "test_episode_balanced_mae": episode_balanced_mae,
        "test_rmse": float(torch.sqrt(((prediction[test] - target[test].float()) ** 2).mean())),
        "test_predictions": prediction[test],
        "test_targets": target[test].float(),
        "test_episode_id": episode_id[test],
        "test_source_namespace": source_namespace[test],
        "test_source_run": source_run[test],
        "test_noise_std": noise_std[test],
    }


def _bootstrap_delta(base: Dict[str, object], variant: Dict[str, object], seed: int,
                     repeats: int = 1000) -> Dict[str, object]:
    groups = sorted(set((int(namespace), int(run), int(ep)) for namespace, run, ep in zip(
        base["test_source_namespace"].tolist(), base["test_source_run"].tolist(),
        base["test_episode_id"].tolist())))
    base_errors = torch.abs(base["test_predictions"] - base["test_targets"])
    variant_errors = torch.abs(variant["test_predictions"] - variant["test_targets"])
    base_groups = list(zip(base["test_source_namespace"].tolist(), base["test_source_run"].tolist(), base["test_episode_id"].tolist()))
    variant_groups = list(zip(variant["test_source_namespace"].tolist(), variant["test_source_run"].tolist(), variant["test_episode_id"].tolist()))
    if set(base_groups) != set(variant_groups):
        raise ValueError("bootstrap arms do not share the same held-out episode groups")
    base_by_group = {g: base_errors[torch.tensor([x == g for x in base_groups])].mean() for g in groups}
    variant_by_group = {g: variant_errors[torch.tensor([x == g for x in variant_groups])].mean() for g in groups}
    deltas = torch.tensor([float(base_by_group[g] - variant_by_group[g]) for g in groups])
    sampled = torch.randint(0, len(groups), (repeats, len(groups)), generator=torch.Generator().manual_seed(seed))
    bootstrap = deltas[sampled].mean(dim=1)
    quantiles = torch.quantile(bootstrap, torch.tensor([0.025, 0.975]))
    base_mae = float(torch.stack(list(base_by_group.values())).mean())
    variant_mae = float(torch.stack(list(variant_by_group.values())).mean())
    return {
        "base_mae": base_mae,
        "variant_mae": variant_mae,
        "row_weighted_base_mae": float(base_errors.mean()),
        "row_weighted_variant_mae": float(variant_errors.mean()),
        "absolute_mae_reduction": base_mae - variant_mae,
        "relative_mae_reduction": (base_mae - variant_mae) / max(base_mae, 1e-8),
        "episode_bootstrap_delta_ci95": [float(quantiles[0]), float(quantiles[1])],
        "episode_count": len(groups),
        "bootstrap_group_key": "(source_namespace, source_run, episode_id)",
    }


def _episode_error_table(fitted: Dict[str, object]) -> List[Dict[str, object]]:
    """Return a compact audit table for the held-out episode estimand.

    The public report intentionally omits row-level predictions, but retaining
    one row per held-out episode makes it possible to see whether an aggregate
    MAE is being driven by a single episode or by a broad effect.  This table
    is descriptive only; the predeclared episode-balanced MAE and bootstrap CI
    remain the gate statistics.
    """
    source_namespace = fitted["test_source_namespace"].tolist()
    source_run = fitted["test_source_run"].tolist()
    episode_id = fitted["test_episode_id"].tolist()
    prediction = fitted["test_predictions"]
    target = fitted["test_targets"]
    errors = torch.abs(prediction - target)
    groups = sorted(set((int(namespace), int(run), int(ep))
                        for namespace, run, ep in zip(source_namespace, source_run, episode_id)))
    table = []
    for namespace, run, ep in groups:
        mask = torch.tensor([(int(ns), int(r), int(e)) == (namespace, run, ep)
                             for ns, r, e in zip(source_namespace, source_run, episode_id)], dtype=torch.bool)
        group_target = target[mask]
        group_errors = errors[mask]
        table.append({
            "source_namespace": namespace,
            "source_run": run,
            "episode_id": ep,
            "rows": int(mask.sum()),
            "return_to_go_mean": float(group_target.mean()),
            "return_to_go_min": float(group_target.min()),
            "return_to_go_max": float(group_target.max()),
            "test_mae": float(group_errors.mean()),
        })
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--model-seed", type=int, default=None,
                        help="model/minibatch RNG seed; default uses --seed, which still defines the split")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--future-action-control", action="store_true",
                        help="fit diagnostic V_HF and V_HFEI arms using on-policy future actions")
    parser.add_argument("--split-mode", choices=("pooled", "namespace_holdout"), default="pooled")
    parser.add_argument("--heldout-source-namespace", type=int,
                        help="namespace ID to hold out when --split-mode=namespace_holdout")
    args = parser.parse_args()
    model_seed = args.seed if args.model_seed is None else args.model_seed
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    torch.set_num_threads(args.num_threads)
    episode_id = dataset["episode_id"].long()
    if "source_namespace" not in dataset:
        raise ValueError("dataset missing source_namespace; actor validation requires checkpoint identity")
    source_namespace = dataset["source_namespace"].long()
    source_run = dataset["source_run"].long()
    source_namespace_keys = dataset.get("metadata", {}).get("source_namespace_keys")
    if not isinstance(source_namespace_keys, list) or not source_namespace_keys:
        raise ValueError("dataset metadata missing source_namespace_keys")
    noise_std = dataset["noise_std"].float()
    if args.split_mode == "pooled":
        train, test, train_episodes, test_episodes = _split(
            episode_id, source_namespace, source_run, args.seed)
        heldout_source_namespace = None
    else:
        if args.heldout_source_namespace is None:
            raise ValueError("namespace_holdout requires --heldout-source-namespace")
        train, test, train_episodes, test_episodes = _namespace_holdout_split(
            episode_id, source_namespace, source_run, args.heldout_source_namespace)
        heldout_source_namespace = args.heldout_source_namespace
    blocks = _features(dataset)
    variant_defs = dict(VARIANTS)
    if args.future_action_control:
        if "F" not in blocks:
            raise ValueError("--future-action-control requires future_action in the dataset")
        variant_defs.update(FUTURE_ACTION_VARIANTS)
    target = dataset["return_to_go"].float()
    aux = dataset["episode_auxiliary"].float()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    report = {
        "schema": "ref2dex.gate1_value_bridge.v2",
        "input": str(args.input.resolve()),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_seed": args.seed,
        "model_seed": model_seed,
        "future_action_control": args.future_action_control,
        "split_unit": "(source_namespace, source_run, episode_id)",
        "split_mode": args.split_mode,
        "heldout_source_namespace": heldout_source_namespace,
        "source_namespace_keys": source_namespace_keys,
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "primary_metric": "held-out episode-balanced MAE of exact return_to_go",
        "secondary_metric": "row-weighted held-out MAE",
        "temporal_encoder": "GRU(hidden=64) for H/E/I; no sequence flattening",
        "variants": {},
        "comparisons_vs_V_H": {},
    }
    standardized_blocks = {key: _standardize(value, train) for key, value in blocks.items()}
    fitted = {}
    for name in variant_defs:
        fitted[name] = _fit_variant(name, standardized_blocks, target, aux, train, test, device,
                                     args.epochs, args.batch_size, model_seed,
                                     episode_id, source_namespace, source_run, noise_std, variant_defs)
        report["variants"][name] = {
            k: v for k, v in fitted[name].items() if not isinstance(v, torch.Tensor)
        }
        report["variants"][name]["heldout_episode_error_table"] = _episode_error_table(fitted[name])
    for offset, name in enumerate(variant_defs):
        if name != "V_H":
            report["comparisons_vs_V_H"][name] = _bootstrap_delta(
                fitted["V_H"], fitted[name], args.seed + 1000 + offset)
    if args.future_action_control:
        report["comparisons_vs_V_HF"] = {
            "V_HFEI": _bootstrap_delta(
                fitted["V_HF"], fitted["V_HFEI"], args.seed + 2000)
        }
    report["status"] = "FIT_COMPLETED"
    report["no_policy_training"] = True
    report["no_online_cm"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
