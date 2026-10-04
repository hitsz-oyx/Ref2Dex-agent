#!/usr/bin/env python3
"""Small Ref4 G-bridge probe on an audited Gate-1 dataset.

This is deliberately independent of the older Gate-1 fitting scripts.  It
asks the narrow Ref4 question: after a history encoder (H) is available, do
ground-truth future object consequences (E), and then interaction (I), add
predictive information about the recorded long-horizon return-to-go G?

The probe uses one fixed episode-group split and the first K=8 consequence
steps.  It does not train a consequence predictor or a policy; predicted E/I
can be added later only if a checkpoint with the same contract is available.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn


VARIANTS = {"H": ("H",), "HE": ("H", "E"), "HEI": ("H", "E", "I")}


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        return h[-1]


class Bridge(nn.Module):
    def __init__(self, dims: Dict[str, int], blocks: List[str], hidden_dim: int = 48):
        super().__init__()
        self.blocks = tuple(blocks)
        self.encoders = nn.ModuleDict({
            key: Encoder(dims[key], hidden_dim) for key in self.blocks
        })
        fusion = hidden_dim * len(self.blocks)
        self.head = nn.Sequential(
            nn.Linear(fusion, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(torch.cat([self.encoders[k](batch[k]) for k in self.blocks], dim=-1)).squeeze(-1)


def _groups(data: Dict[str, torch.Tensor]):
    return sorted(set((int(ns), int(run), int(ep)) for ns, run, ep in zip(
        data["_namespace"].tolist(), data["_run"].tolist(), data["_episode"].tolist())))


def _split(data: Dict[str, torch.Tensor], seed: int):
    groups = _groups(data)
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_test = max(1, round(0.2 * len(groups)))
    test_groups = sorted(shuffled[:n_test])
    test_set = set(test_groups)
    keys = list(zip(data["_namespace"].tolist(), data["_run"].tolist(), data["_episode"].tolist()))
    test = torch.tensor([tuple(map(int, k)) in test_set for k in keys], dtype=torch.bool)
    return ~test, test, sorted(shuffled[n_test:]), test_groups


def _standardize(x: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    flat = x[train].reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    scale = flat.std(0, unbiased=False).clamp_min(1e-6)
    shape = [1] * (x.ndim - 1) + [x.shape[-1]]
    return (x - mean.view(*shape)) / scale.view(*shape)


def _fit(name: str, blocks: Dict[str, torch.Tensor], target: torch.Tensor,
         train: torch.Tensor, test: torch.Tensor, groups_test: List[tuple],
         device: torch.device, epochs: int, batch_size: int, seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = Bridge({key: int(value.shape[-1]) for key, value in blocks.items()}, list(VARIANTS[name])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    y = target.float()
    y_mean = y[train].mean()
    y_scale = y[train].std(unbiased=False).clamp_min(1e-6)
    y_norm = (y - y_mean) / y_scale
    train_idx = train.nonzero(as_tuple=False).flatten()
    losses = []
    for epoch in range(epochs):
        model.train()
        order = train_idx[torch.randperm(train_idx.numel())]
        total = 0.0
        count = 0
        for start in range(0, order.numel(), batch_size):
            idx = order[start:start + batch_size]
            batch = {key: blocks[key][idx].to(device, non_blocking=True) for key in VARIANTS[name]}
            pred = model(batch)
            loss = nn.functional.mse_loss(pred, y_norm[idx].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
            count += idx.numel()
        losses.append(total / max(count, 1))
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, y.numel(), batch_size):
            idx = torch.arange(start, min(start + batch_size, y.numel()))
            batch = {key: blocks[key][idx].to(device, non_blocking=True) for key in VARIANTS[name]}
            predictions.append((model(batch) * y_scale + y_mean).cpu())
    prediction = torch.cat(predictions)
    errors = (prediction - y).abs()
    test_groups_rows = list(zip(
        blocks["_namespace"][test].tolist(), blocks["_run"][test].tolist(),
        blocks["_episode"][test].tolist()))
    by_group = []
    for group in groups_test:
        mask = torch.tensor([tuple(map(int, row)) == group for row in test_groups_rows], dtype=torch.bool)
        by_group.append(float(errors[test][mask].mean()))
    return {
        "name": name,
        "blocks": list(VARIANTS[name]),
        "epochs": epochs,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_mae": float(errors[train].mean()),
        "test_mae": float(errors[test].mean()),
        "test_episode_balanced_mae": float(torch.tensor(by_group).mean()),
        "test_rmse": float(torch.sqrt(((prediction[test] - y[test]) ** 2).mean())),
        "test_group_mae": by_group,
        "training_loss": losses,
        "_predictions": prediction[test],
        "_targets": y[test],
        "_groups": test_groups_rows,
    }


def _bootstrap(base: Dict[str, object], variant: Dict[str, object], seed: int, repeats: int = 2000):
    groups = sorted(set(base["_groups"]))
    base_err = (base["_predictions"] - base["_targets"]).abs()
    var_err = (variant["_predictions"] - variant["_targets"]).abs()
    bg = base["_groups"]
    vg = variant["_groups"]
    base_by = {g: base_err[torch.tensor([x == g for x in bg])].mean() for g in groups}
    var_by = {g: var_err[torch.tensor([x == g for x in vg])].mean() for g in groups}
    delta = torch.tensor([float(base_by[g] - var_by[g]) for g in groups])
    sampled = torch.randint(0, len(groups), (repeats, len(groups)), generator=torch.Generator().manual_seed(seed))
    q = torch.quantile(delta[sampled].mean(1), torch.tensor([0.025, 0.975]))
    reduction = float(delta.mean())
    return {
        "base_episode_balanced_mae": float(torch.stack(list(base_by.values())).mean()),
        "variant_episode_balanced_mae": float(torch.stack(list(var_by.values())).mean()),
        "absolute_mae_reduction": reduction,
        "relative_mae_reduction": reduction / max(float(torch.stack(list(base_by.values())).mean()), 1e-8),
        "episode_bootstrap_delta_ci95": [float(q[0]), float(q[1])],
        "episode_count": len(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--model-seed", type=int, default=20261004)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    horizon = min(args.horizon, int(source["effect"].shape[1]), int(source["interaction"].shape[1]))
    source_namespace = source.get("source_namespace")
    if source_namespace is None:
        # Older assembled e260 files contain one audited source namespace and
        # only expose source_run.  Retain that explicit fact in the split key.
        source_namespace = torch.zeros_like(source["source_run"], dtype=torch.long)
    raw = {
        "H": torch.cat((source["history_state"].float(), source["history_previous_action"].float(),
                         source["history_context"].float(), source["history_progress"].float()), dim=-1),
        "E": source["effect"][:, :horizon].float(),
        "I": source["interaction"][:, :horizon].float(),
        "_namespace": source_namespace.long(),
        "_run": source["source_run"].long(),
        "_episode": source["episode_id"].long(),
    }
    train, test, train_groups, test_groups = _split(raw, args.seed)
    blocks = {key: _standardize(value, train) for key, value in raw.items() if key in VARIANTS["HEI"]}
    # Keep identifiers out of the model while making them available to grouped metrics.
    blocks.update({key: raw[key] for key in ("_namespace", "_run", "_episode")})
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    fitted = {}
    for offset, name in enumerate(VARIANTS):
        fitted[name] = _fit(name, blocks, source["return_to_go"], train, test, test_groups,
                             device, args.epochs, args.batch_size, args.model_seed + offset)
    report = {
        "schema": "ref2dex.ref4_g_bridge_probe.v1",
        "question": "Does GT E and then GT I add independent information for long-horizon G after H?",
        "input": str(args.input.resolve()),
        "device": str(device),
        "seed": args.seed,
        "model_seed": args.model_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "consequence_horizon": horizon,
        "target": "stored exact Monte Carlo return_to_go G (no bootstrap)",
        "split_unit": "(source_namespace, source_run, episode_id)",
        "train_groups": train_groups,
        "test_groups": test_groups,
        "variants": {},
        "comparisons_vs_H": {},
        "predicted_consequence_evaluation": "not run: no matched predictor checkpoint was available",
        "decision_rule": "GT-I has independent value only if HEI improves over HE with a positive episode-bootstrap delta; this probe is exploratory, not a formal validation.",
    }
    for name, result in fitted.items():
        report["variants"][name] = {key: value for key, value in result.items() if not key.startswith("_")}
    for offset, name in enumerate(("HE", "HEI")):
        report["comparisons_vs_H"][name] = _bootstrap(fitted["H"], fitted[name], args.seed + 100 + offset)
    report["comparison_HEI_vs_HE"] = _bootstrap(fitted["HE"], fitted["HEI"], args.seed + 200)
    report["status"] = "FIT_COMPLETED"
    report["interpretation"] = "GT bridge ablation only; do not infer policy utility or predicted-I utility."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
