#!/usr/bin/env python3
"""Ref4 follow-up: predicted-E-only bridge on the fixed Gate-1 split.

The preceding Ref4 bridge showed that GT E can help G while the current GT I
sequence added no independent value.  This probe therefore trains only a
small deployable-input predictor ``(H, A) -> E[1:8]`` and substitutes its
prediction into a bridge trained on GT E.  It deliberately does not train or
evaluate I.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.probe_ref4_g_bridge import Bridge, _split, _standardize


class EffectPredictor(nn.Module):
    def __init__(self, history_dim: int, action_dim: int, horizon: int, effect_dim: int):
        super().__init__()
        self.history = nn.GRU(history_dim, 64, batch_first=True)
        self.action = nn.Sequential(nn.Linear(action_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                  nn.Linear(128, horizon * effect_dim))
        self.horizon = horizon
        self.effect_dim = effect_dim

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history(history)
        fused = torch.cat((hidden[-1], self.action(action)), dim=-1)
        return self.head(fused).reshape(-1, self.horizon, self.effect_dim)


def _group_mae(pred: torch.Tensor, target: torch.Tensor, groups: List[tuple], unique: List[tuple]):
    errors = (pred - target).abs()
    return float(torch.stack([errors[torch.tensor([g == x for g in groups])].mean() for x in unique]).mean())


def _fit_predictor(history: torch.Tensor, action: torch.Tensor, effect: torch.Tensor,
                   train: torch.Tensor, test: torch.Tensor, device: torch.device,
                   epochs: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    h_mean = history[train].reshape(-1, history.shape[-1]).mean(0)
    h_scale = history[train].reshape(-1, history.shape[-1]).std(0, unbiased=False).clamp_min(1e-6)
    a_mean = action[train].mean(0)
    a_scale = action[train].std(0, unbiased=False).clamp_min(1e-6)
    e_flat = effect[train].reshape(-1, effect.shape[-1])
    e_mean = e_flat.mean(0)
    e_scale = e_flat.std(0, unbiased=False).clamp_min(1e-6)
    model = EffectPredictor(history.shape[-1], action.shape[-1], effect.shape[1], effect.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    losses = []
    for epoch in range(epochs):
        order = train_idx[torch.randperm(train_idx.numel())]
        model.train()
        total = 0.0
        count = 0
        for start in range(0, order.numel(), batch_size):
            idx = order[start:start + batch_size]
            h = ((history[idx] - h_mean.view(1, 1, -1)) / h_scale.view(1, 1, -1)).to(device)
            a = ((action[idx] - a_mean) / a_scale).to(device)
            y = ((effect[idx] - e_mean.view(1, 1, -1)) / e_scale.view(1, 1, -1)).to(device)
            loss = nn.functional.mse_loss(model(h, a), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
            count += idx.numel()
        losses.append(total / max(count, 1))
    model.eval()
    predictions = []
    test_idx = test.nonzero(as_tuple=False).flatten()
    with torch.no_grad():
        for start in range(0, test_idx.numel(), batch_size):
            idx = test_idx[start:start + batch_size]
            h = ((history[idx] - h_mean.view(1, 1, -1)) / h_scale.view(1, 1, -1)).to(device)
            a = ((action[idx] - a_mean) / a_scale).to(device)
            predictions.append((model(h, a).cpu() * e_scale.view(1, 1, -1) + e_mean.view(1, 1, -1)))
    prediction = torch.cat(predictions)
    actual = effect[test_idx]
    rmse = torch.sqrt(((prediction - actual) ** 2).mean())
    baseline = torch.sqrt(((actual - e_mean.view(1, 1, -1)) ** 2).mean())
    return prediction, {
        "epochs": epochs, "batch_size": batch_size, "training_loss": losses,
        "test_rmse": float(rmse), "test_mae": float((prediction - actual).abs().mean()),
        "train_mean_baseline_rmse": float(baseline),
        "relative_rmse_reduction_vs_train_mean": float(1.0 - rmse / baseline.clamp_min(1e-8)),
    }


def _fit_bridge(name: str, blocks: Dict[str, torch.Tensor], target: torch.Tensor,
                train: torch.Tensor, test: torch.Tensor, test_groups: List[tuple],
                device: torch.device, epochs: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    selected = ("H",) if name == "H" else ("H", "E")
    dims = {key: blocks[key].shape[-1] for key in selected}
    model = Bridge(dims, list(selected)).to(device)
    y = target.float()
    y_mean = y[train].mean(); y_scale = y[train].std(unbiased=False).clamp_min(1e-6)
    y_norm = (y - y_mean) / y_scale
    train_idx = train.nonzero(as_tuple=False).flatten()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    for _ in range(epochs):
        order = train_idx[torch.randperm(train_idx.numel())]
        model.train()
        for start in range(0, order.numel(), batch_size):
            idx = order[start:start + batch_size]
            batch = {key: blocks[key][idx].to(device) for key in selected}
            pred = model(batch)
            loss = nn.functional.mse_loss(pred, y_norm[idx].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            optimizer.step()
    model.eval(); test_idx = test.nonzero(as_tuple=False).flatten(); predictions = []
    with torch.no_grad():
        for start in range(0, test_idx.numel(), batch_size):
            idx = test_idx[start:start + batch_size]
            predictions.append((model({key: blocks[key][idx].to(device) for key in selected}) * y_scale + y_mean).cpu())
    prediction = torch.cat(predictions)
    actual = y[test_idx]
    groups = list(zip(blocks["_namespace"][test_idx].tolist(), blocks["_run"][test_idx].tolist(), blocks["_episode"][test_idx].tolist()))
    return prediction, {
        "name": name,
        "test_mae": float((prediction - actual).abs().mean()),
        "test_episode_balanced_mae": _group_mae(prediction, actual, groups, test_groups),
        "test_rmse": float(torch.sqrt(((prediction - actual) ** 2).mean())),
        "_prediction": prediction, "_target": actual, "_groups": groups,
    }


def _bootstrap(base: Dict[str, object], variant: Dict[str, object], seed: int):
    groups = sorted(set(base["_groups"]))
    be = (base["_prediction"] - base["_target"]).abs(); ve = (variant["_prediction"] - variant["_target"]).abs()
    bg = base["_groups"]; vg = variant["_groups"]
    b = torch.tensor([float(be[torch.tensor([x == g for x in bg])].mean()) for g in groups])
    v = torch.tensor([float(ve[torch.tensor([x == g for x in vg])].mean()) for g in groups])
    delta = b - v
    sample = torch.randint(0, len(groups), (2000, len(groups)), generator=torch.Generator().manual_seed(seed))
    q = torch.quantile(delta[sample].mean(1), torch.tensor([.025, .975]))
    return {"base_episode_balanced_mae": float(b.mean()), "variant_episode_balanced_mae": float(v.mean()),
            "absolute_mae_reduction": float(delta.mean()),
            "relative_mae_reduction": float(delta.mean() / b.mean().clamp_min(1e-8)),
            "episode_bootstrap_delta_ci95": [float(q[0]), float(q[1])], "episode_count": len(groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--predictor-epochs", type=int, default=8)
    parser.add_argument("--bridge-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    namespace = source.get("source_namespace", torch.zeros_like(source["source_run"]))
    horizon = min(8, source["effect"].shape[1])
    raw = {
        "H": torch.cat((source["history_state"].float(), source["history_previous_action"].float(), source["history_context"].float(), source["history_progress"].float()), dim=-1),
        "A": source["action"].float(), "E": source["effect"][:, :horizon].float(),
        "_namespace": namespace.long(), "_run": source["source_run"].long(), "_episode": source["episode_id"].long(),
    }
    train, test, train_groups, test_groups = _split(raw, args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    predicted_e, predictor_report = _fit_predictor(raw["H"], raw["A"], raw["E"], train, test, device, args.predictor_epochs, args.batch_size, args.seed)
    h = _standardize(raw["H"], train); e = _standardize(raw["E"], train)
    blocks_gt = {"H": h, "E": e, "_namespace": raw["_namespace"], "_run": raw["_run"], "_episode": raw["_episode"]}
    # The bridge is trained on GT E in the training rows.  For test rows only,
    # swap the same normalized channel with predicted E.
    train_e_flat = raw["E"][train].reshape(-1, raw["E"].shape[-1]); e_mean = train_e_flat.mean(0); e_scale = train_e_flat.std(0, unbiased=False).clamp_min(1e-6)
    pred_e_norm = (predicted_e - e_mean.view(1, 1, -1)) / e_scale.view(1, 1, -1)
    blocks_pred = {key: value.clone() for key, value in blocks_gt.items()}
    blocks_pred["E"] = blocks_gt["E"].clone(); blocks_pred["E"][test] = pred_e_norm
    # GT bridge fit and predictions are identical across GT/pred-E evaluation.
    h_pred, gt_result = _fit_bridge("HE", blocks_gt, source["return_to_go"], train, test, test_groups, device, args.bridge_epochs, args.batch_size, args.seed + 1)
    _, h_result = _fit_bridge("H", blocks_gt, source["return_to_go"], train, test, test_groups, device, args.bridge_epochs, args.batch_size, args.seed + 2)
    # Refit the HE bridge once with a deterministic seed and evaluate predicted E.
    # To avoid changing the trained GT bridge, train a copy using GT rows, then
    # feed pred-E only on the held-out rows through the same parameterization.
    # _fit_bridge returns only predictions, so a compact second fit is used here.
    selected = ("H", "E"); torch.manual_seed(args.seed + 1)
    model = Bridge({key: blocks_gt[key].shape[-1] for key in selected}, list(selected)).to(device)
    y = source["return_to_go"].float(); ym = y[train].mean(); ys = y[train].std(unbiased=False).clamp_min(1e-6); yn = (y - ym) / ys
    idx_train = train.nonzero(as_tuple=False).flatten(); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    for _ in range(args.bridge_epochs):
        order = idx_train[torch.randperm(idx_train.numel())]
        for start in range(0, order.numel(), args.batch_size):
            idx = order[start:start + args.batch_size]; loss = nn.functional.mse_loss(model({key: blocks_gt[key][idx].to(device) for key in selected}), yn[idx].to(device)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    ti = test.nonzero(as_tuple=False).flatten(); pred_values = []
    with torch.no_grad():
        for start in range(0, ti.numel(), args.batch_size):
            ix = ti[start:start + args.batch_size]; pred_values.append((model({"H": blocks_pred["H"][ix].to(device), "E": blocks_pred["E"][ix].to(device)}) * ys + ym).cpu())
    pred_values = torch.cat(pred_values)
    predicted_result = {"name": "HE_predicted_E", "_prediction": pred_values, "_target": gt_result["_target"], "_groups": gt_result["_groups"], "test_mae": float((pred_values - gt_result["_target"]).abs().mean()), "test_episode_balanced_mae": _group_mae(pred_values, gt_result["_target"], gt_result["_groups"], test_groups), "test_rmse": float(torch.sqrt(((pred_values - gt_result["_target"]) ** 2).mean()))}
    report = {"schema": "ref2dex.ref4_predicted_e_bridge.v1", "input": str(args.input.resolve()), "device": str(device), "seed": args.seed, "predictor_epochs": args.predictor_epochs, "bridge_epochs": args.bridge_epochs, "batch_size": args.batch_size, "horizon": horizon, "target": "exact Monte Carlo return_to_go G", "split_unit": "(source_namespace, source_run, episode_id)", "train_groups": train_groups, "test_groups": test_groups, "predictor": predictor_report, "bridges": {"H": {k:v for k,v in h_result.items() if not k.startswith("_")}, "HE_GT": {k:v for k,v in gt_result.items() if not k.startswith("_")}, "HE_predicted_E": {k:v for k,v in predicted_result.items() if not k.startswith("_")}}, "comparisons_vs_H": {"HE_GT": _bootstrap(h_result, gt_result, args.seed+10), "HE_predicted_E": _bootstrap(h_result, predicted_result, args.seed+11)}, "predicted_I": "not run by design: current GT-I bridge showed no independent gain", "status": "FIT_COMPLETED"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
