#!/usr/bin/env python3
"""Matched nonlinear task-value representation Probe.

The direct-Q checkpoint and Cm dynamics remain frozen.  Two equal-size residual
MLPs are trained on the same fit transitions: one sees current state/action plus
direct-Q, the other additionally sees only Cm-predicted physical consequences and
ensemble uncertainty.  The target is realized complete return minus direct-Q;
future state and success labels are never inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_cm_critic_residual import extract, load_models
from src.task.CmResidual.physical_value_data import Episodes


def metrics(prediction, target, episode):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    pred_episode, target_episode = [], []
    for identifier in np.unique(episode):
        selected = episode == identifier
        pred_episode.append(float(prediction[selected].mean()))
        target_episode.append(float(target[selected].mean()))
    return {
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "spearman": float(spearmanr(prediction, target).statistic),
        "episode_spearman": float(spearmanr(pred_episode, target_episode).statistic),
    }


def episode_error_bootstrap(predictions, target, episode, draws=10000):
    unique = np.unique(episode)
    rows = []
    for identifier in unique:
        selected = episode == identifier
        rows.append([
            float(np.sqrt(np.mean((predictions[name][selected] - target[selected]) ** 2)))
            for name in predictions
        ] + [
            float(np.mean(np.abs(predictions[name][selected] - target[selected])))
            for name in predictions
        ])
    values = np.asarray(rows)
    rng = np.random.default_rng(20261003)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[indices].mean(1)
    names = list(predictions)
    result = {}
    for metric, offset in (("rmse", 0), ("mae", len(names))):
        result[metric] = {}
        for name in names[1:]:
            column = names.index(name) + offset
            direct_column = names.index("direct") + offset
            delta = sampled[:, column] - sampled[:, direct_column]
            result[metric][name + "_vs_direct"] = {
                "mean": float(np.mean(delta)),
                "lower90": float(np.percentile(delta, 5)),
                "upper90": float(np.percentile(delta, 95)),
            }
    return result


class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_predict(x_fit, residual_fit, x_eval, device, epochs=5, batch_size=4096, seed=20261003):
    mean = x_fit.mean(0)
    scale = x_fit.std(0).clamp_min(1e-6)
    target_mean = residual_fit.mean()
    target_scale = residual_fit.std().clamp_min(1e-6)
    x_fit = ((x_fit - mean) / scale).to(device)
    residual_fit = ((residual_fit - target_mean) / target_scale).to(device)
    x_eval = ((x_eval - mean) / scale).to(device)
    torch.manual_seed(seed)
    model = ResidualMLP(x_fit.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_fit), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size].to(device)
            prediction = model(x_fit[rows])
            loss = (prediction - residual_fit[rows]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        losses.append(total / len(order))
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_eval), batch_size):
            predictions.append(model(x_eval[start:start + batch_size]).cpu())
    return (
        torch.cat(predictions) * target_scale.cpu() + target_mean.cpu(),
        losses,
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_mean": mean.cpu(),
            "input_scale": scale.cpu(),
            "target_mean": target_mean.cpu(),
            "target_scale": target_scale.cpu(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cm-model-output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    fit_values = extract(dataset, dataset.fit, payload, features, direct_q, dynamics, device, 512)
    held_values = extract(dataset, dataset.hold, payload, features, direct_q, dynamics, device, 512)
    fit_target = fit_values["target"]
    held_target = held_values["target"]
    fit_direct = fit_values["direct"]
    held_direct = held_values["direct"]
    fit_residual = fit_target - fit_direct
    held_residual = held_target - held_direct
    base_fit = fit_values["base"].float()
    base_hold = held_values["base"].float()
    cm_fit = fit_values["cm"].float()
    cm_hold = held_values["cm"].float()
    base_prediction, base_losses, _ = train_predict(
        base_fit, fit_residual, base_hold, device, args.epochs, args.batch_size, seed=20261003
    )
    cm_prediction, cm_losses, cm_artifact = train_predict(
        cm_fit, fit_residual, cm_hold, device, args.epochs, args.batch_size, seed=20261004
    )
    target = held_target.numpy()
    direct = held_direct.numpy()
    episode = held_values["episode"].numpy()
    reports = {
        "direct": metrics(direct, target, episode),
        "direct_plus_base_mlp": metrics(direct + base_prediction.numpy(), target, episode),
        "direct_plus_cm_mlp": metrics(direct + cm_prediction.numpy(), target, episode),
    }
    predictions = {
        "direct": direct,
        "base_mlp": direct + base_prediction.numpy(),
        "cm_mlp": direct + cm_prediction.numpy(),
    }
    result = {
        "schema": "ref2dex.cm_task_representation_audit.v1",
        "run_status": "COMPLETED",
        "rows": {"fit": int(len(fit_target)), "held": int(len(held_target))},
        "fit_episodes": int(torch.unique(fit_values["episode"]).numel()),
        "held_episodes": int(torch.unique(held_values["episode"]).numel()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        "architecture": {
            "hidden": [128, 128],
            "activation": "SiLU",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam(lr=1e-3)",
            "target": "realized complete-return minus frozen direct-Q",
        },
        "models": reports,
        "episode_error_bootstrap": episode_error_bootstrap(predictions, target, episode),
        "delta_cm_vs_base_mlp": {
            key: reports["direct_plus_cm_mlp"][key] - reports["direct_plus_base_mlp"][key]
            for key in reports["direct_plus_cm_mlp"]
        },
        "training": {
            "base_losses": base_losses,
            "cm_losses": cm_losses,
            "cm_feature_contract": "frozen ensemble predicted object delta/velocity/contact/reward/terminal plus uncertainty; no future state",
        },
        "interpretation": "Offline matched nonlinear task-representation Probe; no actor, policy, or success claim",
    }
    if args.cm_model_output:
        artifact = {
            "schema": "ref2dex.cm_task_residual_mlp.v1",
            "input_contract": "direct-Q, candidate action, current object state, Cm ensemble physical mean/std; no future state",
            "architecture": {"hidden": [128, 128], "activation": "SiLU"},
            "checkpoint_sha256_source": result["checkpoint_sha256"],
            **cm_artifact,
        }
        args.cm_model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.cm_model_output)
        result["cm_model_output"] = str(args.cm_model_output.resolve())
        result["cm_model_sha256"] = hashlib.sha256(args.cm_model_output.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
