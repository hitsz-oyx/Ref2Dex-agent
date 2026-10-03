#!/usr/bin/env python3
"""Fine-tune Cm only on fit rows selected by frozen ensemble uncertainty."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task.CmResidual.physical_value_contract import advance_events  # noqa: E402
from src.task.CmResidual.physical_value_data import Episodes  # noqa: E402
from src.task.CmResidual.physical_value_models import (  # noqa: E402
    Features, OutcomeNetwork, dynamics_loss, dynamics_output,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("physical value model schema mismatch")
    stats = {key: value.to(device) for key, value in payload["stats"].items()}
    features = Features(**stats, device=device).to(device).eval()
    direct_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    direct_q.load_state_dict(payload["direct_q"], strict=True)
    direct_q.eval()
    dynamics = []
    for index, state in enumerate(payload["dynamics"]):
        model = OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + index).to(device)
        model.load_state_dict(state, strict=True)
        dynamics.append(model)
    return payload, features, direct_q, dynamics


def physical_vector(predicted, contact, reward, terminal, current, batch, features, payload):
    state_std = features.state_std
    reward_scale = payload["reward_scale"].to(current.device).clamp_min(1)
    predicted = predicted.clone()
    predicted[:, 49:51] = contact.sigmoid()
    predicted[:, 51:55], _, _ = advance_events(
        current[:, 51:55], predicted[:, 38], predicted[:, 49:51].bool().all(-1),
        batch["context"][:, 6])
    actual = torch.cat((
        (batch["next_state"][:, 36:39] - current[:, 36:39]) /
        state_std[36:39].clamp_min(.01),
        batch["next_state"][:, 43:49] / state_std[43:49].clamp_min(.01),
        batch["next_state"][:, 49:51], batch["next_state"][:, 51:55],
        batch["reward"][:, None] / reward_scale, batch["done"].float()[:, None],
    ), dim=-1)
    forecast = torch.cat((
        (predicted[:, 36:39] - current[:, 36:39]) /
        state_std[36:39].clamp_min(.01),
        predicted[:, 43:49] / state_std[43:49].clamp_min(.01),
        predicted[:, 49:51], predicted[:, 51:55],
        reward[:, None] / reward_scale, terminal.sigmoid()[:, None],
    ), dim=-1)
    return predicted, actual, forecast


@torch.no_grad()
def physical_screen(dataset, indices, payload, features, direct_q, dynamics, device, batch_size):
    means, stds, actuals, values, episodes = [], [], [], [], []
    gamma = float(payload["gamma"])
    for start in range(0, len(indices), batch_size):
        batch = dataset.batch(indices[start:start + batch_size], device)
        history = features.history(batch["history_state"], batch["history_action"], batch["history_mask"])
        current = batch["state"]
        physical, continuations = [], []
        actual_batch = None
        for model in dynamics:
            predicted, contact, reward, terminal = dynamics_output(
                model, features, batch["history_state"], batch["history_action"],
                batch["history_mask"], batch["context"], batch["action"])
            predicted, actual, forecast = physical_vector(
                predicted, contact, reward, terminal, current, batch, features, payload)
            physical.append(forecast)
            actual_batch = actual
            projected = predicted.clone()
            projected[:, :36] = current[:, :36]
            projected[:, 39:43] = current[:, 39:43]
            next_states = torch.cat((batch["history_state"][:, 1:], projected[:, None]), 1)
            next_actions = torch.cat((batch["history_action"][:, 1:], batch["action"][:, None]), 1)
            next_mask = torch.cat((batch["history_mask"][:, 1:], torch.ones_like(batch["history_mask"][:, :1])), 1)
            continuation = direct_q(
                features.history(next_states, next_actions, next_mask),
                features.context(batch["next_context"]), batch["action"]).squeeze(-1)
            continuations.append(reward + gamma * (1 - terminal.sigmoid()) * continuation)
        physical = torch.stack(physical)
        means.append(physical.mean(0).cpu())
        stds.append(physical.std(0, unbiased=False).cpu())
        # actual is identical across models; retain one copy per batch.
        actuals.append(actual_batch.cpu())
        values.append((torch.stack(continuations).mean(0) - torch.stack(continuations).std(0, unbiased=False)).cpu())
        episodes.append(batch["episode_id"].cpu())
    return {"mean": torch.cat(means), "std": torch.cat(stds), "actual": torch.cat(actuals),
            "value": torch.cat(values), "episode": torch.cat(episodes)}


def train_high_uncertainty(dataset, indices, selected, payload, features, dynamics, device,
                           updates, batch_size, seed):
    optimizer = torch.optim.Adam([parameter for model in dynamics for parameter in model.parameters()], lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reward_scale = payload["reward_scale"].to(device)
    for model in dynamics:
        model.train()
    losses = []
    for _ in range(updates):
        positions = torch.randint(len(selected), (batch_size,), generator=generator)
        batch = dataset.batch(selected[positions], device)
        total = 0.0
        for model in dynamics:
            out, contact, reward, terminal = dynamics_output(
                model, features, batch["history_state"], batch["history_action"],
                batch["history_mask"], batch["context"], batch["action"])
            total = total + dynamics_loss(
                out, contact, reward, terminal, batch["next_state"], batch["reward"],
                batch["done"], features, reward_scale) / len(dynamics)
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite targeted Cm loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([parameter for model in dynamics for parameter in model.parameters()], 5)
        optimizer.step()
        losses.append(float(total.detach()))
    for model in dynamics:
        model.eval()
    return losses


def metrics(screen, mask):
    error = screen["mean"] - screen["actual"]
    selected = error[mask]
    return {
        "physical_rmse": float(error.square().mean().sqrt()),
        "physical_mae": float(error.abs().mean()),
        "selected_rmse": float(selected.square().mean().sqrt()),
        "selected_mae": float(selected.abs().mean()),
    }


def value_metrics(prediction, target, episode):
    x, y = prediction.numpy(), target.numpy()
    p, t = [], []
    for identifier in torch.unique(episode):
        selected = episode == identifier
        p.append(float(prediction[selected].mean()))
        t.append(float(target[selected].mean()))
    return {"rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "mae": float(np.mean(np.abs(x - y))),
            "row_spearman": float(spearmanr(x, y).statistic),
            "episode_spearman": float(spearmanr(p, t).statistic)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-fit-rows", type=int, default=120000)
    parser.add_argument("--max-held-rows", type=int, default=0)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    generator = torch.Generator(device="cpu").manual_seed(2026100314)
    fit_indices = dataset.fit
    if args.max_fit_rows and len(fit_indices) > args.max_fit_rows:
        fit_indices = fit_indices[torch.randperm(len(fit_indices), generator=generator)[:args.max_fit_rows]]
    fit_indices = fit_indices.sort().values
    fit_screen = physical_screen(dataset, fit_indices, payload, features, direct_q, dynamics, device, args.batch_size)
    uncertainty = fit_screen["std"].square().mean(-1).sqrt()
    threshold = torch.quantile(uncertainty, .8)
    selected = fit_indices[uncertainty >= threshold]
    original_dynamics = copy.deepcopy(dynamics)
    losses = train_high_uncertainty(dataset, fit_indices, selected, payload, features, dynamics,
                                    device, args.updates, args.batch_size, 2026100315)
    held_indices = dataset.hold
    if args.max_held_rows and len(held_indices) > args.max_held_rows:
        held_indices = held_indices[torch.randperm(len(held_indices), generator=torch.Generator(device="cpu").manual_seed(2026100316))[:args.max_held_rows]].sort().values
    held_screen_original = physical_screen(dataset, held_indices, payload, features, direct_q,
                                           original_dynamics, device, args.batch_size)
    held_screen_adapted = physical_screen(dataset, held_indices, payload, features, direct_q,
                                          dynamics, device, args.batch_size)
    held_uncertainty = held_screen_original["std"].square().mean(-1).sqrt()
    held_mask = held_uncertainty >= torch.quantile(held_uncertainty, .8)
    physical_original = metrics(held_screen_original, held_mask)
    physical_adapted = metrics(held_screen_adapted, held_mask)
    target_original = value_metrics(held_screen_original["value"],
                                    dataset.data["return"][held_indices], held_screen_original["episode"])
    target_adapted = value_metrics(held_screen_adapted["value"],
                                   dataset.data["return"][held_indices], held_screen_adapted["episode"])
    top_improvement = (physical_adapted["selected_rmse"] - physical_original["selected_rmse"]) / max(physical_original["selected_rmse"], 1e-8)
    overall_change = (physical_adapted["physical_rmse"] - physical_original["physical_rmse"]) / max(physical_original["physical_rmse"], 1e-8)
    target_delta = {key: target_adapted[key] - target_original[key] for key in target_original}
    gates = {
        "top_uncertainty_physical_rmse_improves_10pct": top_improvement <= -.10,
        "overall_physical_rmse_not_worse_2pct": overall_change <= .02,
        "value_target_rmse_improves_0_5": target_delta["rmse"] <= -.5,
        "value_episode_spearman_loss_within_0_01": target_delta["episode_spearman"] >= -.01,
    }
    result = {
        "schema": "ref2dex.cm_uncertainty_retraining_probe.v1",
        "run_status": "COMPLETED",
        "experiment_id": "P-20261003-cm-uncertainty-retraining",
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "fit_rows": len(fit_indices), "selected_high_uncertainty_rows": len(selected),
        "held_rows": len(held_indices), "held_episodes": int(torch.unique(held_screen_original["episode"]).numel()),
        "contract": {"selection": "fit-only top20% frozen disagreement", "updates": args.updates,
                     "orientation_included": False, "actor_changed": False, "reward_changed": False,
                     "ordinary_data_expanded": False},
        "fit_uncertainty_threshold": float(threshold),
        "training_loss": {"first": losses[0], "last": losses[-1], "min": min(losses)},
        "held_physical": {"original": physical_original, "adapted": physical_adapted},
        "held_value_target": {"original": target_original, "adapted": target_adapted},
        "delta": {"top_physical_rmse_relative": top_improvement, "overall_physical_rmse_relative": overall_change, "value": target_delta},
        "gates": gates,
        "probe_label": "PROMISING" if all(gates.values()) else "UNPROMISING",
        "decision": "Authorize one matched policy Probe with adapted Cm bundle" if all(gates.values()) else "Close targeted uncertainty retraining; no policy follow-up",
    }
    bundle = copy.deepcopy(payload)
    bundle["dynamics"] = [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in dynamics]
    bundle["source_checkpoint_sha256"] = result["checkpoint_sha256"]
    bundle["adaptation"] = {"selection": "fit-only top20% frozen disagreement", "updates": args.updates,
                             "selected_rows": len(selected), "probe": result["experiment_id"]}
    torch.save(bundle, args.output.parent / "cm_uncertainty_adapted_physical_value.pt")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
