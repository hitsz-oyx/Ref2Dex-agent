#!/usr/bin/env python3
"""Probe conservative Cm synthetic transitions as a critic-training target.

The frozen physical model is used only to produce a one-step value-equivalence
target.  It never changes the actor observation or action.  A copy of the
frozen direct-Q is fine-tuned on a fixed blend of the real complete return and
the model target, then evaluated on the deterministic held episode split.
"""
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

from src.task.CmResidual.physical_value_contract import advance_events
from src.task.CmResidual.physical_value_data import Episodes
from src.task.CmResidual.physical_value_models import (
    Features, OutcomeNetwork, dynamics_output,
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
        model.eval()
        dynamics.append(model)
    return payload, features, direct_q, dynamics


def metrics(prediction: torch.Tensor, target: torch.Tensor, episode: torch.Tensor):
    x, y = prediction.numpy(), target.numpy()
    pred_episode, target_episode = [], []
    for identifier in torch.unique(episode):
        selected = episode == identifier
        pred_episode.append(float(prediction[selected].mean()))
        target_episode.append(float(target[selected].mean()))
    return {
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "mae": float(np.mean(np.abs(x - y))),
        "row_spearman": float(spearmanr(x, y).statistic),
        "episode_spearman": float(spearmanr(pred_episode, target_episode).statistic),
    }


@torch.no_grad()
def synthetic_target(dataset, indices, payload, features, direct_q, dynamics, device,
                     batch_size, uncertainty_gate):
    """Return conservative one-step target, uncertainty, and gate for indices."""
    targets, uncertainties, gates, returns, episodes, directs = [], [], [], [], [], []
    gamma = float(payload["gamma"])
    reward_scale = payload["reward_scale"].to(device).clamp_min(1)
    state_scale = features.state_std
    for start in range(0, len(indices), batch_size):
        current = dataset.batch(indices[start:start + batch_size], device)
        history = features.history(current["history_state"], current["history_action"],
                                    current["history_mask"])
        direct = direct_q(history, features.context(current["context"]),
                          current["action"]).squeeze(-1)
        state = current["state"]
        values, physical = [], []
        for model in dynamics:
            out, contact, reward, terminal = dynamics_output(
                model, features, current["history_state"], current["history_action"],
                current["history_mask"], current["context"], current["action"])
            out = out.clone()
            out[:, 49:51] = contact.sigmoid()
            out[:, 51:55], _, _ = advance_events(
                state[:, 51:55], out[:, 38], (out[:, 49:51] > .5).all(-1),
                current["context"][:, 6])
            # Keep the hand and object orientation on the observed state.  Cm
            # supplies only object translation/velocity and contact/event
            # consequences to the synthetic critic transition.
            projected = out.clone()
            projected[:, :36] = state[:, :36]
            projected[:, 39:43] = state[:, 39:43]
            next_history_state = torch.cat((current["history_state"][:, 1:], projected[:, None]), 1)
            next_history_action = torch.cat((current["history_action"][:, 1:], current["action"][:, None]), 1)
            next_history_mask = torch.cat((current["history_mask"][:, 1:],
                                           torch.ones_like(current["history_mask"][:, :1])), 1)
            continuation = direct_q(
                features.history(next_history_state, next_history_action, next_history_mask),
                features.context(current["next_context"]), current["action"]).squeeze(-1)
            values.append(reward + gamma * (1 - terminal.sigmoid()) * continuation)
            physical.append(torch.cat((
                (projected[:, 36:39] - state[:, 36:39]) /
                state_scale[36:39].clamp_min(.01),
                projected[:, 43:49] / state_scale[43:49].clamp_min(.01),
                out[:, 49:51], reward[:, None] / reward_scale,
                terminal.sigmoid()[:, None],
            ), dim=-1))
        values = torch.stack(values)
        physical = torch.stack(physical)
        target = values.mean(0) - values.std(0, unbiased=False)
        uncertainty = physical.std(0, unbiased=False).square().mean(-1).sqrt()
        gate = ((uncertainty <= uncertainty_gate) & ~current["done"] &
                torch.isfinite(target) & torch.isfinite(uncertainty))
        targets.append(target.cpu())
        uncertainties.append(uncertainty.cpu())
        gates.append(gate.cpu())
        returns.append(current["return"].cpu())
        episodes.append(current["episode_id"].cpu())
        directs.append(direct.cpu())
    return {
        "model_target": torch.cat(targets),
        "uncertainty": torch.cat(uncertainties),
        "gate": torch.cat(gates),
        "return": torch.cat(returns),
        "episode": torch.cat(episodes),
        "direct": torch.cat(directs),
    }


@torch.no_grad()
def predict_q(dataset, indices, network, features, device, batch_size):
    predictions = []
    for start in range(0, len(indices), batch_size):
        b = dataset.batch(indices[start:start + batch_size], device)
        predictions.append(network(
            features.history(b["history_state"], b["history_action"], b["history_mask"]),
            features.context(b["context"]), b["action"]).squeeze(-1).cpu())
    return torch.cat(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-fit-rows", type=int, default=120000)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--uncertainty-gate", type=float, default=.5)
    parser.add_argument("--model-weight", type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.model_weight < 1:
        raise ValueError("model weight must be in (0,1)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Episodes(args.collections)
    device = torch.device(args.device)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    generator = torch.Generator(device="cpu").manual_seed(2026100307)
    fit_indices = dataset.fit
    if args.max_fit_rows and len(fit_indices) > args.max_fit_rows:
        fit_indices = fit_indices[torch.randperm(len(fit_indices), generator=generator)[:args.max_fit_rows]]
    held_indices = dataset.hold
    fit_values = synthetic_target(dataset, fit_indices, payload, features, direct_q, dynamics,
                                  device, args.batch_size, args.uncertainty_gate)
    held_values = synthetic_target(dataset, held_indices, payload, features, direct_q, dynamics,
                                   device, args.batch_size, args.uncertainty_gate)
    weight = args.model_weight
    fit_target = torch.where(
        fit_values["gate"],
        (1 - weight) * fit_values["return"] + weight * fit_values["model_target"],
        fit_values["return"],
    )
    held_target = held_values["return"]
    baseline = metrics(held_values["direct"], held_target, held_values["episode"])
    model_target_metrics = metrics(held_values["model_target"], held_target, held_values["episode"])

    augmented = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    augmented.load_state_dict(payload["direct_q"], strict=True)
    augmented.train()
    optimizer = torch.optim.Adam(augmented.parameters(), lr=1e-4)
    return_scale = payload["return_scale"].to(device).clamp_min(1)
    for update in range(args.updates):
        positions = torch.randint(len(fit_indices), (args.batch_size,), generator=generator)
        indices = fit_indices[positions]
        b = dataset.batch(indices, device)
        prediction = augmented(
            features.history(b["history_state"], b["history_action"], b["history_mask"]),
            features.context(b["context"]), b["action"]).squeeze(-1)
        target = fit_target[positions].to(device)
        loss = ((prediction - target) / return_scale).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite augmented critic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(augmented.parameters(), 5)
        optimizer.step()
    augmented.eval()
    augmented_pred = predict_q(dataset, held_indices, augmented, features, device, args.batch_size)
    trained = metrics(augmented_pred, held_target, held_values["episode"])
    held_gate = held_values["gate"]
    augmented_checkpoint = args.output.parent / "augmented_direct_q.pt"
    torch.save({"schema": "ref2dex.cm_model_based_augmented_q.v1",
                "state_dict": {key: value.detach().cpu() for key, value in augmented.state_dict().items()},
                "source_checkpoint": str(args.checkpoint.resolve()),
                "source_checkpoint_sha256": sha(args.checkpoint.resolve()),
                "fit_rows": len(fit_indices), "updates": args.updates,
                "uncertainty_gate": args.uncertainty_gate,
                "model_weight": args.model_weight}, augmented_checkpoint)
    result = {
        "schema": "ref2dex.cm_model_based_critic_probe.v1",
        "run_status": "COMPLETED",
        "collections": [str(path.resolve()) for path in args.collections],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "augmented_q_checkpoint": str(augmented_checkpoint.resolve()),
        "augmented_q_checkpoint_sha256": sha(augmented_checkpoint),
        "rows": {"fit_used": len(fit_indices), "held": len(held_indices)},
        "episodes": {"fit_used": int(torch.unique(fit_values["episode"]).numel()),
                     "held": int(torch.unique(held_values["episode"]).numel())},
        "contract": {
            "projection": "observed hand q/dq and object orientation; Cm object translation/velocity/contact/events",
            "target": "conservative ensemble one-step reward plus frozen direct-Q continuation",
            "actor_input_changed": False, "actor_action_changed": False,
            "success_label_used": False, "uncertainty_gate": args.uncertainty_gate,
            "model_weight": args.model_weight,
        },
        "gate": {
            "fit_fraction": float(fit_values["gate"].float().mean()),
            "held_fraction": float(held_gate.float().mean()),
            "fit_uncertainty_p50": float(fit_values["uncertainty"].median()),
            "held_uncertainty_p50": float(held_values["uncertainty"].median()),
        },
        "held_models": {
            "frozen_direct_q": baseline,
            "frozen_cm_model_target": model_target_metrics,
            "direct_q_after_cm_target_finetune": trained,
        },
        "delta_vs_direct": {
            key: trained[key] - baseline[key]
            for key in baseline
        },
        "probe_label": "PROMISING" if (
            trained["rmse"] < baseline["rmse"] and
            trained["mae"] < baseline["mae"] and
            trained["episode_spearman"] >= baseline["episode_spearman"] - .01 and
            float(fit_values["gate"].float().mean()) >= .20
        ) else "UNPROMISING",
        "interpretation": "Offline critic-training Probe only; no actor or policy utility claim",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
