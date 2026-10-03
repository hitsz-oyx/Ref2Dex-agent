#!/usr/bin/env python3
"""Compare observed-vs-Cm-predicted object orientation in one-step value targets."""
from __future__ import annotations

import argparse
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
    p_episode, t_episode = [], []
    for identifier in torch.unique(episode):
        selected = episode == identifier
        p_episode.append(float(prediction[selected].mean()))
        t_episode.append(float(target[selected].mean()))
    return {
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "mae": float(np.mean(np.abs(x - y))),
        "row_spearman": float(spearmanr(x, y).statistic),
        "episode_spearman": float(spearmanr(p_episode, t_episode).statistic),
    }


@torch.no_grad()
def screen(dataset, indices, payload, features, direct_q, dynamics, device, batch_size):
    gamma = float(payload["gamma"])
    targets, direct_values, projected_values, orientation_values = [], [], [], []
    episodes, orientation_errors = [], []
    state_scale = features.state_std
    for start in range(0, len(indices), batch_size):
        batch = dataset.batch(indices[start:start + batch_size], device)
        history = features.history(batch["history_state"], batch["history_action"],
                                   batch["history_mask"])
        context = features.context(batch["context"])
        direct = direct_q(history, context, batch["action"]).squeeze(-1)
        values_projected, values_orientation, errors = [], [], []
        current = batch["state"]
        for model in dynamics:
            predicted, contact, reward, terminal = dynamics_output(
                model, features, batch["history_state"], batch["history_action"],
                batch["history_mask"], batch["context"], batch["action"])
            predicted = predicted.clone()
            predicted[:, 49:51] = contact.sigmoid()
            predicted[:, 51:55], _, _ = advance_events(
                current[:, 51:55], predicted[:, 38],
                predicted[:, 49:51].bool().all(-1), batch["context"][:, 6])
            observed = predicted.clone()
            observed[:, :36] = current[:, :36]
            observed[:, 39:43] = current[:, 39:43]
            oriented = predicted.clone()
            oriented[:, :36] = current[:, :36]
            # oriented retains the model's canonical predicted quaternion.
            def continuation(next_state):
                next_history_state = torch.cat((batch["history_state"][:, 1:],
                                                next_state[:, None]), 1)
                next_history_action = torch.cat((batch["history_action"][:, 1:],
                                                 batch["action"][:, None]), 1)
                next_mask = torch.cat((batch["history_mask"][:, 1:],
                                      torch.ones_like(batch["history_mask"][:, :1])), 1)
                return direct_q(features.history(next_history_state, next_history_action, next_mask),
                                features.context(batch["next_context"]), batch["action"]).squeeze(-1)
            values_projected.append(reward + gamma * (1 - terminal.sigmoid()) * continuation(observed))
            values_orientation.append(reward + gamma * (1 - terminal.sigmoid()) * continuation(oriented))
            dot = (oriented[:, 39:43] * current[:, 39:43]).sum(-1).abs().clamp(0, 1)
            errors.append(2 * torch.arccos(dot))
        values_projected = torch.stack(values_projected)
        values_orientation = torch.stack(values_orientation)
        targets.append(batch["return"].cpu())
        direct_values.append(direct.cpu())
        projected_values.append((values_projected.mean(0) - values_projected.std(0, unbiased=False)).cpu())
        orientation_values.append((values_orientation.mean(0) - values_orientation.std(0, unbiased=False)).cpu())
        episodes.append(batch["episode_id"].cpu())
        orientation_errors.append(torch.stack(errors).mean(0).cpu())
    return {
        "target": torch.cat(targets),
        "direct": torch.cat(direct_values),
        "observed_orientation": torch.cat(projected_values),
        "predicted_orientation": torch.cat(orientation_values),
        "episode": torch.cat(episodes),
        "orientation_error_rad": torch.cat(orientation_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    result_data = screen(dataset, dataset.hold, payload, features, direct_q, dynamics,
                         device, args.batch_size)
    baseline = metrics(result_data["direct"], result_data["target"], result_data["episode"])
    observed = metrics(result_data["observed_orientation"], result_data["target"], result_data["episode"])
    predicted = metrics(result_data["predicted_orientation"], result_data["target"], result_data["episode"])
    delta = {key: predicted[key] - observed[key] for key in observed}
    gates = {
        "rmse_improvement_at_least_0_5_vs_observed_orientation": delta["rmse"] <= -.5,
        "episode_spearman_loss_within_0_01": delta["episode_spearman"] >= -.01,
        "finite_orientation_error": bool(torch.isfinite(result_data["orientation_error_rad"]).all()),
    }
    result = {
        "schema": "ref2dex.cm_orientation_consequence_probe.v1",
        "run_status": "COMPLETED",
        "experiment_id": "P-20261003-cm-orientation-consequence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "rows": len(result_data["target"]),
        "held_episodes": int(torch.unique(result_data["episode"]).numel()),
        "contract": {"hand_q_dq_observed": True, "object_translation_velocity_cm": True,
                     "object_orientation": "treatment_cm_predicted_vs_control_observed",
                     "actor_changed": False, "reward_changed": False,
                     "success_predictor": False},
        "held_models": {"direct_q": baseline, "observed_orientation": observed,
                        "predicted_orientation": predicted},
        "delta_predicted_vs_observed": delta,
        "orientation_error_rad": {
            "mean": float(result_data["orientation_error_rad"].mean()),
            "p90": float(torch.quantile(result_data["orientation_error_rad"], .9)),
            "finite_fraction": float(torch.isfinite(result_data["orientation_error_rad"]).float().mean()),
        },
        "gates": gates,
        "probe_label": "PROMISING" if all(gates.values()) else "UNPROMISING",
        "decision": "Authorize one fixed critic fine-tune then matched policy Probe" if all(gates.values()) else "Close object-orientation consequence contract; no critic fine-tune or policy follow-up",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
