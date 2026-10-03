#!/usr/bin/env python3
"""Screen whether Cm ensemble disagreement identifies useful data regions."""
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


@torch.no_grad()
def collect(dataset, indices, payload, features, direct_q, dynamics, device, batch_size):
    state_std = features.state_std
    reward_scale = payload["reward_scale"].to(device).clamp_min(1)
    uncertainties, errors, residuals, episodes = [], [], [], []
    for start in range(0, len(indices), batch_size):
        batch = dataset.batch(indices[start:start + batch_size], device)
        current = batch["state"]
        physical = []
        for model in dynamics:
            predicted, contact, reward, terminal = dynamics_output(
                model, features, batch["history_state"], batch["history_action"],
                batch["history_mask"], batch["context"], batch["action"])
            predicted = predicted.clone()
            predicted[:, 49:51] = contact.sigmoid()
            predicted[:, 51:55], _, _ = advance_events(
                current[:, 51:55], predicted[:, 38],
                predicted[:, 49:51].bool().all(-1), batch["context"][:, 6])
            physical.append(torch.cat((
                (predicted[:, 36:39] - current[:, 36:39]) / state_std[36:39].clamp_min(.01),
                predicted[:, 43:49] / state_std[43:49].clamp_min(.01),
                predicted[:, 49:51], reward[:, None] / reward_scale,
                terminal.sigmoid()[:, None],
            ), dim=-1))
        physical = torch.stack(physical)
        actual = torch.cat((
            (batch["next_state"][:, 36:39] - current[:, 36:39]) / state_std[36:39].clamp_min(.01),
            batch["next_state"][:, 43:49] / state_std[43:49].clamp_min(.01),
            batch["next_state"][:, 49:51], batch["reward"][:, None] / reward_scale,
            batch["done"].float()[:, None],
        ), dim=-1)
        mean = physical.mean(0)
        uncertainties.append(physical.std(0, unbiased=False).square().mean(-1).sqrt().cpu())
        errors.append((actual - mean).square().mean(-1).sqrt().cpu())
        history = features.history(batch["history_state"], batch["history_action"], batch["history_mask"])
        q = direct_q(history, features.context(batch["context"]), batch["action"]).squeeze(-1)
        residuals.append((batch["return"] - q).abs().cpu())
        episodes.append(batch["episode_id"].cpu())
    return {"uncertainty": torch.cat(uncertainties), "error": torch.cat(errors),
            "residual": torch.cat(residuals), "episode": torch.cat(episodes)}


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
    data = collect(dataset, dataset.hold, payload, features, direct_q, dynamics, device, args.batch_size)
    u, e, r, episode = data["uncertainty"], data["error"], data["residual"], data["episode"]
    q50, q80 = torch.quantile(u, torch.tensor([.50, .80]))
    low = u <= q50
    high = u >= q80
    error_ratio = float(e[high].mean() / e[low].mean().clamp_min(1e-8))
    residual_mass = float(r[high].sum() / r.sum().clamp_min(1e-8))
    high_episodes = int(torch.unique(episode[high]).numel())
    correlation_error = float(spearmanr(u.numpy(), e.numpy()).statistic)
    correlation_residual = float(spearmanr(u.numpy(), r.numpy()).statistic)
    gates = {
        "top20_error_ratio_at_least_1_25": error_ratio >= 1.25,
        "top20_residual_mass_at_least_0_30": residual_mass >= .30,
        "top20_spans_at_least_100_episodes": high_episodes >= 100,
    }
    result = {
        "schema": "ref2dex.cm_uncertainty_acquisition_probe.v1",
        "run_status": "COMPLETED",
        "experiment_id": "P-20261003-cm-uncertainty-acquisition",
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "rows": len(u), "held_episodes": int(torch.unique(episode).numel()),
        "contract": {"uncertainty_outputs": "object translation/velocity/contact/reward/terminal",
                     "orientation_included": False, "action_changed": False,
                     "cm_refit": False, "success_predictor": False},
        "quantiles": {"u50": float(q50), "u80": float(q80)},
        "top20": {"error_ratio_vs_bottom50": error_ratio,
                  "absolute_q_residual_mass": residual_mass,
                  "episode_count": high_episodes},
        "spearman": {"uncertainty_vs_physical_error": correlation_error,
                     "uncertainty_vs_absolute_q_residual": correlation_residual},
        "gates": gates,
        "probe_label": "PROMISING" if all(gates.values()) else "UNPROMISING",
        "decision": "Authorize one fixed high-uncertainty acquisition/retraining Probe" if all(gates.values()) else "Close uncertainty-guided data responsibility; no collection or policy follow-up",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
