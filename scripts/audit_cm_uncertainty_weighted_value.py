#!/usr/bin/env python3
"""Test Cm disagreement as a direct-Q loss-allocation signal.

Cm supplies only a frozen uncertainty score.  Both Q copies learn the same
real complete-return target from the same batches; the treatment doubles the
loss weight on fit top-20% disagreement rows.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_cm_model_based_critic import load_models, metrics, sha, synthetic_target, predict_q
from src.task.CmResidual.physical_value_data import Episodes
from src.task.CmResidual.physical_value_models import OutcomeNetwork


def weighted_metrics(prediction, target, episode, uncertainty, threshold):
    selected = uncertainty >= threshold
    result = metrics(prediction, target, episode)
    if int(selected.sum()) < 2:
        raise ValueError("held top-uncertainty support is too small")
    top = metrics(prediction[selected], target[selected], episode[selected])
    bottom = metrics(prediction[~selected], target[~selected], episode[~selected])
    return result, top, bottom, int(selected.sum())


def train_copy(dataset, fit_indices, fit_values, state, payload, features, device,
               updates, batch_size, generator, weights):
    network = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    network.load_state_dict(state, strict=True)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    scale = payload["return_scale"].to(device).clamp_min(1)
    positions = torch.randint(len(fit_indices), (updates, batch_size), generator=generator)
    losses = []
    for row in positions:
        indices = fit_indices[row]
        batch = dataset.batch(indices, device)
        prediction = network(
            features.history(batch["history_state"], batch["history_action"], batch["history_mask"]),
            features.context(batch["context"]), batch["action"]).squeeze(-1)
        target = batch["return"]
        row_weights = weights[row].to(device)
        loss = (((prediction - target) / scale).square() * row_weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite weighted-Q loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()
        losses.append(float(loss.detach()))
    network.eval()
    return network, {"first": losses[0], "last": losses[-1], "mean": float(np.mean(losses))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--max-fit-rows", type=int, default=120000)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    generator = torch.Generator(device="cpu").manual_seed(2026100312)
    fit_indices = dataset.fit
    if args.max_fit_rows and len(fit_indices) > args.max_fit_rows:
        fit_indices = fit_indices[torch.randperm(len(fit_indices), generator=generator)[:args.max_fit_rows]]
    held_indices = dataset.hold
    fit_values = synthetic_target(dataset, fit_indices, payload, features, direct_q, dynamics,
                                  device, args.batch_size, uncertainty_gate=1e9)
    held_values = synthetic_target(dataset, held_indices, payload, features, direct_q, dynamics,
                                   device, args.batch_size, uncertainty_gate=1e9)
    fit_threshold = float(torch.quantile(fit_values["uncertainty"], .8))
    fit_top = fit_values["uncertainty"] >= fit_threshold
    held_top = held_values["uncertainty"] >= fit_threshold
    fit_weights = torch.where(fit_top, torch.full_like(fit_values["uncertainty"], 2.0),
                              torch.ones_like(fit_values["uncertainty"]))
    initial_state = {key: value.detach().cpu().clone() for key, value in direct_q.state_dict().items()}
    baseline_network, baseline_loss = train_copy(
        dataset, fit_indices, fit_values, initial_state, payload, features, device,
        args.updates, args.batch_size, torch.Generator(device="cpu").manual_seed(2026100313),
        torch.ones_like(fit_weights))
    weighted_network, weighted_loss = train_copy(
        dataset, fit_indices, fit_values, initial_state, payload, features, device,
        args.updates, args.batch_size, torch.Generator(device="cpu").manual_seed(2026100313),
        fit_weights)
    frozen = predict_q(dataset, held_indices, direct_q, features, device, args.batch_size)
    baseline = predict_q(dataset, held_indices, baseline_network, features, device, args.batch_size)
    weighted = predict_q(dataset, held_indices, weighted_network, features, device, args.batch_size)
    target = held_values["return"]
    frozen_all, frozen_top, frozen_bottom, held_top_count = weighted_metrics(
        frozen, target, held_values["episode"], held_values["uncertainty"], fit_threshold)
    baseline_all, baseline_top, baseline_bottom, _ = weighted_metrics(
        baseline, target, held_values["episode"], held_values["uncertainty"], fit_threshold)
    weighted_all, weighted_top, weighted_bottom, _ = weighted_metrics(
        weighted, target, held_values["episode"], held_values["uncertainty"], fit_threshold)
    top_improvement = (baseline_top["rmse"] - weighted_top["rmse"]) / max(baseline_top["rmse"], 1e-8)
    overall_rmse_change = (weighted_all["rmse"] - baseline_all["rmse"]) / max(baseline_all["rmse"], 1e-8)
    overall_mae_change = (weighted_all["mae"] - baseline_all["mae"]) / max(baseline_all["mae"], 1e-8)
    episode_drop = baseline_all["episode_spearman"] - weighted_all["episode_spearman"]
    result = {
        "schema": "ref2dex.cm_uncertainty_weighted_value.v1",
        "run_status": "COMPLETED",
        "experiment_id": "P-20261003-cm-uncertainty-weighted-value",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "collections": [str(path.resolve()) for path in args.collections],
        "fit_rows": int(len(fit_indices)), "held_rows": int(len(held_indices)),
        "fit_threshold_p80": fit_threshold,
        "fit_top_rows": int(fit_top.sum()), "held_top_rows": held_top_count,
        "contract": {"target": "real complete return", "uncertainty": "ensemble physical translation/velocity/contact/reward/terminal disagreement",
                     "orientation_excluded": True, "treatment_top_weight": 2.0,
                     "actor_input_changed": False, "actor_action_changed": False,
                     "synthetic_target_used": False, "same_batches": True},
        "training": {"updates": args.updates, "batch_size": args.batch_size,
                     "control_loss": baseline_loss, "treatment_loss": weighted_loss},
        "held": {"frozen_direct_q": frozen_all, "unweighted_refit": baseline_all,
                 "weighted_refit": weighted_all, "frozen_top20": frozen_top,
                 "unweighted_top20": baseline_top, "weighted_top20": weighted_top,
                 "unweighted_bottom80": baseline_bottom, "weighted_bottom80": weighted_bottom},
        "deltas": {"top20_rmse_improvement_vs_unweighted": top_improvement,
                   "overall_rmse_change": overall_rmse_change,
                   "overall_mae_change": overall_mae_change,
                   "episode_spearman_drop": episode_drop},
        "gates": {"top20_rmse_ge_5pct": bool(top_improvement >= .05),
                  "overall_rmse_not_worse_1pct": bool(overall_rmse_change <= .01),
                  "overall_mae_not_worse_1pct": bool(overall_mae_change <= .01),
                  "episode_spearman_not_worse_.01": bool(episode_drop <= .01)},
        "probe_label": "PROMISING" if (top_improvement >= .05 and overall_rmse_change <= .01 and
                                        overall_mae_change <= .01 and episode_drop <= .01) else "UNPROMISING",
        "interpretation": "Cm uncertainty allocates real-return direct-Q fitting only; no actor or policy utility claim",
    }
    weighted_path = args.output.parent / "uncertainty_weighted_direct_q.pt"
    baseline_path = args.output.parent / "unweighted_refit_direct_q.pt"
    torch.save({"schema": "ref2dex.cm_uncertainty_weighted_q.v1", "state_dict": weighted_network.state_dict(),
                "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": sha(args.checkpoint.resolve()),
                "fit_threshold_p80": fit_threshold, "top_weight": 2.0}, weighted_path)
    torch.save({"schema": "ref2dex.cm_unweighted_refit_q.v1", "state_dict": baseline_network.state_dict(),
                "source_checkpoint": str(args.checkpoint.resolve()), "source_checkpoint_sha256": sha(args.checkpoint.resolve())}, baseline_path)
    result["artifacts"] = {"weighted": str(weighted_path.resolve()), "weighted_sha256": sha(weighted_path),
                           "unweighted": str(baseline_path.resolve()), "unweighted_sha256": sha(baseline_path)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
