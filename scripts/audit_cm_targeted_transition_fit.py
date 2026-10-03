#!/usr/bin/env python3
"""Fit frozen Cm output heads on native uncertainty-targeted transitions."""
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
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cm_uncertainty_retraining import (
    load_models, metrics, physical_screen, sha, value_metrics,
)
from src.task.CmResidual.physical_value_data import Episodes
from src.task.CmResidual.physical_value_models import dynamics_loss, dynamics_output
from src.task.CmResidual.physical_value_contract import advance_events, quaternion_loss


def native_rows(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cm_uncertainty_acquisition_native.v1":
        raise ValueError("native acquisition schema mismatch")
    selected = payload["treated"].bool()
    if int(selected.sum()) < 32:
        raise ValueError("targeted transition support below predeclared minimum")
    state = payload["state"][selected].to(device)
    next_state = payload["future_state"][selected, 0].to(device)
    context = payload["context"][selected].to(device)
    action = payload["actual_action"][selected].to(device)
    history_state = payload["history_state"][selected].to(device)
    history_action = payload["history_action"][selected].to(device)
    history_mask = payload["history_mask"][selected].to(device)
    contact = payload["future_contact"][selected, 0].to(device).float()
    done = payload["future_done"][selected, 0].to(device).bool()
    # The native collector records the physical state contract but not the
    # training-time shaping components.  The model's event reward is the only
    # reward target derivable without inventing a task reward label.
    _, event_reward, _ = advance_events(
        state[:, 51:55], next_state[:, 38], contact.bool().all(-1), context[:, 6])
    return payload, dict(state=state, next_state=next_state, context=context,
                         action=action, history_state=history_state,
                         history_action=history_action, history_mask=history_mask,
                         contact=contact, done=done, reward=event_reward)


def physical_loss(predicted, contact_logits, reward, terminal, target, actual_reward,
                  done, features, reward_scale):
    groups = []
    for part in (slice(0, 18), slice(18, 36), slice(36, 39), slice(43, 49)):
        groups.append(((predicted[:, part] - target[:, part]) / features.state_std[part]).square().mean(-1))
    groups.append(quaternion_loss(predicted[:, 39:43], target[:, 39:43]))
    groups.append(torch.nn.functional.binary_cross_entropy_with_logits(
        contact_logits, target[:, 49:51], reduction="none").mean(-1))
    groups.append(((reward - actual_reward) / reward_scale.clamp_min(1)).square())
    groups.append(torch.nn.functional.binary_cross_entropy_with_logits(
        terminal, done.float(), reduction="none"))
    return torch.stack(groups).mean()


def fit_models(rows, payload, features, dynamics, updates, batch_size, seed):
    # Restrict adaptation to the final output projection.  This preserves the
    # learned representation while making the responsibility change explicit.
    trainable = []
    for model in dynamics:
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.head[-1].weight.requires_grad_(True)
        model.head[-1].bias.requires_grad_(True)
        trainable.extend([model.head[-1].weight, model.head[-1].bias])
    optimizer = torch.optim.Adam(trainable, lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reward_scale = payload["reward_scale"].to(rows["state"].device)
    losses = []
    for _ in range(updates):
        index = torch.randint(len(rows["state"]), (min(batch_size, len(rows["state"])),), generator=generator)
        batch = {key: value[index] for key, value in rows.items()}
        total = 0.0
        for model in dynamics:
            predicted, contact, reward, terminal = dynamics_output(
                model, features, batch["history_state"], batch["history_action"],
                batch["history_mask"], batch["context"], batch["action"])
            total = total + physical_loss(
                predicted, contact, reward, terminal, batch["next_state"],
                batch["reward"], batch["done"], features, reward_scale) / len(dynamics)
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite targeted transition loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5)
        optimizer.step()
        losses.append(float(total.detach()))
    for model in dynamics:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-records", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    native_payload, rows = native_rows(args.native_records.resolve(), device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    original_dynamics = copy.deepcopy(dynamics)
    losses = fit_models(rows, payload, features, dynamics, args.updates, args.batch_size, 2026100331)
    held_indices = dataset.hold
    original = physical_screen(dataset, held_indices, payload, features, direct_q,
                               original_dynamics, device, args.batch_size)
    adapted = physical_screen(dataset, held_indices, payload, features, direct_q,
                              dynamics, device, args.batch_size)
    uncertainty = original["std"].square().mean(-1).sqrt()
    high = uncertainty >= torch.quantile(uncertainty, .8)
    physical_original, physical_adapted = metrics(original, high), metrics(adapted, high)
    target_original = value_metrics(original["value"], dataset.data["return"][held_indices], original["episode"])
    target_adapted = value_metrics(adapted["value"], dataset.data["return"][held_indices], adapted["episode"])
    top_delta = (physical_adapted["selected_rmse"] - physical_original["selected_rmse"]) / max(physical_original["selected_rmse"], 1e-8)
    overall_delta = (physical_adapted["physical_rmse"] - physical_original["physical_rmse"]) / max(physical_original["physical_rmse"], 1e-8)
    value_delta = {key: target_adapted[key] - target_original[key] for key in target_original}
    gates = {
        "top_uncertainty_physical_rmse_improves_10pct": top_delta <= -.10,
        "overall_physical_rmse_not_worse_2pct": overall_delta <= .02,
        "value_target_rmse_improves_0_5": value_delta["rmse"] <= -.5,
        "value_episode_spearman_loss_within_0_01": value_delta["episode_spearman"] >= -.01,
    }
    result = {
        "schema": "ref2dex.cm_targeted_transition_fit_probe.v1",
        "run_status": "COMPLETED",
        "experiment_id": "P-20261003-cm-targeted-transition-fit",
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "native_records": str(args.native_records.resolve()),
        "native_records_sha256": sha(args.native_records.resolve()),
        "targeted_fit_rows": len(rows["state"]),
        "held_rows": len(held_indices),
        "held_episodes": int(torch.unique(original["episode"]).numel()),
        "contract": {"updates": args.updates, "batch_size": args.batch_size,
                     "adapted_parameters": "final output projection only",
                     "orientation_included": False, "actor_changed": False,
                     "direct_q_changed": False, "ordinary_data_expanded": False,
                     "native_reward_target": "event reward derivable from state/contact; task shaping frozen"},
        "training_loss": {"first": losses[0], "last": losses[-1], "min": min(losses)},
        "held_physical": {"original": physical_original, "adapted": physical_adapted},
        "held_value_target": {"original": target_original, "adapted": target_adapted},
        "delta": {"top_physical_rmse_relative": top_delta,
                   "overall_physical_rmse_relative": overall_delta, "value": value_delta},
        "gates": gates,
        "probe_label": "PROMISING" if all(gates.values()) else "UNPROMISING",
        "decision": "Authorize one matched policy Probe with targeted-transition Cm bundle" if all(gates.values()) else "Close targeted transition fit; no policy follow-up",
    }
    bundle = copy.deepcopy(payload)
    bundle["dynamics"] = [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in dynamics]
    bundle["source_checkpoint_sha256"] = result["checkpoint_sha256"]
    bundle["adaptation"] = {"selection": "native uncertainty-targeted first-step transitions",
                             "updates": args.updates, "targeted_rows": len(rows["state"]),
                             "adapted_parameters": "final output projection only",
                             "probe": result["experiment_id"]}
    torch.save(bundle, args.output.parent / "cm_targeted_transition_adapted_physical_value.pt")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
