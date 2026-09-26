#!/usr/bin/env python3
"""Diagnose whether a frozen Cm distinguishes upward from downward effects.

Observed transitions are on-policy, not counterfactual action evaluations. This
probe only routes the next experiment; it does not establish policy utility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask, sha256
from src.task.CmResidual.tools.probe_cm_weight_components import auroc, pearson


@torch.inference_mode()
def audit(path: Path, model: FrozenCmLite, batch_size: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError(f"unexpected transition schema: {path}")
    mask = first_episode_mask(payload["done"], 64)
    q = payload["q"][mask].float()
    action = payload["action"][mask].float()
    state = payload["object_state"][mask].float()
    next_state = payload["next_object_state"][mask].float()
    contact = (payload["hand_contact"][mask].bool() &
               payload["object_contact"][mask].bool()).flatten()
    realized_dz = next_state[:, 2] - state[:, 2]
    predicted_dz, probabilities = [], []
    for start in range(0, q.shape[0], batch_size):
        stop = start + batch_size
        pred = model.predict(q[start:stop], action[start:stop], state[start:stop])
        predicted_dz.append(pred["delta_world"][:, 2].cpu())
        probabilities.append(pred["contact_probability"].cpu())
    predicted_dz = torch.cat(predicted_dz)
    probabilities = torch.cat(probabilities)
    up = contact & (realized_dz >= .003)
    down = contact & (realized_dz <= -.003)
    consequential = up | down
    scores = {
        "abs_effect": (predicted_dz.abs() / .003).clamp(0, 1),
        "signed_up": (predicted_dz / .003).clamp(0, 1),
        "contact_x_signed_up": probabilities * (predicted_dz / .003).clamp(0, 1),
        "signed_down": (-predicted_dz / .003).clamp(0, 1),
    }
    summaries = {}
    for name, score in scores.items():
        count = max(1, score.numel() // 10)
        top = torch.topk(score, count).indices
        summaries[name] = {
            "up_auroc": auroc(score, up),
            "down_auroc": auroc(score, down),
            "up_auroc_given_contact": auroc(score[contact], up[contact]),
            "down_auroc_given_contact": auroc(score[contact], down[contact]),
            "top_decile_up_rate": float(up[top].float().mean()),
            "top_decile_down_rate": float(down[top].float().mean()),
            "top_decile_mean_realized_dz_mm": float(realized_dz[top].mean() * 1000),
        }
    return {
        "transition": str(path.resolve()), "transition_sha256": sha256(path),
        "samples": int(mask.sum()), "contact_rate": float(contact.float().mean()),
        "up_rate": float(up.float().mean()), "down_rate": float(down.float().mean()),
        "predicted_up_fraction": float((predicted_dz > 0).float().mean()),
        "predicted_up_fraction_given_consequential":
            float((predicted_dz[consequential] > 0).float().mean()),
        "signed_dz_pearson_given_contact": pearson(predicted_dz[contact], realized_dz[contact]),
        "scores": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("output must be new and batch size positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = FrozenCmLite(str(args.checkpoint), "cpu", args.checkpoint_sha256)
    result = {"schema": "ref2dex.cm_signed_effect_probe.v1",
              "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": args.checkpoint_sha256,
              "event_threshold_m": .003,
              "audits": [audit(path, model, args.batch_size) for path in args.transition]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
