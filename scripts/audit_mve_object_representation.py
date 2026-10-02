#!/usr/bin/env python3
"""Offline held-transition audit for object-projected Cm MVE targets."""
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

from src.task.CmResidual.physical_value_contract import advance_events
from src.task.CmResidual.physical_value_data import Episodes
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, dynamics_output


def load_models(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("physical value model schema mismatch")
    features = Features(**{key: value.to(device) for key, value in payload["stats"].items()},
                        device=device).to(device).eval()
    value = OutcomeNetwork(payload["context_dim"], 0, 1, 9283).to(device)
    value.load_state_dict(payload["value"])
    value.eval()
    direct_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    direct_q.load_state_dict(payload["direct_q"])
    direct_q.eval()
    dynamics = []
    for index, state in enumerate(payload["dynamics"]):
        model = OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + index).to(device)
        model.load_state_dict(state)
        model.eval()
        dynamics.append(model)
    return payload, features, value, direct_q, dynamics


@torch.no_grad()
def evaluate(collections, checkpoint, device, max_rows):
    dataset = Episodes(collections)
    payload, features, value, direct_q, dynamics = load_models(checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(20261003)
    indices = dataset.hold[torch.randperm(len(dataset.hold), generator=generator)]
    if max_rows > 0:
        indices = indices[:max_rows]
    result = {"target": [], "direct": [], "mve_full": [], "mve_object_current": [], "episode": []}
    batch_size = 512
    for start in range(0, len(indices), batch_size):
        current = dataset.batch(indices[start:start + batch_size], device)
        next_batch, _ = dataset.future(indices[start:start + batch_size], 1, device)
        history = features.history(current["history_state"], current["history_action"],
                                    current["history_mask"])
        direct = direct_q(history, features.context(current["context"]),
                           current["action"]).squeeze(-1)
        current_state = current["state"]
        predictions = {"full": [], "object": []}
        for model in dynamics:
            out, contact, reward, terminal = dynamics_output(
                model, features, current["history_state"], current["history_action"],
                current["history_mask"], current["context"], current["action"])
            out = out.clone()
            out[:, 49:51] = contact.sigmoid()
            out[:, 51:55], _, _ = advance_events(
                current_state[:, 51:55], out[:, 38],
                (out[:, 49:51] > .5).all(-1), current["context"][:, 6])

            def continuation(next_state, next_action):
                next_history_state = torch.cat((current["history_state"][:, 1:], next_state[:, None]), 1)
                next_history_action = torch.cat((current["history_action"][:, 1:], next_action[:, None]), 1)
                next_history_mask = torch.cat((current["history_mask"][:, 1:],
                                               torch.ones_like(current["history_mask"][:, :1])), 1)
                return value(features.history(next_history_state, next_history_action, next_history_mask),
                             features.context(current["next_context"])).squeeze(-1)

            predictions["full"].append(
                reward + payload["gamma"] * (1 - terminal.sigmoid()) * continuation(out, current["action"]))
            projected = out.clone()
            # This is available at the decision instant: retain the current hand
            # configuration while Cm supplies only object consequences/events.
            projected[:, :36] = current_state[:, :36]
            projected[:, 39:43] = current_state[:, 39:43]
            predictions["object"].append(
                reward + payload["gamma"] * (1 - terminal.sigmoid()) * continuation(projected, current["action"]))
        full = torch.stack(predictions["full"])
        obj = torch.stack(predictions["object"])
        result["target"].append(current["return"].cpu())
        result["direct"].append(direct.cpu())
        result["mve_full"].append((full.mean(0) - full.std(0, unbiased=False)).cpu())
        result["mve_object_current"].append((obj.mean(0) - obj.std(0, unbiased=False)).cpu())
        result["episode"].append(current["episode_id"].cpu())
    return payload, {key: torch.cat(value) for key, value in result.items()}


def metrics(prediction, target, episode):
    x, y = prediction.numpy(), target.numpy()
    episode_pred, episode_target = [], []
    for identifier in torch.unique(episode):
        selected = episode == identifier
        episode_pred.append(float(prediction[selected].mean()))
        episode_target.append(float(target[selected].mean()))
    return {
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "mae": float(np.mean(np.abs(x - y))),
        "spearman": float(spearmanr(x, y).statistic),
        "episode_spearman": float(spearmanr(episode_pred, episode_target).statistic),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="0 evaluates the complete deterministic held split")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    payload, values = evaluate(args.collections, checkpoint, torch.device(args.device), args.max_rows)
    target = values["target"]
    result = {
        "schema": "ref2dex.mve_object_representation_audit.v1",
        "run_status": "COMPLETED",
        "rows": len(target),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "collection_results": [str((path / "results.json").resolve()) for path in args.collections],
        "models": {name: metrics(values[name], target, values["episode"])
                   for name in ("direct", "mve_full", "mve_object_current")},
        "delta_vs_direct": {},
        "projection": "current hand q/dq and current object orientation plus Cm-predicted object translation/velocity/contact/events",
        "target": "realized complete-return from the held transition; no future state is an input",
        "interpretation": "Offline target-quality Probe only; this does not establish candidate action utility or policy gain",
    }
    direct = result["models"]["direct"]
    for name in ("mve_full", "mve_object_current"):
        result["delta_vs_direct"][name] = {
            key: result["models"][name][key] - direct[key]
            for key in ("rmse", "mae", "spearman", "episode_spearman")
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
