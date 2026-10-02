#!/usr/bin/env python3
"""Offline critic-residual Probe using frozen Cm physical consequences.

The frozen direct-Q remains the task-value baseline.  Cm contributes only
predicted one-step object/contact consequences and ensemble uncertainty to a
small residual value head; the target is the realized complete return, never a
success label or a policy action.
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
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task.CmResidual.physical_value_contract import advance_events
from src.task.CmResidual.physical_value_data import Episodes
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, dynamics_output


def load_models(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("physical value model schema mismatch")
    stats = {key: value.to(device) for key, value in payload["stats"].items()}
    features = Features(**stats, device=device).to(device).eval()
    direct_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    direct_q.load_state_dict(payload["direct_q"])
    direct_q.eval()
    dynamics = []
    for index, state in enumerate(payload["dynamics"]):
        model = OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + index).to(device)
        model.load_state_dict(state)
        model.eval()
        dynamics.append(model)
    return payload, features, direct_q, dynamics


@torch.no_grad()
def extract(dataset, indices, payload, features, direct_q, dynamics, device, batch_size):
    outputs = {"base": [], "cm": [], "direct": [], "target": [], "episode": []}
    for start in range(0, len(indices), batch_size):
        current = dataset.batch(indices[start:start + batch_size], device)
        history = features.history(current["history_state"], current["history_action"],
                                    current["history_mask"])
        direct = direct_q(history, features.context(current["context"]),
                           current["action"]).squeeze(-1)
        state = current["state"]
        physical = []
        for model in dynamics:
            out, contact, reward, terminal = dynamics_output(
                model, features, current["history_state"], current["history_action"],
                current["history_mask"], current["context"], current["action"])
            out = out.clone()
            out[:, 49:51] = contact.sigmoid()
            out[:, 51:55], _, _ = advance_events(
                state[:, 51:55], out[:, 38], (out[:, 49:51] > .5).all(-1), current["context"][:, 6])
            physical.append(torch.cat((
                out[:, 36:39] - state[:, 36:39],
                out[:, 43:49],
                out[:, 49:51],
                reward[:, None],
                terminal.sigmoid()[:, None],
            ), dim=-1))
        physical = torch.stack(physical)
        mean, std = physical.mean(0), physical.std(0, unbiased=False)
        # The base residual head can recalibrate direct-Q from current action/state;
        # the Cm head receives only frozen predicted physical consequences in excess.
        base = torch.cat((direct[:, None], current["action"], state[:, 36:51]), dim=-1)
        cm = torch.cat((base, mean, std), dim=-1)
        outputs["base"].append(base.cpu())
        outputs["cm"].append(cm.cpu())
        outputs["direct"].append(direct.cpu())
        outputs["target"].append(current["return"].cpu())
        outputs["episode"].append(current["episode_id"].cpu())
    return {key: torch.cat(value) for key, value in outputs.items()}


def fit_predict(x_fit, y_fit, x_eval):
    mean = x_fit.mean(0)
    scale = x_fit.std(0)
    scale[scale < 1e-8] = 1.0
    model = Ridge(alpha=1.0).fit((x_fit - mean) / scale, y_fit)
    return model.predict((x_eval - mean) / scale)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-fit-rows", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    fit_indices = dataset.fit
    if args.max_fit_rows and len(fit_indices) > args.max_fit_rows:
        generator = torch.Generator(device="cpu").manual_seed(20261003)
        fit_indices = fit_indices[torch.randperm(len(fit_indices), generator=generator)[:args.max_fit_rows]]
    fit_values = extract(dataset, fit_indices, payload, features, direct_q, dynamics,
                         device, args.batch_size)
    held_values = extract(dataset, dataset.hold, payload, features, direct_q, dynamics,
                          device, args.batch_size)
    fit_target = fit_values["target"].numpy()
    held_target = held_values["target"].numpy()
    held_episode = held_values["episode"].numpy()
    direct = held_values["direct"].numpy()
    reports = {"direct": metrics(direct, held_target, held_episode)}
    predictions = {}
    for name in ("base", "cm"):
        prediction = direct + fit_predict(
            fit_values[name].numpy(), fit_target - fit_values["direct"].numpy(),
            held_values[name].numpy())
        predictions[name] = prediction
        reports["direct_plus_" + name + "_residual"] = metrics(prediction, held_target, held_episode)
    result = {
        "schema": "ref2dex.cm_critic_residual_audit.v1",
        "run_status": "COMPLETED",
        "rows": {"fit": len(fit_values["target"]), "held": len(held_target)},
        "fit_episodes": int(torch.unique(fit_values["episode"]).numel()),
        "held_episodes": int(torch.unique(held_values["episode"]).numel()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        "models": reports,
        "delta_vs_direct": {
            name: {key: value - reports["direct"][key] for key, value in reports[name].items()}
            for name in ("direct_plus_base_residual", "direct_plus_cm_residual")
        },
        "cm_feature_contract": "frozen ensemble predicted object delta/velocity/contact/reward/terminal plus current state/action; no future state",
        "target": "realized complete-return from held transition; no success label",
        "interpretation": "Offline critic-residual sufficiency Probe; no actor, policy, or success claim",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
