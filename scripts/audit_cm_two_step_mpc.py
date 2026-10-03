#!/usr/bin/env python3
"""Held-only two-step Cm model-predictive planning Probe."""
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

from src.task.CmResidual.physical_value_contract import advance_events, make_candidates
from src.task.CmResidual.physical_value_data import Episodes
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, decode_dynamics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("physical value checkpoint schema mismatch")
    features = Features(**{key: value.to(device) for key, value in payload["stats"].items()},
                        device=device).to(device).eval()
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


def project(out, contact, terminal, state, context, features):
    out = out.clone()
    out[:, 49:51] = contact.sigmoid()
    out[:, 51:55], _, _ = advance_events(
        state[:, 51:55], out[:, 38], (out[:, 49:51] > .5).all(-1), context[:, 6])
    # Cm owns object translation/velocity and contact/event consequences.  The
    # hand pose and object orientation remain the observed state.
    out[:, :36] = state[:, :36]
    out[:, 39:43] = state[:, 39:43]
    return out


def shifted_history(history_state, history_action, history_mask, next_state, action):
    return (
        torch.cat((history_state[:, 1:], next_state[:, None]), 1),
        torch.cat((history_action[:, 1:], action[:, None]), 1),
        torch.cat((history_mask[:, 1:], torch.ones_like(history_mask[:, :1])), 1),
    )


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
def evaluate(dataset, indices, payload, features, direct_q, dynamics, device, batch_size):
    outputs = {key: [] for key in ("target", "direct", "mve_one", "mpc_two", "episode")}
    changed = []
    total = 0
    gamma = float(payload["gamma"])
    for start in range(0, len(indices), batch_size):
        current = dataset.batch(indices[start:start + batch_size], device)
        b = len(current["state"])
        state = current["state"]
        first, valid_first = make_candidates(current["action"])
        count = first.shape[1]
        history = current["history_state"]
        actions = current["history_action"]
        masks = current["history_mask"]
        ctx = current["context"]
        next_ctx = current["next_context"]

        def repeat_rows(value, repeats):
            return value[:, None].expand(-1, repeats, *value.shape[1:]).reshape(
                b * repeats, *value.shape[1:])

        h1 = repeat_rows(history, count)
        a1 = repeat_rows(actions, count)
        m1 = repeat_rows(masks, count)
        c1 = repeat_rows(ctx, count)
        nc1 = repeat_rows(next_ctx, count)
        first_flat = first.reshape(-1, 18)
        direct_scores = direct_q(features.history(h1, a1, m1), features.context(c1), first_flat).squeeze(-1)
        direct_scores = direct_scores.reshape(b, count).masked_fill(~valid_first, -torch.inf)

        one_models, two_models = [], []
        for model in dynamics:
            # Keep the explicit feature/history contract in one place; the
            # dynamics helper also performs the state normalization and FK.
            raw1 = model(features.history(h1, a1, m1), features.context(c1), first_flat)
            out1, contact1, reward1, terminal1 = decode_dynamics(raw1, repeat_rows(state, count), features)
            pred1 = project(out1, contact1, terminal1, repeat_rows(state, count), c1, features)
            h2, a2hist, m2 = shifted_history(h1, a1, m1, pred1, first_flat)

            # Second action candidates are the same bounded local panel around
            # each imagined first action.  This is a short MPC search, not an
            # actor observation or a success classifier.
            second, valid_second = make_candidates(first_flat)
            second_count = second.shape[1]
            n2 = b * count
            h2r = h2[:, None].expand(-1, second_count, *h2.shape[1:]).reshape(
                n2 * second_count, *h2.shape[1:])
            a2r = a2hist[:, None].expand(-1, second_count, *a2hist.shape[1:]).reshape(
                n2 * second_count, *a2hist.shape[1:])
            m2r = m2[:, None].expand(-1, second_count, *m2.shape[1:]).reshape(
                n2 * second_count, *m2.shape[1:])
            c2r = c1[:, None].expand(-1, second_count, *c1.shape[1:]).reshape(
                n2 * second_count, *c1.shape[1:])
            ncr = nc1[:, None].expand(-1, second_count, *nc1.shape[1:]).reshape(
                n2 * second_count, *nc1.shape[1:])
            second_flat = second.reshape(-1, 18)
            current2 = pred1[:, None].expand(-1, second_count, -1).reshape(-1, pred1.shape[-1])
            raw2 = model(features.history(h2r, a2r, m2r), features.context(c2r), second_flat)
            out2, contact2, reward2, terminal2 = decode_dynamics(raw2, current2, features)
            pred2 = project(out2, contact2, terminal2, current2, c2r, features)
            h3, a3hist, m3 = shifted_history(h2r, a2r, m2r, pred2, second_flat)
            tail = direct_q(features.history(h3, a3hist, m3), features.context(ncr), second_flat).squeeze(-1)
            second_score = reward2 + gamma * (1 - terminal2.sigmoid()) * tail
            second_score = second_score.reshape(b * count, second_count).masked_fill(~valid_second, -torch.inf)
            best_second = second_score.max(-1).values.reshape(b, count)
            reward1_panel = reward1.reshape(b, count)
            terminal1_panel = terminal1.sigmoid().reshape(b, count)
            one_tail = direct_q(
                features.history(h2, a2hist, m2), features.context(nc1), first_flat).squeeze(-1).reshape(b, count)
            one_score = reward1_panel + gamma * (1 - terminal1_panel) * one_tail
            one_models.append(one_score.masked_fill(~valid_first, -torch.inf))
            two_models.append((reward1_panel + gamma * (1 - terminal1_panel) * best_second).masked_fill(~valid_first, -torch.inf))

        one_stack = torch.stack(one_models)
        two_stack = torch.stack(two_models)
        one_score = one_stack.mean(0) - one_stack.std(0, unbiased=False)
        two_score = two_stack.mean(0) - two_stack.std(0, unbiased=False)
        direct_index = direct_scores.argmax(-1)
        two_index = two_score.argmax(-1)
        finite = torch.isfinite(two_score).all(-1)
        changed.append((two_index != direct_index)[finite].cpu())
        outputs["target"].append(current["return"].cpu())
        outputs["direct"].append(direct_scores.max(-1).values.cpu())
        outputs["mve_one"].append(one_score.max(-1).values.cpu())
        outputs["mpc_two"].append(two_score.max(-1).values.cpu())
        outputs["episode"].append(current["episode_id"].cpu())
        total += int(finite.sum())
        print(json.dumps({"rows_done": min(start + batch_size, len(indices)), "finite": total}), flush=True)
    return {key: torch.cat(value) for key, value in outputs.items()}, torch.cat(changed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    dataset = Episodes(args.collections)
    held = dataset.hold
    generator = torch.Generator(device="cpu").manual_seed(2026100308)
    held = held[torch.randperm(len(held), generator=generator)[:args.max_rows]]
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), torch.device(args.device))
    values, changed = evaluate(dataset, held, payload, features, direct_q, dynamics,
                               torch.device(args.device), args.batch_size)
    target, episode = values["target"], values["episode"]
    models = {name: metrics(values[name], target, episode) for name in ("direct", "mve_one", "mpc_two")}
    change_fraction = float(changed.float().mean()) if len(changed) else 0.0
    result = {
        "schema": "ref2dex.cm_two_step_mpc_probe.v1",
        "run_status": "COMPLETED",
        "rows": len(target),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "models": models,
        "delta_vs_direct": {name: {key: models[name][key] - models["direct"][key]
                                    for key in models["direct"]} for name in ("mve_one", "mpc_two")},
        "first_action_argmax_changed_fraction": change_fraction,
        "contract": "two recursive Cm transitions, observed hand q/dq and object orientation preserved, direct-Q tail",
        "probe_label": "PROMISING" if (
            models["mpc_two"]["rmse"] <= models["direct"]["rmse"] - .5 and
            models["mpc_two"]["episode_spearman"] >= models["direct"]["episode_spearman"] - .01 and
            .05 <= change_fraction <= .60
        ) else "UNPROMISING",
        "interpretation": "Held target-quality and action-change screen only; no native or policy claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
