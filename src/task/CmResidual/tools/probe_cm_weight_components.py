#!/usr/bin/env python3
"""Cheap offline probe of the two frozen CmLite PPO-weight components.

This is diagnostic only: association with an observed one-step effect does not
establish that a component improves PPO or counterfactual action choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auroc(score: torch.Tensor, label: torch.Tensor) -> float:
    """Rank AUC, with tied scores assigned their average rank."""
    score, label = score.double().flatten(), label.bool().flatten()
    positives, negatives = int(label.sum()), int((~label).sum())
    if not positives or not negatives:
        return float("nan")
    ordered = torch.argsort(score, stable=True)
    sorted_scores = score[ordered]
    ranks = torch.arange(1, score.numel() + 1, dtype=torch.float64)
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    if bool((counts > 1).any()):
        first = torch.cumsum(counts, 0) - counts
        average = (2 * first + counts + 1).double() / 2
        ranks = torch.repeat_interleave(average, counts)
    positive_rank_sum = ranks[label[ordered]].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) /
                 (positives * negatives))


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double().flatten(), y.double().flatten()
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float(x.dot(y) / denominator) if denominator > 0 else float("nan")


def component_summary(score: torch.Tensor, event: torch.Tensor,
                      true_effect: torch.Tensor) -> dict[str, float]:
    count = max(1, score.numel() // 10)
    top = torch.topk(score, count).indices
    bottom = torch.topk(score, count, largest=False).indices
    return {
        "event_auroc": auroc(score, event),
        "effect_pearson": pearson(score, true_effect),
        "top_decile_event_rate": float(event[top].float().mean()),
        "bottom_decile_event_rate": float(event[bottom].float().mean()),
        "top_decile_mean_abs_dz_mm": float(true_effect[top].mean() * 1000),
        "bottom_decile_mean_abs_dz_mm": float(true_effect[bottom].mean() * 1000),
    }


@torch.inference_mode()
def audit(path: Path, model: FrozenCmLite, num_envs: int,
          batch_size: int) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError(f"transition schema mismatch: {path}")
    mask = first_episode_mask(payload["done"], num_envs)
    q = payload["q"][mask].float()
    action = payload["action"][mask].float()
    state = payload["object_state"][mask].float()
    next_state = payload["next_object_state"][mask].float()
    contact = (payload["hand_contact"][mask].bool() &
               payload["object_contact"][mask].bool()).flatten()
    true_effect = (next_state[:, 2] - state[:, 2]).abs()
    event = contact & (true_effect >= 0.003)
    generator = torch.Generator().manual_seed(153)
    shuffled_action = action[torch.randperm(action.shape[0], generator=generator)]
    predictions: dict[str, list[torch.Tensor]] = {
        "p": [], "effect": [], "shuffled_p": [], "shuffled_effect": []}
    for start in range(0, q.shape[0], batch_size):
        end = start + batch_size
        actual = model.predict(q[start:end], action[start:end], state[start:end])
        shuffled = model.predict(q[start:end], shuffled_action[start:end], state[start:end])
        predictions["p"].append(actual["contact_probability"].cpu())
        predictions["effect"].append(
            (actual["delta_world"][:, 2].abs() / 0.003).clamp(0, 1).cpu())
        predictions["shuffled_p"].append(shuffled["contact_probability"].cpu())
        predictions["shuffled_effect"].append(
            (shuffled["delta_world"][:, 2].abs() / 0.003).clamp(0, 1).cpu())
    values = {key: torch.cat(chunks) for key, chunks in predictions.items()}
    p, effect = values["p"], values["effect"]
    shuffled_p, shuffled_effect = values["shuffled_p"], values["shuffled_effect"]
    result = {
        "transition": str(path.resolve()),
        "transition_sha256": sha256(path),
        "samples": int(q.shape[0]),
        "contact_rate": float(contact.float().mean()),
        "consequential_event_rate": float(event.float().mean()),
        "component_mean": {"contact": float(p.mean()), "effect": float(effect.mean()),
                           "joint": float((p * effect).mean())},
        "component_saturation": {
            "contact_p_gt_0_9": float((p > 0.9).float().mean()),
            "effect_eq_1": float((effect == 1).float().mean())},
        "predictive_association": {
            "contact": component_summary(p, event, true_effect),
            "effect": component_summary(effect, event, true_effect),
            "joint": component_summary(p * effect, event, true_effect),
            "shuffled_joint": component_summary(shuffled_p * shuffled_effect,
                                                event, true_effect)},
        "action_shuffle": {
            "mean_abs_contact_change": float((p - shuffled_p).abs().mean()),
            "mean_abs_effect_change": float((effect - shuffled_effect).abs().mean()),
            "mean_abs_joint_change": float((p * effect - shuffled_p * shuffled_effect).abs().mean()),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.num_envs < 1 or args.batch_size < 1:
        raise ValueError("output must be new and sizes positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = FrozenCmLite(str(args.checkpoint), "cpu", args.checkpoint_sha256)
    result = {
        "schema": "ref2dex.cm_weight_component_probe.v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "num_envs": args.num_envs,
        "event_definition": "contact and abs(realized_world_dz)>=3mm",
        "audits": [audit(path, model, args.num_envs, args.batch_size)
                   for path in args.transition],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
