#!/usr/bin/env python3
"""Audit a frozen CmLite on first-episode transitions of a scratch PPO actor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.tools.train_cmlite import metrics, prepare


def first_episode_mask(done: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Drop reset-contaminated terminal rows and repeated later episodes."""
    if done.ndim == 2 and done.shape[1] == 1:
        done = done[:, 0]
    if done.ndim != 1 or num_envs < 1 or done.numel() % num_envs:
        raise ValueError("done must contain complete step-major environment batches")
    ended = done.bool().reshape(-1, num_envs)
    if not ended.any(dim=0).all():
        raise ValueError("each environment must have a completed first episode")
    prior_done = ended.long().cumsum(dim=0) - ended.long()
    return ((prior_done == 0) & ~ended).reshape(-1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError("output must be new and batch size positive")
    payload = torch.load(args.transition, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError("transition schema mismatch")
    mask = first_episode_mask(payload["done"], args.num_envs)
    data = {key: value[mask] for key, value in payload.items()
            if isinstance(value, torch.Tensor)}
    frozen = FrozenCmLite(str(args.checkpoint), "cpu", args.checkpoint_sha256)
    prepared = prepare(data, frozen.feature_mode)
    stats = {key: getattr(frozen, key) for key in
             ("feature_mean", "feature_std", "target_mean", "target_std")}
    true_metrics = metrics(frozen.model, prepared, stats, torch.device("cpu"), args.batch_size)
    generator = torch.Generator().manual_seed(129)
    shuffled = dict(data)
    shuffled["action"] = data["action"][torch.randperm(mask.sum().item(), generator=generator)]
    shuffled_metrics = metrics(frozen.model, prepare(shuffled, frozen.feature_mode),
                               stats, torch.device("cpu"), args.batch_size)
    delta_change_sum = 0.0
    for start in range(0, data["q"].shape[0], args.batch_size):
        stop = start + args.batch_size
        q, state = data["q"][start:stop], data["object_state"][start:stop]
        real = frozen.predict(q, data["action"][start:stop], state)["delta_world"]
        permuted = frozen.predict(q, shuffled["action"][start:stop], state)["delta_world"]
        delta_change_sum += (real - permuted).norm(dim=-1).sum().item()
    result = {
        "schema": "ref2dex.cmlite_on_policy_audit.v1",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "feature_mode": frozen.feature_mode,
        "transition": str(args.transition.resolve()),
        "transition_sha256": sha256(args.transition),
        "num_envs": args.num_envs,
        "first_episode_nonterminal_samples": int(mask.sum()),
        "true_action": true_metrics,
        "shuffled_action": shuffled_metrics,
        "mean_predicted_delta_change_on_shuffle_mm":
            delta_change_sum / data["q"].shape[0] * 1000,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
