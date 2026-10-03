#!/usr/bin/env python3
"""Audit the offline Gate 1 interaction augmentation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROW_KEYS = (
    "state", "previous_action", "context", "history_state",
    "history_previous_action", "history_context", "history_progress", "action",
    "reward", "reward_components", "return_to_go", "effect", "future_action",
    "future_valid_mask", "episode_id", "step", "motion_id", "noise_std",
    "source_run", "done_at_decision", "episode_auxiliary",
)
BASE_DIM = 80
AUGMENTED_DIM = 119
AUGMENTED_DELTA_DIM = AUGMENTED_DIM - BASE_DIM


def audit(base: dict, augmented: dict) -> list[str]:
    errors: list[str] = []
    base_i = base.get("interaction")
    aug_i = augmented.get("interaction")
    if not isinstance(base_i, torch.Tensor) or base_i.ndim != 3:
        errors.append("base interaction is not [N,H,D]")
    if not isinstance(aug_i, torch.Tensor) or aug_i.ndim != 3:
        errors.append("augmented interaction is not [N,H,D]")
    if errors:
        return errors
    if tuple(base_i.shape[:2]) != tuple(aug_i.shape[:2]):
        errors.append("base/augmented row or horizon shape differs")
    if base_i.shape[-1] != BASE_DIM:
        errors.append(f"base interaction dim is {base_i.shape[-1]}, expected {BASE_DIM}")
    if aug_i.shape[-1] != AUGMENTED_DIM:
        errors.append(f"augmented interaction dim is {aug_i.shape[-1]}, expected {AUGMENTED_DIM}")
    if aug_i.shape[-1] - base_i.shape[-1] != AUGMENTED_DELTA_DIM:
        errors.append("unexpected appended interaction width")
    if not torch.equal(base_i, aug_i[..., :BASE_DIM]):
        errors.append("base interaction was changed")
    if aug_i.shape[1] and not bool(torch.equal(
            aug_i[:, 0, BASE_DIM:], torch.zeros_like(aug_i[:, 0, BASE_DIM:]))):
        errors.append("first future slot appended deltas are not zero")
    if not torch.isfinite(aug_i).all():
        errors.append("augmented interaction contains non-finite values")
    for key in ROW_KEYS:
        if key not in base or key not in augmented:
            errors.append(f"missing row key: {key}")
        elif not torch.equal(base[key], augmented[key]):
            errors.append(f"row key changed: {key}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--augmented", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    augmented = torch.load(args.augmented, map_location="cpu", weights_only=False)
    errors = audit(base, augmented)
    result = {
        "schema": "ref2dex.gate1_interaction_augmentation_audit.v1",
        "run_status": "COMPLETED",
        "base": str(args.base.resolve()),
        "augmented": str(args.augmented.resolve()),
        "rows": int(augmented["interaction"].shape[0]),
        "horizon": int(augmented["interaction"].shape[1]),
        "base_interaction_dim": BASE_DIM,
        "augmented_interaction_dim": AUGMENTED_DIM,
        "errors": errors,
        "augmentation_contract_ready": not errors,
        "no_model_fit": True,
        "no_physics_replay": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
