#!/usr/bin/env python3
"""Validate the assembled Gate 1 window contract without fitting a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    required = {
        "state", "previous_action", "context", "action", "reward", "reward_components",
        "return_to_go", "effect", "interaction", "future_valid_mask", "episode_id", "step",
        "motion_id", "noise_std", "episode_auxiliary", "metadata",
    }
    missing = sorted(required.difference(payload))
    errors = []
    if missing:
        errors.append(f"missing keys: {missing}")
    n = int(payload["return_to_go"].shape[0]) if "return_to_go" in payload else 0
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] != n:
            if key not in {"metadata"}:
                errors.append(f"row misalignment {key}: {tuple(value.shape)} vs {n}")
    metadata = payload.get("metadata", {})
    for key in ("target_definition", "effect_layout", "interaction_layout", "split_unit"):
        if not isinstance(metadata.get(key), str):
            errors.append(f"missing metadata.{key}")
    if n:
        for key in ("state", "context", "action", "return_to_go", "effect", "interaction"):
            if key in payload and not torch.isfinite(payload[key]).all():
                errors.append(f"non-finite values in {key}")
        if "future_valid_mask" in payload and not bool(payload["future_valid_mask"].all()):
            errors.append("future mask contains invalid padding")
        if "episode_id" in payload and "step" in payload:
            pairs = list(zip(payload["episode_id"].tolist(), payload["step"].tolist()))
            if len(set(pairs)) != len(pairs):
                errors.append("duplicate episode/step sample")
    result = {
        "schema": "ref2dex.gate1_assembled_dataset_audit.v1",
        "run_status": "COMPLETED",
        "input": str(args.input.resolve()),
        "rows": n,
        "episodes": len(set(payload["episode_id"].tolist())) if n and "episode_id" in payload else 0,
        "effect_shape": list(payload["effect"].shape) if "effect" in payload else None,
        "interaction_shape": list(payload["interaction"].shape) if "interaction" in payload else None,
        "errors": errors,
        "gate1_dataset_ready": not errors and n > 0,
        "no_model_fit": True,
        "no_physics_replay": True,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
