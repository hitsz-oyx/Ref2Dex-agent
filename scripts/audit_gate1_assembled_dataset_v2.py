#!/usr/bin/env python3
"""Contract audit for corrected Gate 1 v2 windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REQUIRED = {
    "history_state", "history_previous_action", "history_context", "history_progress",
    "action", "effect", "interaction", "return_to_go", "future_valid_mask",
    "future_action", "episode_id", "step", "source_run", "noise_std", "episode_auxiliary", "metadata",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    errors = sorted(REQUIRED.difference(data))
    n = int(data["return_to_go"].shape[0]) if "return_to_go" in data else 0
    if n == 0:
        errors.append("empty dataset")
    for key in REQUIRED:
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] != n:
            errors.append(f"row misalignment: {key} {tuple(value.shape)}")
    metadata = data.get("metadata", {})
    for key in ("history_layout", "effect_layout", "interaction_layout", "future_action_layout",
                "target_definition", "split_unit"):
        if not isinstance(metadata.get(key), str):
            errors.append(f"missing metadata.{key}")
    for key in ("history_state", "history_previous_action", "history_context", "history_progress",
                "action", "effect", "interaction", "future_action", "return_to_go"):
        if key in data and not torch.isfinite(data[key]).all():
            errors.append(f"non-finite: {key}")
    if "future_valid_mask" in data and not bool(data["future_valid_mask"].all()):
        errors.append("future padding mask is not full")
    if n and "episode_id" in data and "step" in data:
        pairs = list(zip(data["source_run"].tolist(), data["episode_id"].tolist(), data["step"].tolist()))
        if len(set(pairs)) != len(pairs):
            errors.append("duplicate episode/step")
    result = {
        "schema": "ref2dex.gate1_assembled_dataset_v2_audit.v1",
        "run_status": "COMPLETED",
        "input": str(args.input.resolve()),
        "rows": n,
        "episodes": len(set(zip(data["source_run"].tolist(), data["episode_id"].tolist()))) if n and "source_run" in data else 0,
        "history_shape": list(data["history_state"].shape) if "history_state" in data else None,
        "effect_shape": list(data["effect"].shape) if "effect" in data else None,
        "interaction_shape": list(data["interaction"].shape) if "interaction" in data else None,
        "errors": errors,
        "v2_dataset_ready": not errors,
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
