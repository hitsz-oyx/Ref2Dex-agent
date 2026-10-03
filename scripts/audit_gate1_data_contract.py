#!/usr/bin/env python3
"""Audit whether saved contact-consequence records satisfy the Gate 1 data contract.

This is an offline contract audit only.  It does not fit a value model, replay
physics, or infer a Gate 1 result from task-outcome proxies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RETURN_KEYS = ("reward", "rewards", "return_to_go", "returns", "reward_components")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def inspect(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("record payload is not a mapping")
    n = len(payload.get("episode_id", []))
    if n <= 0:
        raise ValueError("empty episode_id")
    required = {
        key: tensor_shape(payload.get(key))
        for key in ("history", "actual_action", "future_state", "future_contact")
    }
    for key, shape in required.items():
        if shape is None or not shape or shape[0] != n:
            raise ValueError(f"missing or misaligned {key}: {shape}")
    episode_ids = [str(value) for value in payload["episode_id"]]
    episode_count = len(set(episode_ids))
    motion = payload.get("motion_id")
    motion_values = sorted(set(int(x) for x in motion.tolist())) if isinstance(motion, torch.Tensor) else []
    return_fields = [key for key in RETURN_KEYS if key in payload]
    outcome = payload.get("outcome")
    outcome_fields = sorted(outcome) if isinstance(outcome, dict) else []
    has_force_fields = all(key in payload for key in ("future_hand_force", "future_object_force"))
    has_relative_hand_pose = any(key in payload for key in ("future_hand_body_pose", "future_hand_pose", "future_rigid_state"))
    contract = {
        "has_history": True,
        "has_actual_action": True,
        "has_future_object_state": "future_state" in payload,
        "has_explicit_effect_definition": isinstance(payload.get("effect_definition"), str),
        "has_relative_hand_pose_or_velocity": has_relative_hand_pose,
        "has_future_force_fields": has_force_fields,
        "has_explicit_interaction_definition": isinstance(payload.get("interaction_definition"), str),
        "has_step_reward_or_return": bool(return_fields),
        "episode_grouped": episode_count <= n,
        "action_propensity_present": "propensity" in payload or "allocation_probabilities" in payload,
    }
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema": payload.get("schema"),
        "rows": n,
        "episodes": episode_count,
        "motions": motion_values,
        "shapes": required,
        "return_fields": return_fields,
        "outcome_fields": outcome_fields,
        "contract": contract,
        "interpretation": {
            "future_state_is_effect_only": False,
            "future_contact_is_full_interaction": False,
            "task_outcome_proxy_is_return": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True,
                        help="record file or directory; directories are searched recursively")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths: list[Path] = []
    for item in args.input:
        if item.is_file():
            paths.append(item)
        elif item.is_dir():
            paths.extend(sorted(item.rglob("records.pt")))
        else:
            raise FileNotFoundError(item)
    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise ValueError("no records.pt inputs")
    reports = []
    errors = []
    for path in paths:
        try:
            reports.append(inspect(path))
        except Exception as error:  # preserve every file-level failure
            errors.append({"path": str(path), "error": repr(error)})
    all_have_reward = bool(reports) and all(report["contract"]["has_step_reward_or_return"] for report in reports)
    all_have_effect = bool(reports) and all(report["contract"]["has_explicit_effect_definition"] for report in reports)
    all_have_interaction = bool(reports) and all(report["contract"]["has_explicit_interaction_definition"] for report in reports)
    result = {
        "schema": "ref2dex.gate1_data_contract_audit.v1",
        "run_status": "COMPLETED",
        "inputs": [str(path) for path in paths],
        "reports": reports,
        "errors": errors,
        "aggregate": {
            "files": len(reports),
            "rows": sum(report["rows"] for report in reports),
            "episodes": sum(report["episodes"] for report in reports),
            "all_have_step_reward_or_return": all_have_reward,
            "all_have_explicit_effect_definition": all_have_effect,
            "all_have_explicit_interaction_definition": all_have_interaction,
        },
        "gate1_ready": bool(reports) and not errors and all_have_reward and all_have_effect and all_have_interaction,
        "no_model_fit": True,
        "no_physics_replay": True,
        "interpretation": (
            "READY_FOR_GATE1"
            if reports and not errors and all_have_reward and all_have_effect and all_have_interaction
            else "NOT_READY_FOR_GATE1: saved records lack the complete reward/effect/interaction contract"
        ),
    }
    output = args.output.resolve()
    if ROOT not in output.parents:
        raise ValueError("output must stay inside the repository")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"run_status": result["run_status"], "gate1_ready": result["gate1_ready"],
                      "files": len(reports), "errors": len(errors),
                      "rows": result["aggregate"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
