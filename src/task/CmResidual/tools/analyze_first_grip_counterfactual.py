"""Audit frozen CmLite on paired first-grip actions from matched physics runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from src.task.CmResidual.cmlite import FrozenCmLite


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(path: Path, expected_mode: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if (data.get("schema") != "ref2dex.first_grip_counterfactual_arm.v1"
            or data.get("grip_reflex_mode") != expected_mode):
        raise ValueError(f"wrong first-grip arm schema/mode: {path}")
    ids = data["env_id"].tolist()
    if len(ids) != len(set(ids)) or any(not 0 <= int(value) < 64 for value in ids):
        raise ValueError(f"duplicate/invalid env IDs: {path}")
    return data


def match_arms(off: dict, boosted: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return only pre-intervention matched indices; reject state mismatch."""
    a = {int(value): i for i, value in enumerate(off["env_id"].tolist())}
    b = {int(value): i for i, value in enumerate(boosted["env_id"].tolist())}
    common = sorted(set(a) & set(b))
    ai = torch.tensor([a[key] for key in common], dtype=torch.long)
    bi = torch.tensor([b[key] for key in common], dtype=torch.long)
    if not common:
        return ai, bi, {"common": 0, "matched": 0}
    meta = ((off["motion_id"][ai] == boosted["motion_id"][bi]) &
            (off["start_frame"][ai] == boosted["start_frame"][bi]) &
            (off["progress"][ai] == boosted["progress"][bi]) &
            (off["executed_candidate"][ai] == 0) &
            (boosted["executed_candidate"][bi] == 1))
    q_error = (off["q"][ai] - boosted["q"][bi]).abs().amax(1)
    action_error = (off["base_action"][ai] - boosted["base_action"][bi]).abs().amax(1)
    xyz_error = (off["object_state"][ai, :3] - boosted["object_state"][bi, :3]).norm(dim=1)
    state_error = (off["object_state"][ai] - boosted["object_state"][bi]).abs().amax(1)
    valid = meta & (q_error <= 1e-4) & (action_error <= 1e-4) & (
        xyz_error <= 1e-5) & (state_error <= 1e-4)
    report = {
        "common": len(common), "matched": int(valid.sum().item()),
        "metadata_aligned": int(meta.sum().item()),
        "max_q_abs_diff": float(q_error.max().item()),
        "max_base_action_abs_diff": float(action_error.max().item()),
        "max_object_position_diff_m": float(xyz_error.max().item()),
        "max_object_state_abs_diff": float(state_error.max().item()),
    }
    return ai[valid], bi[valid], report


@torch.inference_mode()
def predict_effect(model: FrozenCmLite, data: dict, index: torch.Tensor,
                   batch_size: int = 4096) -> torch.Tensor:
    predictions = []
    for start in range(0, index.numel(), batch_size):
        chosen = index[start:start + batch_size]
        q, state = data["q"][chosen], data["object_state"][chosen]
        base = model.predict(q, data["base_action"][chosen], state)["delta_world"]
        boosted = model.predict(q, data["boosted_action"][chosen], state)["delta_world"]
        predictions.append(boosted - base)
    return torch.cat(predictions) if predictions else torch.empty((0, 3))


def effect_metrics(actual: torch.Tensor, predicted: torch.Tensor,
                   goal_direction: torch.Tensor) -> dict:
    if actual.shape != predicted.shape or actual.shape != goal_direction.shape:
        raise ValueError("effect and goal tensors must have matching [N,3] shape")
    if not actual.numel():
        return {"pairs": 0}
    actual_norm = actual.norm(dim=1)
    predicted_norm = predicted.norm(dim=1)
    error = (predicted - actual).norm(dim=1)
    zero = actual_norm.mean().item()
    informative = (actual_norm > 2e-4) & (goal_direction.norm(dim=1) > 1e-6)
    actual_projection = (actual * goal_direction).sum(1)
    predicted_projection = (predicted * goal_direction).sum(1)
    agreement = (actual_projection.sign() == predicted_projection.sign()) & informative
    return {
        "pairs": actual.shape[0],
        "actual_effect_mean_mm": float(actual_norm.mean().item() * 1000),
        "actual_effect_median_mm": float(actual_norm.median().item() * 1000),
        "actual_effect_p90_mm": float(torch.quantile(actual_norm, 0.9).item() * 1000),
        "predicted_effect_mean_mm": float(predicted_norm.mean().item() * 1000),
        "effect_epe_mm": float(error.mean().item() * 1000),
        "zero_effect_epe_mm": float(zero * 1000),
        "effect_epe_improvement_fraction": float(1 - error.mean().item() / max(zero, 1e-12)),
        "informative_pairs": int(informative.sum().item()),
        "goal_progress_sign_agreement": float(
            agreement.sum().item() / max(informative.sum().item(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, action="append", required=True)
    parser.add_argument("--always", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.off) != len(args.always) or len(args.off) != len(args.seed):
        raise ValueError("off, always, and seed lists must have equal length")
    if len(set(args.seed)) != len(args.seed) or args.output.exists():
        raise ValueError("seeds must be unique and output must be new")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("CmLite checkpoint SHA mismatch")
    model = FrozenCmLite(str(args.checkpoint), "cpu", args.checkpoint_sha256)
    records = []
    pooled_actual, pooled_predicted, pooled_goal = [], [], []
    for seed, off_path, always_path in zip(args.seed, args.off, args.always):
        off = load_arm(off_path, "off")
        always = load_arm(always_path, "always")
        ai, bi, match = match_arms(off, always)
        actual = always["next_object_state"][bi, :3] - off["next_object_state"][ai, :3]
        predicted = predict_effect(model, off, ai)
        goal = off["goal_position"][ai] - off["object_state"][ai, :3]
        pooled_actual.append(actual)
        pooled_predicted.append(predicted)
        pooled_goal.append(goal)
        records.append({
            "seed": seed, "off": str(off_path.resolve()), "off_sha256": sha256(off_path),
            "always": str(always_path.resolve()), "always_sha256": sha256(always_path),
            "match": match, "effect": effect_metrics(actual, predicted, goal),
        })
    total_matched = sum(item["match"]["matched"] for item in records)
    pooled = effect_metrics(torch.cat(pooled_actual), torch.cat(pooled_predicted),
                            torch.cat(pooled_goal))
    if total_matched < 40 or any(item["match"]["matched"] < 15 for item in records):
        verdict = "INCONCLUSIVE_PRESTATE_MISMATCH"
    elif pooled["informative_pairs"] < 20:
        verdict = "INCONCLUSIVE_SMALL_ONE_STEP_EFFECT"
    elif (pooled["effect_epe_improvement_fraction"] >= 0.2 and
          pooled["goal_progress_sign_agreement"] >= 0.6):
        verdict = "PASSED_LOCAL_CAUSAL_EFFECT_GATE"
    else:
        verdict = "FAILED_LOCAL_CAUSAL_EFFECT_GATE"
    report = {
        "schema": "ref2dex.first_grip_counterfactual_audit.v1",
        "run_status": "COMPLETED", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "work_version": "V1.43", "seeds": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "per_seed": records, "pooled": pooled, "verdict": verdict,
        "official_policy_checkpoint": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": pooled, "verdict": verdict}, sort_keys=True))


if __name__ == "__main__":
    main()
