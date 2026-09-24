"""Test whether fixed Cm scores stratify randomized immediate treatment effects."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.task.CmResidual.cmlite import local_to_world_translation
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits
from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_intervention_handflow import INPUTS, sha256
from src.task.CmResidual.tools.probe_local_geometric_cm import (
    SixRegionCm, _surface_geometry_class, extract_features,
)
from src.task.CmResidual.tools.probe_randomized_geometric_cm import (
    CALIBRATION, CALIBRATION_SHA256, RawActionEffect, load_randomized_rows,
    raw_inputs,
)
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT_ROOT = ROOT / "outputs/CmResidual/agent_randomized_geometric_cm_400"
CHECKPOINT_SHA256 = {
    "geometric": "d9069bf2845fc26474b4ad38dfab44eef1094ff0f2bc3b62988aafd981a153f6",
    "raw_action": "9414543961c818bd44fb0b099e41f75d5e6024753d3fb8d98f13166c1fb35360",
}
TEST_KEYS = ("test_s146_d03", "test_s148_d01")


def group_ate(y, assignment, step, group):
    return weighted_step_difference(y[group], assignment[group], step[group])[0]


def ranking_report(score, y, assignment, step, env_id, *, seed, bootstraps):
    lower, upper = np.quantile(score, [.25, .75])
    if not np.isfinite(lower + upper) or upper - lower < 1e-6:
        raise ValueError("model treatment scores have no useful spread")
    low, high = score <= lower, score >= upper
    if min(low.sum(), high.sum()) < 40:
        raise ValueError("too few states in effect-score quartiles")
    low_ate = group_ate(y, assignment, step, low)
    high_ate = group_ate(y, assignment, step, high)
    rng = np.random.default_rng(seed)
    unique_envs = np.unique(env_id)
    differences = []
    for _ in range(bootstraps):
        sampled_envs = rng.choice(unique_envs, len(unique_envs), replace=True)
        indices = np.concatenate([np.flatnonzero(env_id == env) for env in sampled_envs])
        try:
            low_effect = group_ate(y[indices], assignment[indices], step[indices],
                                   low[indices])
            high_effect = group_ate(y[indices], assignment[indices], step[indices],
                                    high[indices])
        except ValueError:
            continue
        differences.append(high_effect - low_effect)
    if len(differences) < bootstraps * .9:
        raise ValueError("bootstrap effective resamples too few")
    return {
        "score_q25_mm": float(lower), "score_q75_mm": float(upper),
        "score_mean_low_mm": float(score[low].mean()),
        "score_mean_high_mm": float(score[high].mean()),
        "low_count": int(low.sum()), "high_count": int(high.sum()),
        "observed_ate_low_mm": low_ate,
        "observed_ate_high_mm": high_ate,
        "observed_ate_high_minus_low_mm": high_ate - low_ate,
        "cluster_bootstrap_95ci_mm": np.quantile(differences, [.025, .975]).tolist(),
        "bootstrap_effective_resamples": len(differences),
    }


@torch.no_grad()
def score_dataset(rows, model_geom, model_raw, *, kinematics, geometry, lower, upper,
                  calibration):
    predictions = {}
    for sign, label in ((1, "plus"), (-1, "minus")):
        changed = dict(rows)
        changed["action"] = rows["base_action"].clone()
        changed["action"][:, 2] = (
            changed["action"][:, 2] + sign * rows["delta_z_action"]).clamp(-1, 1)
        features = extract_features(changed, kinematics=kinematics, geometry=geometry,
                                    lower=lower, upper=upper, calibration=calibration,
                                    batch_size=32)
        predictions[label] = {
            "geometric": model_geom(features["region"], features["context"])["delta_local"],
            "raw_action": model_raw(raw_inputs(changed)),
        }
    score = {}
    for name in predictions["plus"]:
        local = predictions["plus"][name] - predictions["minus"][name]
        score[name] = (local_to_world_translation(rows["object_state"], local)[:, 2]
                       * 1000).numpy()
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=240927)
    args = parser.parse_args()
    if args.output.exists() or not 100 <= args.bootstraps <= 10000:
        raise ValueError("new output and bounded bootstrap count required")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "run_id": args.output.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "test_sha256": {key: INPUTS[key][1] for key in TEST_KEYS},
        "calibration_sha256": CALIBRATION_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 20,
        "output_budget_mb": 10, "bootstraps": args.bootstraps,
        "stop_rule": "input/checkpoint drift, non-finite score, sparse groups or wall budget",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    start = time.monotonic()
    try:
        torch.set_num_threads(2)
        if sha256(CALIBRATION) != CALIBRATION_SHA256:
            raise ValueError("calibration SHA drift")
        for name, digest in CHECKPOINT_SHA256.items():
            if sha256(CHECKPOINT_ROOT / f"{name}.pt") != digest:
                raise ValueError(f"{name} checkpoint SHA drift")
        coefficients = torch.tensor(json.loads(CALIBRATION.read_text())
                                    ["coefficients"]["action_velocity"])
        calibration = {"alpha": coefficients[:, 0], "velocity": coefficients[:, 1],
                       "bias": coefficients[:, 2]}
        hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
        kinematics = TorchInspireKinematics(hand_urdf, "cpu")
        geometry = _surface_geometry_class()(
            hand_urdf=hand_urdf, object_urdf=ASSETS / "mjcf/airplane.urdf",
            query_links=QUERY_LINKS, object_count=256, hand_count=256,
            seed=42, device="cpu")
        lower, upper = native_joint_limits(hand_urdf, "cpu")
        model_geom = SixRegionCm().eval()
        model_geom.load_state_dict(torch.load(CHECKPOINT_ROOT / "geometric.pt",
                                              map_location="cpu", weights_only=False)["model"])
        model_raw = RawActionEffect(torch.zeros(67), torch.ones(67)).eval()
        model_raw.load_state_dict(torch.load(CHECKPOINT_ROOT / "raw_action.pt",
                                             map_location="cpu", weights_only=False)["model"])
        reports = {}
        for key in TEST_KEYS:
            rows = load_randomized_rows(key)
            score = score_dataset(rows, model_geom, model_raw, kinematics=kinematics,
                                  geometry=geometry, lower=lower, upper=upper,
                                  calibration=calibration)
            payload = torch.load(INPUTS[key][0], map_location="cpu", weights_only=False)
            full = payload["records"]
            selected = (full["assignment"].reshape(-1) != 0).numpy()
            assignment = full["assignment"].reshape(-1).numpy()[selected]
            step = full["global_step"].reshape(-1).numpy()[selected]
            env_id = (np.arange(len(selected)) % 64)[selected]
            observed = ((full["next_object_state"][:, 2] -
                         full["object_state"][:, 2]) * 1000).numpy()[selected]
            if not all(np.isfinite(value).all() for value in (*score.values(), observed)):
                raise FloatingPointError("non-finite score or observed effect")
            if any(len(value) != len(observed) for value in score.values()):
                raise ValueError("model score and randomized outcome alignment mismatch")
            torch.save({"schema": "ref2dex.cm_cate_score.v1", "test_key": key,
                        "score_mm": {name: torch.from_numpy(value) for name, value in score.items()},
                        "observed_dz_mm": torch.from_numpy(observed),
                        "assignment": torch.from_numpy(assignment),
                        "global_step": torch.from_numpy(step),
                        "env_id": torch.from_numpy(env_id)}, args.output / f"{key}_scores.pt")
            reports[key] = {
                "samples": len(observed),
                "overall_ate_mm": weighted_step_difference(observed, assignment, step)[0],
                "models": {name: ranking_report(value, observed, assignment, step, env_id,
                                                 seed=args.seed, bootstraps=args.bootstraps)
                           for name, value in score.items()},
            }
            print(json.dumps({"key": key, "overall_ate_mm": reports[key]["overall_ate_mm"],
                              "high_minus_low_mm": {
                                  name: value["observed_ate_high_minus_low_mm"]
                                  for name, value in reports[key]["models"].items()}},
                             sort_keys=True), flush=True)
        report = {"schema": "ref2dex.cm_cate_ranking_probe.v1",
                  "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
                  "reports": reports, "elapsed_seconds": time.monotonic() - start,
                  "limit": "conditional average effects under sequential randomization; not individual pairs or PPO utility"}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path),
                        scores_sha256={key: sha256(args.output / f"{key}_scores.pt")
                                       for key in TEST_KEYS})
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
