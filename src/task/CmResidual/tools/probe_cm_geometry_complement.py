"""Frozen two-model ensemble test on fresh randomized contact followups."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.task.CmResidual.contact_aware_cm import ContactAwareCm, RawContactAwareCm
from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
import src.task.CmResidual.tools.probe_contact_aware_cm as previous


ROOT = Path(__file__).resolve().parents[4]
MODEL_ROOT = ROOT / "outputs/CmResidual/agent_contact_aware_cm_s151152_train_s153154_test"
MODEL_SHA = {
    "geometric": "5657699f4fd4b97a0162e6ccd098ffadb073d05a27828d2e616c2fb997dfca23",
    "raw_action": "ce711dca743fb47ff84adf45f3f580e4afd0237f7f0defd55a4ec411a5c34d5d",
}
TEST_DATA = {
    155: (ROOT / "outputs/CmResidual/agent_randomized_followup_s155_d01_h5_n64/transitions.pt",
          "5b1e64f0ced1aad299db0adf5445521829c7ade19999c5ec3892c2007efc0f7c"),
    156: (ROOT / "outputs/CmResidual/agent_randomized_followup_s156_d01_h5_n64/transitions.pt",
          "0dbf7ae466eace755ea41ae44a6aff610faa2c09a7e3091804b0fa4a197692f5"),
}


def group_gap(outcome, assignment, stratum, low, high, indices):
    def group_ate(mask):
        chosen = indices[mask[indices]]
        return weighted_step_difference(outcome[chosen], assignment[chosen],
                                        stratum[chosen])[0]
    low_ate = group_ate(low)
    high_ate = group_ate(high)
    return {"low_ate": low_ate, "high_ate": high_ate,
            "high_minus_low": high_ate - low_ate}


def paired_ranking(scores, outcome, assignment, stratum, env_id, *, bootstraps, seed):
    if set(scores) != {"geometric", "raw_action", "ensemble"}:
        raise ValueError("expected fixed three scores")
    groups = {}
    all_indices = np.arange(len(outcome))
    for name, score in scores.items():
        q25, q75 = np.quantile(score, [.25, .75])
        if q75 - q25 < 1e-6:
            raise ValueError(f"score spread too small: {name}")
        low, high = score <= q25, score >= q75
        if min(low.sum(), high.sum()) < 80:
            raise ValueError(f"quartile too sparse: {name}")
        groups[name] = (low, high)
    observed = {name: group_gap(outcome, assignment, stratum, *groups[name], all_indices)
                for name in scores}
    identifiers = np.unique(env_id)
    clusters = [np.flatnonzero(env_id == identifier) for identifier in identifiers]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in scores}
    differences = []
    for _ in range(bootstraps):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in selected])
        try:
            gaps = {name: group_gap(outcome, assignment, stratum, *groups[name], indices)
                    ["high_minus_low"] for name in scores}
        except ValueError:
            continue
        for name, gap in gaps.items():
            draws[name].append(gap)
        differences.append(gaps["ensemble"] - gaps["raw_action"])
    if len(differences) < .9 * bootstraps:
        raise ValueError("too few paired cluster bootstraps")
    return {"models": {name: {**observed[name],
                               "cluster_95ci": np.quantile(draws[name], [.025, .975]).tolist()}
                       for name in scores},
            "ensemble_minus_raw_gap": observed["ensemble"]["high_minus_low"] -
                                      observed["raw_action"]["high_minus_low"],
            "ensemble_minus_raw_gap_paired_cluster_95ci": np.quantile(
                differences, [.025, .975]).tolist(),
            "effective_bootstraps": len(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or not 100 <= args.bootstraps <= 5000:
        parser.error("new output and bounded bootstrap count required")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "test_seed_sha256": {seed: TEST_DATA[seed][1] for seed in TEST_DATA},
                "checkpoint_sha256": MODEL_SHA, "ensemble_weight": .5,
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 30,
                "output_budget_mb": 10,
                "stop_rule": "input/model drift, non-finite score or sparse randomization"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        for name, digest in MODEL_SHA.items():
            if sha256(MODEL_ROOT / f"{name}.pt") != digest:
                raise ValueError(f"checkpoint SHA drift: {name}")
        previous.DATA.update(TEST_DATA)
        geometry = previous.make_geometry()
        models = {"geometric": ContactAwareCm().eval(),
                  "raw_action": RawContactAwareCm(torch.zeros(67), torch.ones(67)).eval()}
        for name, model in models.items():
            payload = torch.load(MODEL_ROOT / f"{name}.pt", map_location="cpu",
                                 weights_only=False)
            if payload.get("schema") != "ref2dex.contact_aware_cm.v1":
                raise ValueError("model schema mismatch")
            model.load_state_dict(payload["model"], strict=True)
        reports = {}
        pooled = {key: [] for key in ("outcome", "assignment", "stratum", "env_id")}
        scores = {key: [] for key in ("geometric", "raw_action", "ensemble")}
        with torch.no_grad():
            for seed_index, seed in enumerate(TEST_DATA):
                rows = previous.load_rows(seed)
                data = previous.model_input(rows, geometry)
                predicted = {name: previous.predict(model, data,
                                                    "raw" if name == "raw_action" else "geometric")
                             ["contact_fraction"] for name, model in models.items()}
                predicted["ensemble"] = .5 * (predicted["geometric"] + predicted["raw_action"])
                reports[seed] = {
                    "n": len(rows["q"]),
                    "contact_rmse": {name: float(((value - data["contact"]).square().mean()).sqrt())
                                     for name, value in predicted.items()},
                    "observed_ate": weighted_step_difference(
                        rows["contact_target"].numpy(), rows["assignment"].numpy(),
                        rows["step"].numpy())[0],
                }
                for name, model in models.items():
                    scores[name].append(previous.candidate_contact_score(
                        model, rows, geometry,
                        kind="raw" if name == "raw_action" else "geometric"))
                scores["ensemble"].append(.5 * (scores["geometric"][-1] +
                                                 scores["raw_action"][-1]))
                pooled["outcome"].append(rows["contact_target"].numpy())
                pooled["assignment"].append(rows["assignment"].numpy())
                pooled["stratum"].append((rows["step"] + seed_index * 1000).numpy())
                pooled["env_id"].append((rows["env_id"] + seed_index * 64).numpy())
        pooled = {key: np.concatenate(value) for key, value in pooled.items()}
        scores = {key: np.concatenate(value) for key, value in scores.items()}
        rank = paired_ranking(scores, pooled["outcome"], pooled["assignment"],
                              pooled["stratum"], pooled["env_id"],
                              bootstraps=args.bootstraps, seed=240935)
        passed = (all(reports[seed]["contact_rmse"]["ensemble"] <
                      reports[seed]["contact_rmse"]["raw_action"] for seed in TEST_DATA)
                  and rank["ensemble_minus_raw_gap"] >= .03
                  and rank["ensemble_minus_raw_gap_paired_cluster_95ci"][0] > 0)
        report = {"schema": "ref2dex.cm_geometry_complement_probe.v1",
                  "run_status": "COMPLETED", "test_prediction": reports,
                  "contact_effect_ranking_pooled": rank,
                  "prespecified_continue_gate_passed": bool(passed),
                  "elapsed_seconds": time.monotonic() - started,
                  "limit": "Probe of fixed frozen-model complementarity, not policy utility"}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path))
        print(json.dumps({"run_status": "COMPLETED", "gate": passed,
                          "test_prediction": reports, "ranking": rank}, sort_keys=True))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
