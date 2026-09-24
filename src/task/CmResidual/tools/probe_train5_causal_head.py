"""Object-level LOO signed Cm effect test on five official-origin train objects."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.task.CmResidual.cmlite import local_translation_target
from src.task.CmResidual.tools.probe_crossobject_causal_head import (
    effect_estimate, fit, inputs, evaluate,
)
from src.task.CmResidual.tools.probe_expert_crossobject_cm import ROOT, calibrate
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


SOURCE = ROOT / "outputs/CmResidual/agent_expert_train5_randomized_s186_h5"
SOURCE_SHA = "7f196f3df518732156bacd852b24b3359dc389f1770bab1f06ceed980e1c2559"
SPLIT_SHA = "694b74889e5d5523be3b96485e5e91bdc31fb420aec2766ec0fb4a2dc40d1c3c"
OFFICIAL_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"
MOTION_NAMES = ["cubesmall", "mug", "toothpaste", "waterbottle", "airplane", "cubesmall"]
OBJECT_NAMES = ["airplane", "cubesmall", "mug", "toothpaste", "waterbottle"]


def load_rows(*, source: Path = SOURCE, source_sha: str = SOURCE_SHA,
              split_sha: str = SPLIT_SHA, source_objects: list[str] = OBJECT_NAMES,
              motion_names: list[str] = MOTION_NAMES,
              partition: str = "train") -> dict:
    manifest = json.loads((source / "run_manifest.json").read_text())
    path = source / "transitions.pt"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("source_partition") != partition or
            manifest.get("source_actor_role") != "official_data_collector" or
            manifest.get("source_objects") != source_objects or
            manifest.get("checkpoint_sha256") != OFFICIAL_SHA or
            manifest.get("motion_manifest_sha256") != split_sha or
            manifest.get("transitions_sha256") != source_sha or
            sha256(path) != source_sha):
        raise ValueError("randomized source drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("run_status") != "COMPLETED" or
            payload.get("schema") != "ref2dex.randomized_action_followup.v1" or
            payload.get("followup_horizon") != 5 or
            payload.get("intervention_axes") != [2] or
            payload.get("delta_z_action") != .1):
        raise ValueError("randomized transition schema drift")
    full = payload["records"]
    take = full["assignment"].reshape(-1) != 0
    if not full["pre_contact"][take].all():
        raise ValueError("treated state without pre-contact")
    dose = full["executed_action"] - full["base_action"]
    expected = torch.zeros_like(dose)
    expected[:, 2] = full["assignment"] * .1
    if not torch.allclose(dose, expected, atol=1e-5):
        raise ValueError("executed action dose mismatch")
    rows = {key: value[take].clone() for key, value in full.items()}
    rows["env_id"] = torch.arange(len(take))[take] % 64
    rows["action"] = rows["executed_action"].float()
    rows["target"] = local_translation_target(
        rows["object_state"], rows["followup_object_state"])
    rows["category"] = torch.zeros(len(rows["q"]), dtype=torch.long)
    rows["contact_fraction"] = rows["followup_contact_count"].float() / 5
    survived = ((rows["followup_progress"] - rows["progress"] == 5) &
                (rows["followup_reset"] == 0))
    if not survived.all() or set(rows["motion_id"].tolist()) != set(range(len(motion_names))):
        raise ValueError("incomplete followup or motion ID set")
    if not all(torch.isfinite(value.float()).all() for value in rows.values()):
        raise FloatingPointError("non-finite Cm source")
    return rows


def select(rows: dict, motion_ids: list[int],
           motion_names: list[str] = MOTION_NAMES) -> tuple[dict, list[str]]:
    mask = torch.isin(rows["motion_id"], torch.tensor(motion_ids))
    part = {key: value[mask].clone() for key, value in rows.items()}
    mapping = {old: new for new, old in enumerate(motion_ids)}
    part["motion_id"] = torch.tensor([mapping[int(i)] for i in part["motion_id"]])
    return part, [motion_names[i] for i in motion_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 500:
        parser.error("new output and 1..500 steps required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_sha256": SOURCE_SHA, "split_sha256": SPLIT_SHA,
                "folds": OBJECT_NAMES, "motion_id_to_object": MOTION_NAMES,
                "steps": args.steps, "gpu_count": 0, "cpu_threads": 2,
                "wall_budget_minutes": 30, "output_budget_mb": 10,
                "stop_rule": "source drift, object leakage, non-finite or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        rows = load_rows()
        results = {}
        for fold, object_name in enumerate(OBJECT_NAMES):
            held_ids = [i for i, name in enumerate(MOTION_NAMES) if name == object_name]
            train_ids = [i for i, name in enumerate(MOTION_NAMES) if name != object_name]
            train, train_names = select(rows, train_ids)
            val, val_names = select(rows, held_ids)
            if object_name in train_names or set(train_names) & set(val_names):
                raise ValueError("object identity leaked into fit fold")
            setup = calibrate(train)
            models = {}
            for kind in ("geometric", "raw"):
                train_state, train_difference = inputs(train, train_names, setup, kind)
                val_state, val_difference = inputs(val, val_names, setup, kind)
                model = fit(train, train_state, train_difference,
                            seed=240970 + fold, steps=args.steps)
                models[kind] = evaluate(model, val, val_state, val_difference,
                                        seed=240980 + fold)
            models["constant"] = {"predicted_ate_mm": effect_estimate(train),
                                  "actual_ate_mm": effect_estimate(val)}
            models["fit_rows"] = len(train["q"])
            models["heldout_rows"] = len(val["q"])
            results[object_name] = models
            print(json.dumps({"completed_fold": object_name,
                              "actual_ate_mm": models["constant"]["actual_ate_mm"]},
                             sort_keys=True), flush=True)
        mae = {kind: float(np.mean([abs(results[name][kind]["predicted_ate_mm"] -
                                        results[name][kind]["actual_ate_mm"])
                                    for name in OBJECT_NAMES]))
               for kind in ("geometric", "raw", "constant")}
        signs = sum(np.sign(results[name]["geometric"]["predicted_ate_mm"]) ==
                    np.sign(results[name]["geometric"]["actual_ate_mm"])
                    for name in OBJECT_NAMES)
        gate = mae["geometric"] <= .8 * min(mae["raw"], mae["constant"]) and signs >= 4
        report = {"schema": "ref2dex.train5_causal_head_loo.v1",
                  "classification": "Probe", "source_actor_role": "official_data_collector",
                  "folds": results, "object_ate_mae_mm": mae,
                  "geometric_sign_matches": int(signs),
                  "continue_to_alarmclock": gate,
                  "elapsed_seconds": time.monotonic() - started}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(report_path))
        print(json.dumps({"object_ate_mae_mm": mae,
                          "geometric_sign_matches": int(signs),
                          "continue_to_alarmclock": gate}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
