"""Train fixed multiaxis ten-step Cm variants on randomized physical transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch

from src.task.CmResidual.cmlite import (
    local_to_world_translation, local_translation_target,
)
from src.task.CmResidual.contact_aware_cm import ContactAwareCm, RawContactAwareCm
from src.task.CmResidual.tools.analyze_multiaxis_followup import load_run, sha256
from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.probe_contact_aware_cm import (
    make_geometry, model_input, prediction_metrics, train,
)


ROOT = Path(__file__).resolve().parents[4]
AXES = (0, 1, 2)


def load_rows(path: Path, digest: str) -> dict:
    if sha256(path) != digest:
        raise ValueError("Cm input SHA mismatch")
    load_run(path, 64)  # verify randomized assignment and exact executed dose
    payload = torch.load(path, map_location="cpu", weights_only=False)
    full = payload["records"]
    selected = full["assignment"] != 0
    rows = {"q": full["q"][selected].float(),
            "dof_vel": full["dof_vel"][selected].float(),
            "action": full["executed_action"][selected].float(),
            "base_action": full["base_action"][selected].float(),
            "object_state": full["object_state"][selected].float(),
            "next_object_state": full["next_object_state"][selected].float(),
            "followup_object_state": full["followup_object_state"][selected].float(),
            "contact_target": full["followup_contact_count"][selected].float() / 10,
            "assignment": full["assignment"][selected].reshape(-1),
            "step": full["global_step"][selected].reshape(-1)}
    rows["step_target"] = local_translation_target(rows["object_state"],
                                                    rows["next_object_state"])
    rows["followup_target"] = local_translation_target(rows["object_state"],
                                                        rows["followup_object_state"])
    rows["target"] = rows["step_target"]
    rows["category"] = (rows["target"].norm(dim=-1) <= .002).long()
    if not all(torch.isfinite(value.float()).all() for value in rows.values()):
        raise FloatingPointError("non-finite multiaxis Cm row")
    return rows


def concatenate(parts: list[dict]) -> dict:
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def no_action_data(data: dict) -> dict:
    result = dict(data)
    result["raw"] = data["raw"].clone()
    result["raw"][:, 18:36] = 0
    return result


def stratified_schedule(assignment: torch.Tensor, steps: int,
                        batch_size: int, seed: int) -> list[torch.Tensor]:
    if batch_size % 6 or steps < 1:
        raise ValueError("batch must have equal six-cell quotas")
    groups = [torch.where(assignment == code)[0]
              for code in (1, -1, 2, -2, 3, -3)]
    if any(len(group) < 16 for group in groups):
        raise ValueError("too few samples in one randomized action cell")
    generator = torch.Generator().manual_seed(seed)
    per = batch_size // 6
    return [torch.cat([group[torch.randint(len(group), (per,), generator=generator)]
                       for group in groups]) for _ in range(steps)]


def predict(model, data: dict, kind: str) -> dict:
    return model(data["raw"]) if kind == "raw" else model(data["region"], data["context"])


def factual_metrics(model, data: dict, rows: dict, kind: str) -> dict:
    with torch.no_grad():
        output = predict(model, data, kind)
        world = local_to_world_translation(rows["object_state"],
                                           output["followup_delta_local"])
        actual = rows["followup_object_state"][:, :3] - rows["object_state"][:, :3]
        metrics = prediction_metrics(model, data, kind=kind)
        metrics["world_axis_h10_rmse_mm"] = {
            ("x", "y", "z")[axis]: float(((world[rows["assignment"].abs() == axis + 1, axis] -
                                            actual[rows["assignment"].abs() == axis + 1, axis])
                                           .square().mean().sqrt()) * 1000)
            for axis in AXES}
        return metrics


def candidate_data(rows: dict, geometry: dict, axis: int, sign: int) -> dict:
    candidate = dict(rows)
    candidate["action"] = rows["base_action"].clone()
    candidate["action"][:, axis] += sign * .1
    if (candidate["action"][:, axis].abs() > 1 + 1e-6).any():
        raise ValueError("candidate action clips")
    return model_input(candidate, geometry)


def observed_effect(rows: dict, axis: int) -> dict:
    selected = rows["assignment"].abs() == axis + 1
    assignment = rows["assignment"][selected].sign().numpy()
    steps = rows["step"][selected].numpy()
    displacement = (rows["followup_object_state"][selected, axis] -
                    rows["object_state"][selected, axis]).numpy() * 1000
    contact = rows["contact_target"][selected].numpy() * 100
    return {"h10_object_mm": weighted_step_difference(displacement, assignment, steps)[0],
            "h10_contact_pp": weighted_step_difference(contact, assignment, steps)[0],
            "count": int(selected.sum())}


def candidate_effects(models: dict, rows: dict, geometry: dict) -> dict:
    report = {name: {} for name in models}
    for axis in AXES:
        selected = rows["assignment"].abs() == axis + 1
        subset = {key: value[selected] for key, value in rows.items()}
        plus, minus = (candidate_data(subset, geometry, axis, sign)
                       for sign in (1, -1))
        for name, (model, kind) in models.items():
            positive = no_action_data(plus) if name == "state_only" else plus
            negative = no_action_data(minus) if name == "state_only" else minus
            with torch.no_grad():
                pos = predict(model, positive, kind)
                neg = predict(model, negative, kind)
                pos_world = local_to_world_translation(subset["object_state"],
                                                       pos["followup_delta_local"])
                neg_world = local_to_world_translation(subset["object_state"],
                                                       neg["followup_delta_local"])
                report[name][("x", "y", "z")[axis]] = {
                    "predicted_h10_object_mm": float((pos_world[:, axis] -
                                                      neg_world[:, axis]).mean() * 1000),
                    "predicted_h10_contact_pp": float((pos["contact_fraction"] -
                                                       neg["contact_fraction"]).mean() * 100)}
    return report


def execute(args, manifest: dict) -> None:
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(240941)
    geometry = make_geometry()
    train_rows = concatenate([load_rows(path, digest) for path, digest in args.train])
    test_rows = load_rows(args.test[0], args.test[1])
    train_data = model_input(train_rows, geometry)
    test_data = model_input(test_rows, geometry)
    mean = train_data["raw"].mean(0)
    std = train_data["raw"].std(0).clamp_min(.01)
    raw = RawContactAwareCm(mean, std)
    state_only = RawContactAwareCm(mean, std)
    state_only.load_state_dict(raw.state_dict(), strict=True)
    geometric = ContactAwareCm()
    models = {"raw_action": (raw, "raw"),
              "state_only": (state_only, "raw"),
              "geometric": (geometric, "geometric")}
    schedule = stratified_schedule(train_rows["assignment"], args.steps, 96, 240942)
    losses = {"raw_action": train(raw, train_data, schedule, kind="raw"),
              "state_only": train(state_only, no_action_data(train_data), schedule, kind="raw"),
              "geometric": train(geometric, train_data, schedule, kind="geometric")}
    factual = {name: factual_metrics(model,
                                    no_action_data(test_data) if name == "state_only" else test_data,
                                    test_rows, kind)
               for name, (model, kind) in models.items()}
    observed = {("x", "y", "z")[axis]: observed_effect(test_rows, axis)
                for axis in AXES}
    contrast = candidate_effects(models, test_rows, geometry)
    baseline_rmse = factual["state_only"]["world_axis_h10_rmse_mm"]["x"]
    actual_x = observed["x"]["h10_object_mm"]
    gate = {name: {
        "x_rmse_improvement_fraction": 1 - factual[name]["world_axis_h10_rmse_mm"]["x"] /
        baseline_rmse,
        "predicted_over_observed_x_effect": contrast[name]["x"]["predicted_h10_object_mm"] /
        actual_x if abs(actual_x) > 1e-8 else None}
        for name in ("raw_action", "geometric")}
    report = {"schema": "ref2dex.multiaxis_h10_cm_probe.v1",
              "run_status": "COMPLETED", "train_samples": len(train_rows["q"]),
              "test_samples": len(test_rows["q"]),
              "parameter_count": {name: sum(p.numel() for p in model.parameters())
                                  for name, (model, _) in models.items()},
              "final_train_loss": losses, "factual_prediction": factual,
              "observed_randomized_effect": observed,
              "candidate_contrast_prediction": contrast,
              "gate_components": gate,
              "elapsed_seconds": time.monotonic() - started,
              "limits": "Probe: held-out action-effect prediction, not policy utility"}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, (model, _) in models.items():
        torch.save({"schema": "ref2dex.multiaxis_h10_cm.v1", "name": name,
                    "model": model.state_dict()}, args.output / f"{name}.pt")
    manifest.update(run_status="COMPLETED", final_step=args.steps,
                    report_sha256=sha256(report_path),
                    checkpoint_sha256={name: sha256(args.output / f"{name}.pt")
                                       for name in models},
                    elapsed_seconds=time.monotonic() - started)
    print(json.dumps({"run_status": "COMPLETED", "observed_x": actual_x,
                      "gate_components": gate}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs=2, action="append", required=True,
                        metavar=("PATH", "SHA256"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    args.train = [(Path(path), digest) for path, digest in args.train]
    args.test = (Path(args.test[0]), args.test[1])
    if (args.output.exists() or len(args.train) != 2 or
            not 1 <= args.steps <= 1000 or
            any(path == args.test[0] for path, _ in args.train)):
        parser.error("new output, two disjoint train inputs and bounded updates required")
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "experiment_id": "P-20260924-multiaxis-h10-cm",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "train_input_sha256": {str(path.resolve()): digest for path, digest in args.train},
                "test_input_sha256": {str(args.test[0].resolve()): args.test[1]},
                "steps": args.steps, "batch_size": 96, "cpu_threads": 2,
                "gpu_count": 0, "wall_budget_minutes": 10,
                "output_budget_mb": 100,
                "stop_rule": "source drift, leakage, non-finite output or budget"}
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        execute(args, manifest)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
