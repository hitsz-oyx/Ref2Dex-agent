"""Held-out structured sequence Cm versus matched state-only capacity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.task.CmResidual.cm_two_step_structured import StructuredTwoStepCm
from src.task.CmResidual.tools.analyze_two_step_sequence import contrasts, sha256
from src.task.CmResidual.tools.probe_two_step_cm import (
    candidate_effect, factual, load_data, schedule, train,
)


ROOT = Path(__file__).resolve().parents[4]


def execute(args, manifest: dict) -> None:
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(241150)
    parts = [load_data(path, digest) for path, digest in args.train]
    data = {key: torch.cat([part[key] for part in parts])
            for key in ("raw", "raw_base", "object_state", "step", "followup",
                        "contact", "assignment", "global_step")}
    test = load_data(*args.test)
    mean = data["raw_base"].mean(0)
    std = data["raw_base"].std(0).clamp_min(.01)
    action = StructuredTwoStepCm(mean, std)
    state_only = StructuredTwoStepCm(mean, std)
    state_only.load_state_dict(action.state_dict(), strict=True)
    batches = schedule(data["assignment"], args.steps)
    losses = {"sequence_action": train(action, data, batches, state_only=False),
              "state_only": train(state_only, data, batches, state_only=True)}
    actual = contrasts(test["audit"]["values"]["object_x_mm"],
                       test["audit"]["assignment"], test["audit"]["steps"])
    metrics = {"sequence_action": factual(action, test, state_only=False),
               "state_only": factual(state_only, test, state_only=True)}
    predicted = {"sequence_action": candidate_effect(action, test, state_only=False),
                 "state_only": candidate_effect(state_only, test, state_only=True)}
    improvement = 1 - metrics["sequence_action"]["world_x_h10_rmse_mm"] / (
        metrics["state_only"]["world_x_h10_rmse_mm"])
    ratio = predicted["sequence_action"]["interaction_x_mm"] / (
        actual["interaction"] if abs(actual["interaction"]) > 1e-8 else float("nan"))
    if not np.isfinite([improvement, ratio]).all():
        raise FloatingPointError("nonfinite structured sequence Cm gate")
    report = {"schema": "ref2dex.structured_two_step_cm_probe.v1",
              "run_status": "COMPLETED", "train_samples": len(data["raw"]),
              "test_samples": len(test["raw"]),
              "parameter_count": sum(parameter.numel() for parameter in action.parameters()),
              "final_train_loss": losses, "heldout_factual": metrics,
              "observed_heldout_effect": actual, "predicted_heldout_effect": predicted,
              "gate_components": {"x_rmse_improvement_fraction": improvement,
                                  "predicted_over_observed_interaction": ratio},
              "elapsed_seconds": time.monotonic() - started,
              "limits": "held-out sequence effect prediction, not policy utility"}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, model in (("sequence_action", action), ("state_only", state_only)):
        torch.save({"schema": "ref2dex.structured_two_step_cm.v1", "name": name,
                    "model": model.state_dict()}, args.output / f"{name}.pt")
    manifest.update(run_status="COMPLETED", final_step=args.steps,
                    report_sha256=sha256(report_path),
                    checkpoint_sha256={name: sha256(args.output / f"{name}.pt")
                                       for name in ("sequence_action", "state_only")},
                    elapsed_seconds=time.monotonic() - started)
    print(json.dumps({"run_status": "COMPLETED", "gate_components": report["gate_components"],
                      "observed_interaction_x_mm": actual["interaction"]},
                     sort_keys=True), flush=True)


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
            len({args.train[0][0], args.train[1][0], args.test[0]}) != 3 or
            not 1 <= args.steps <= 1000):
        parser.error("new output, two train inputs, disjoint test and bounded steps required")
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "experiment_id": "P-20260924-cm-two-step-structured-model",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "train_input_sha256": {str(path.resolve()): digest for path, digest in args.train},
                "test_input_sha256": {str(args.test[0].resolve()): args.test[1]},
                "steps": args.steps, "batch_size": 96, "cpu_threads": 2,
                "gpu_count": 0, "wall_budget_minutes": 10, "output_budget_mb": 100,
                "stop_rule": "source drift, leakage, nonfinite output or budget"}
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
