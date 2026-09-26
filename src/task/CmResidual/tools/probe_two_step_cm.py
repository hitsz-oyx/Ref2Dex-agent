"""Train sequence-conditioned raw Cm against same-capacity state-only control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.task.CmResidual.cm_ppo_auxiliary import raw_cm_input
from src.task.CmResidual.cmlite import local_to_world_translation, local_translation_target
from src.task.CmResidual.contact_aware_cm import RawContactAwareCm
from src.task.CmResidual.tools.analyze_two_step_sequence import contrasts, load_run, sha256


ROOT = Path(__file__).resolve().parents[4]


def load_data(path: Path, digest: str) -> dict:
    if sha256(path) != digest:
        raise ValueError("sequence data SHA mismatch")
    audit = load_run(path, 64)
    source = torch.load(path, map_location="cpu", weights_only=False)["records"]
    selected = source["assignment"] != 0
    q = source["q"][selected].float()
    vel = source["dof_vel"][selected].float()
    obj = source["object_state"][selected].float()
    base = source["base_action"][selected].float()
    assignment = source["assignment"][selected].reshape(-1)
    first = assignment.sign().float() * .1
    second = first * (assignment.abs() == 2).float()
    raw_base = raw_cm_input(q, vel, obj, base)
    raw = torch.cat((raw_base, first[:, None], second[:, None]), dim=-1)
    result = {"raw": raw, "raw_base": raw_base, "object_state": obj,
              "step": local_translation_target(obj,
                                               source["next_object_state"][selected].float()),
              "followup": local_translation_target(
                  obj, source["followup_object_state"][selected].float()),
              "contact": source["followup_contact_count"][selected].float() / 10,
              "assignment": assignment, "global_step": source["global_step"][selected],
              "audit": audit}
    if raw.shape[1] != 69 or not all(torch.isfinite(value.float()).all()
                                     for key, value in result.items() if torch.is_tensor(value)):
        raise ValueError("nonfinite or invalid sequence model input")
    return result


def no_sequence(raw: torch.Tensor) -> torch.Tensor:
    result = raw.clone()
    result[:, -2:] = 0
    return result


def schedule(assignment: torch.Tensor, steps: int) -> list[torch.Tensor]:
    groups = [torch.where(assignment == code)[0] for code in (-2, -1, 1, 2)]
    if min(len(group) for group in groups) < 50:
        raise ValueError("insufficient four-arm training data")
    generator = torch.Generator().manual_seed(241141)
    return [torch.cat([group[torch.randint(len(group), (24,), generator=generator)]
                       for group in groups]) for _ in range(steps)]


def train(model: RawContactAwareCm, data: dict, batches: list[torch.Tensor],
          *, state_only: bool) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    raw = no_sequence(data["raw"]) if state_only else data["raw"]
    for ids in batches:
        model.train()
        output = model(raw[ids])
        one = F.smooth_l1_loss(output["delta_local"] / .01,
                               data["step"][ids] / .01, beta=.5)
        future = F.smooth_l1_loss(output["followup_delta_local"] / .02,
                                  data["followup"][ids] / .02, beta=.5)
        contact = F.mse_loss(output["contact_fraction"], data["contact"][ids])
        loss = .5 * one + .5 * future + 2 * contact
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite sequence Cm loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return float(loss.detach())


@torch.no_grad()
def factual(model: RawContactAwareCm, data: dict, *, state_only: bool) -> dict:
    raw = no_sequence(data["raw"]) if state_only else data["raw"]
    prediction = model(raw)
    world = local_to_world_translation(data["object_state"],
                                       prediction["followup_delta_local"])
    actual = local_to_world_translation(data["object_state"], data["followup"])
    return {"world_x_h10_rmse_mm": float((world[:, 0] - actual[:, 0]).square().mean().sqrt() * 1000),
            "world_translation_epe_mm": float((world - actual).norm(dim=-1).mean() * 1000),
            "contact_rmse": float((prediction["contact_fraction"] - data["contact"])
                                  .square().mean().sqrt())}


@torch.no_grad()
def candidate_effect(model: RawContactAwareCm, data: dict, *, state_only: bool) -> dict:
    values = {}
    for code in (-2, -1, 1, 2):
        first = np.sign(code) * .1
        second = first if abs(code) == 2 else 0
        doses = torch.tensor([first, second], dtype=data["raw_base"].dtype)
        raw = torch.cat((data["raw_base"], doses.expand(len(data["raw_base"]), -1)), dim=-1)
        if state_only:
            raw = no_sequence(raw)
        output = model(raw)
        world = local_to_world_translation(data["object_state"],
                                           output["followup_delta_local"])
        values[code] = world[:, 0].numpy() * 1000
    short = float(np.mean(values[1] - values[-1]))
    long = float(np.mean(values[2] - values[-2]))
    return {"one_effect_x_mm": short, "two_effect_x_mm": long,
            "interaction_x_mm": long - short}


def execute(args, manifest: dict) -> None:
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(241140)
    train_data = load_data(*args.train)
    test_data = load_data(*args.test)
    mean = train_data["raw"].mean(0)
    std = train_data["raw"].std(0).clamp_min(.01)
    action = RawContactAwareCm(mean, std)
    state_only = RawContactAwareCm(mean, std)
    state_only.load_state_dict(action.state_dict(), strict=True)
    batches = schedule(train_data["assignment"], args.steps)
    losses = {"sequence_action": train(action, train_data, batches, state_only=False),
              "state_only": train(state_only, train_data, batches, state_only=True)}
    actual = contrasts(test_data["audit"]["values"]["object_x_mm"],
                       test_data["audit"]["assignment"],
                       test_data["audit"]["steps"])
    metrics = {"sequence_action": factual(action, test_data, state_only=False),
               "state_only": factual(state_only, test_data, state_only=True)}
    predicted = {"sequence_action": candidate_effect(action, test_data, state_only=False),
                 "state_only": candidate_effect(state_only, test_data, state_only=True)}
    rmse_improvement = 1 - metrics["sequence_action"]["world_x_h10_rmse_mm"] / (
        metrics["state_only"]["world_x_h10_rmse_mm"])
    interaction_ratio = predicted["sequence_action"]["interaction_x_mm"] / (
        actual["interaction"] if abs(actual["interaction"]) > 1e-8 else float("nan"))
    report = {"schema": "ref2dex.two_step_cm_probe.v1",
              "run_status": "COMPLETED", "train_samples": len(train_data["raw"]),
              "test_samples": len(test_data["raw"]),
              "parameter_count": sum(p.numel() for p in action.parameters()),
              "final_train_loss": losses, "heldout_factual": metrics,
              "observed_heldout_effect": actual,
              "predicted_heldout_effect": predicted,
              "gate_components": {"x_rmse_improvement_fraction": rmse_improvement,
                                  "predicted_over_observed_interaction": interaction_ratio},
              "elapsed_seconds": time.monotonic() - started,
              "limits": "held-out sequence effect prediction, not policy utility"}
    if not np.isfinite([rmse_improvement, interaction_ratio]).all():
        raise FloatingPointError("nonfinite sequence Cm gate")
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, model in (("sequence_action", action), ("state_only", state_only)):
        torch.save({"schema": "ref2dex.two_step_cm.v1", "name": name,
                    "model": model.state_dict()}, args.output / f"{name}.pt")
    manifest.update(run_status="COMPLETED", final_step=args.steps,
                    report_sha256=sha256(report_path),
                    checkpoint_sha256={name: sha256(args.output / f"{name}.pt")
                                       for name in ("sequence_action", "state_only")},
                    elapsed_seconds=time.monotonic() - started)
    print(json.dumps({"run_status": "COMPLETED",
                      "gate_components": report["gate_components"],
                      "observed_interaction_x_mm": actual["interaction"]},
                     sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--test", nargs=2, required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    args.train = (Path(args.train[0]), args.train[1])
    args.test = (Path(args.test[0]), args.test[1])
    if (args.output.exists() or args.train[0] == args.test[0] or
            not 1 <= args.steps <= 1000):
        parser.error("new output, disjoint inputs and bounded steps required")
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "experiment_id": "P-20260924-cm-two-step-model",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "train_input_sha256": sha256(args.train[0]),
                "test_input_sha256": sha256(args.test[0]),
                "steps": args.steps, "batch_size": 96, "cpu_threads": 2,
                "gpu_count": 0, "wall_budget_minutes": 10,
                "output_budget_mb": 100,
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
