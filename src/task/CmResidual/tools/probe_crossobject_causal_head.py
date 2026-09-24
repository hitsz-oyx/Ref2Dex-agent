"""Leave-one-train-object-out probe of a signed action-effect Cm head."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.probe_contact_aware_cm import ranked_effect
from src.task.CmResidual.tools.probe_expert_crossobject_cm import (
    ROOT, TRAIN, calibrate, geometric_inputs, load_rows,
)
from src.task.CmResidual.tools.probe_randomized_geometric_cm import raw_inputs
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


def subset(rows: dict, ids: list[int]) -> tuple[dict, list[str]]:
    selected = torch.isin(rows["motion_id"], torch.tensor(ids))
    part = {key: value[selected].clone() for key, value in rows.items()}
    mapping = {old: new for new, old in enumerate(ids)}
    part["motion_id"] = torch.tensor([mapping[int(i)] for i in part["motion_id"]])
    names = [["airplane", "mug", "toothpaste"][i] for i in ids]
    return part, names


def inputs(rows: dict, names: list[str], setup: dict, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    arms = []
    for sign in (1, -1):
        candidate = dict(rows)
        candidate["action"] = rows["base_action"].clone()
        candidate["action"][:, 2] = (candidate["action"][:, 2] + sign * .1).clamp(-1, 1)
        if kind == "geometric":
            feature = geometric_inputs(candidate, names, setup)
            flat = torch.cat((feature["region"].flatten(1), feature["context"]), 1)
        else:
            flat = raw_inputs(candidate)
        arms.append(flat)
    return (arms[0] + arms[1]) / 2, arms[0] - arms[1]


class CausalHead(nn.Module):
    def __init__(self, state_mean: torch.Tensor, state_std: torch.Tensor,
                 diff_mean: torch.Tensor, diff_std: torch.Tensor, width: int = 48):
        super().__init__()
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std.clamp_min(.1))
        self.register_buffer("diff_mean", diff_mean)
        self.register_buffer("diff_std", diff_std.clamp_min(.1))
        sdim, ddim = len(state_mean), len(diff_mean)
        self.baseline = nn.Sequential(nn.Linear(sdim, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU(),
                                      nn.Linear(width, 1))
        self.effect = nn.Sequential(nn.Linear(sdim + ddim, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, 1))
        nn.init.zeros_(self.baseline[-1].weight)
        nn.init.zeros_(self.baseline[-1].bias)
        nn.init.zeros_(self.effect[-1].weight)
        nn.init.zeros_(self.effect[-1].bias)

    def forward(self, state: torch.Tensor, difference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = (state - self.state_mean) / self.state_std
        d = (difference - self.diff_mean) / self.diff_std
        return self.baseline(s).squeeze(-1) * 20, self.effect(torch.cat((s, d), 1)).squeeze(-1) * 20


def outcome(rows: dict) -> torch.Tensor:
    return ((rows["followup_object_state"][:, 2] - rows["object_state"][:, 2]) *
            rows["contact_fraction"] * 1000)


def effect_estimate(rows: dict) -> float:
    return weighted_step_difference(
        outcome(rows).numpy(), rows["assignment"].numpy(),
        (rows["global_step"] + rows["motion_id"] * 1000).numpy())[0]


def fit(train: dict, state: torch.Tensor, difference: torch.Tensor,
        *, seed: int, steps: int) -> CausalHead:
    torch.manual_seed(seed)
    model = CausalHead(state.mean(0), state.std(0),
                       difference.mean(0), difference.std(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    y = outcome(train)
    a = train["assignment"].float()
    for _ in range(steps):
        ids = torch.randint(len(y), (96,), generator=generator)
        base, effect = model(state[ids], difference[ids])
        predicted = base + .5 * a[ids] * effect
        loss = F.smooth_l1_loss(predicted / 20, y[ids] / 20, beta=.5)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite causal Cm fit")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: CausalHead, val: dict, state: torch.Tensor,
             difference: torch.Tensor, seed: int) -> dict:
    base, effect = model(state, difference)
    y = outcome(val)
    a = val["assignment"].float()
    score = effect.numpy()
    report = {"actual_ate_mm": effect_estimate(val),
              "predicted_ate_mm": float(effect.mean()),
              "predicted_effect_std_mm": float(effect.std()),
              "factual_mae_mm": float((base + .5 * a * effect - y).abs().mean())}
    if len(score) >= 320 and float(np.quantile(score, .75) - np.quantile(score, .25)) >= 1e-6:
        report["ranked_uplift"] = ranked_effect(
            score, y.numpy(), a.numpy(), val["global_step"].numpy(),
            val["env_id"].numpy(), bootstraps=300, seed=seed)
    else:
        report["ranked_uplift"] = {"status": "SPARSE_QUARTILES_OR_NO_SCORE_SPREAD"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 500:
        parser.error("new output and 1..500 bounded steps required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_sha256": sha256(TRAIN / "transitions.pt"),
                "folds": ["airplane", "mug", "toothpaste"],
                "steps": args.steps, "gpu_count": 0, "cpu_threads": 2,
                "wall_budget_minutes": 45, "output_budget_mb": 10,
                "stop_rule": "source drift, leakage, non-finite or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        all_rows, source = load_rows(TRAIN, "train")
        if source["source_objects"] != ["airplane", "mug", "toothpaste"]:
            raise ValueError("train-object order drift")
        results = {}
        for held_id, held_name in enumerate(source["source_objects"]):
            train_ids = [i for i in range(3) if i != held_id]
            train, train_names = subset(all_rows, train_ids)
            val, val_names = subset(all_rows, [held_id])
            setup = calibrate(train)
            model_results = {}
            for kind in ("geometric", "raw"):
                train_state, train_difference = inputs(train, train_names, setup, kind)
                val_state, val_difference = inputs(val, val_names, setup, kind)
                model = fit(train, train_state, train_difference,
                            seed=240940 + held_id, steps=args.steps)
                model_results[kind] = evaluate(
                    model, val, val_state, val_difference,
                    seed=240950 + held_id)
            constant = effect_estimate(train)
            actual = effect_estimate(val)
            model_results["constant"] = {"predicted_ate_mm": constant,
                                         "actual_ate_mm": actual}
            results[held_name] = model_results
            print(json.dumps({held_name: model_results}, sort_keys=True), flush=True)
        object_mae = {
            kind: float(np.mean([abs(results[name][kind]["predicted_ate_mm"] -
                                     results[name][kind]["actual_ate_mm"])
                                 for name in results]))
            for kind in ("geometric", "raw", "constant")}
        positive = sum(results[name]["geometric"]["predicted_ate_mm"] > 0
                       for name in results)
        gate = positive >= 2 and object_mae["geometric"] < min(
            object_mae["raw"], object_mae["constant"])
        report = {"schema": "ref2dex.crossobject_causal_head_loo.v1",
                  "source_actor_role": "official_data_collector",
                  "folds": results, "object_mae_mm": object_mae,
                  "positive_geometric_objects": positive,
                  "continue_to_new_apple_seed": gate,
                  "elapsed_seconds": time.monotonic() - started}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output / "report.json"))
        print(json.dumps({"object_mae_mm": object_mae, "positive": positive,
                          "continue_to_new_apple_seed": gate}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
