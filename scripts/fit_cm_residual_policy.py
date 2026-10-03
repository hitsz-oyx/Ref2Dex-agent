#!/usr/bin/env python3
"""Fit Cm consequence ensembles and bounded residual actors from native source rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task.CmResidual.cm_residual_policy import (
    CONTEXT_DIM, HISTORY_DIM, RESIDUAL_SCALE_M, SCHEMA,
    ResidualActor, ResidualConsequenceModel, consequence_score,
    world_height_score_mm,
    context_features, residual_scale, two_step_targets,
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(paths: list[Path], device: torch.device) -> dict[str, torch.Tensor]:
    payloads = []
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("schema") != "ref2dex.cm_residual_source.v2":
            raise ValueError(f"source schema mismatch: {path}")
        if (record.get("seed") is None or record.get("simulator_seed") is None
                or int(record["seed"]) != int(record["simulator_seed"])):
            raise ValueError(f"source simulator seed provenance mismatch: {path}")
        if record["future_done"].any() or record["residual_saturated"].any():
            raise ValueError(f"source contains incomplete or saturated rows: {path}")
        payloads.append(record)
    if not payloads:
        raise ValueError("at least one source records file is required")
    keys = ("state", "history", "residual", "residual_id", "future_state", "future_contact",
            "future_clearance", "rest_z", "initial_hand_force", "initial_object_force",
            "initial_clearance", "mass_kg", "split_group_bucket", "motion_id", "start_frame")
    data = {key: torch.cat([p[key] for p in payloads], 0) for key in keys}
    gravity = float(payloads[0]["gravity_magnitude"])
    if any(abs(float(p["gravity_magnitude"]) - gravity) > 1e-8 for p in payloads):
        raise ValueError("source gravity mismatch")
    context = context_features(data["state"], data["initial_hand_force"], data["initial_object_force"],
                               data["mass_kg"], gravity, data["initial_clearance"], data["rest_z"])
    target = two_step_targets(data["state"], data["future_state"], data["future_contact"],
                              data["future_clearance"], data["rest_z"])
    result = {
        "history": data["history"].double(),
        "context": context.double(),
        "residual": data["residual"].double() / residual_scale("cpu", torch.float64),
        "target": target.double(),
        "object_quaternion": data["state"][:, 39:43].double(),
        "bucket": data["split_group_bucket"].long(),
        "motion_id": data["motion_id"].long(),
        "start_frame": data["start_frame"].long(),
    }
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("non-finite source feature or target")
    return {key: value.to(device=device, dtype=torch.float32 if value.dtype.is_floating_point else value.dtype)
            for key, value in result.items()}


def normalize(data: dict[str, torch.Tensor], fit: torch.Tensor) -> dict[str, torch.Tensor]:
    history_fit = data["history"][fit]
    context_fit = data["context"][fit]
    return {
        "history_mean": history_fit.mean((0, 1)),
        "history_std": history_fit.std((0, 1), unbiased=False).clamp_min(.001),
        "context_mean": context_fit.mean(0),
        "context_std": context_fit.std(0, unbiased=False).clamp_min(.001),
    }


def scaled(data: dict[str, torch.Tensor], norm: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    history = ((data["history"] - norm["history_mean"]) / norm["history_std"]).clamp(-8, 8)
    context = ((data["context"] - norm["context_mean"]) / norm["context_std"]).clamp(-8, 8)
    return history, context


def train_model(history, context, residual, target, ids, seed: int, updates: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ResidualConsequenceModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 30000)
    for _ in range(updates):
        pick = ids[torch.randint(len(ids), (min(256, len(ids)),), generator=generator, device=device)]
        optimizer.zero_grad(set_to_none=True)
        output = model(history[pick], context[pick], residual[pick])
        loss = F.smooth_l1_loss(output[:, :3], target[pick, :3]) + F.binary_cross_entropy_with_logits(
            output[:, 3:], target[pick, 3:])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Cm loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval().requires_grad_(False)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, float(loss)


@torch.no_grad()
def ensemble_predict(states, history, context, residuals, device):
    values = []
    for state in states:
        model = ResidualConsequenceModel().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        values.append(model(history, context, residuals))
    return torch.stack(values)


@torch.no_grad()
def candidate_labels(states, history, context, object_quaternion, device,
                     uncertainty_limit: float | None = None):
    axis = torch.tensor((-1., 0., 1.), device=device)
    grid = torch.cartesian_prod(axis, axis, axis).reshape(-1, 3)
    n = len(history)
    candidates = grid[None].expand(n, -1, -1)
    h = history[:, None].expand(-1, len(grid), -1, -1).reshape(-1, history.shape[1], history.shape[2])
    c = context[:, None].expand(-1, len(grid), -1).reshape(-1, context.shape[-1])
    r = candidates.reshape(-1, 3)
    outputs = ensemble_predict(states, h, c, r, device).reshape(len(states), n, len(grid), -1)
    quaternion = object_quaternion[None, :, :].expand(len(states), -1, -1).reshape(-1, 4)
    score, loss, clearance = consequence_score(outputs.reshape(-1, outputs.shape[-1]), quaternion)
    score = score.reshape(len(states), n, len(grid))
    loss = loss.reshape(len(states), n, len(grid))
    clearance = clearance.reshape(len(states), n, len(grid))
    zero_index = ((grid == 0).all(-1)).nonzero().item()
    zero_score = score[:, :, zero_index].mean(0)
    zero_loss = loss[:, :, zero_index].mean(0)
    zero_clearance = clearance[:, :, zero_index].mean(0)
    mean_score = score.mean(0)
    std_score = score.std(0, unbiased=False)
    safe = (loss.mean(0) <= zero_loss[:, None] + .02) & (clearance.mean(0) <= zero_clearance[:, None] + .02)
    if uncertainty_limit is not None:
        safe &= std_score <= uncertainty_limit
    safe &= mean_score > zero_score[:, None]
    safe[:, zero_index] = True
    value = mean_score.masked_fill(~safe, -torch.inf)
    selected = value.argmax(-1)
    selected_value = value.gather(-1, selected[:, None]).squeeze(-1)
    labels = grid[selected]
    labels[selected_value <= zero_score] = 0
    return labels, dict(grid=grid, zero_score=zero_score, score=mean_score,
                        score_std=std_score, contact_loss=loss.mean(0),
                        clearance_loss=clearance.mean(0), selected=selected)


def train_actor(history, context, labels, ids, seed: int, updates: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    actor = ResidualActor().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=5e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 40000)
    for _ in range(updates):
        pick = ids[torch.randint(len(ids), (min(256, len(ids)),), generator=generator, device=device)]
        optimizer.zero_grad(set_to_none=True)
        prediction = actor(history[pick], context[pick])
        loss = F.smooth_l1_loss(prediction, labels[pick])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite residual actor loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
    actor.eval().requires_grad_(False)
    return {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}, float(loss)


def fit(args):
    begin = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but it is unavailable")
    paths = [path.resolve() for path in args.records]
    data = load_rows(paths, device)
    fit_mask = data["bucket"] < 50
    cal_mask = (data["bucket"] >= 50) & (data["bucket"] < 70)
    held_mask = data["bucket"] >= 70
    if int(fit_mask.sum()) < 128 or int(cal_mask.sum()) < 64 or int(held_mask.sum()) < 64:
        raise ValueError("fit/cal/held support is insufficient")
    norm = normalize(data, fit_mask)
    history, context = scaled(data, norm)
    fit_ids = fit_mask.nonzero().flatten()
    cal_ids = cal_mask.nonzero().flatten()
    target = data["target"]
    residual = data["residual"]
    model_states, losses, actor_states, actor_losses = {}, {}, {}, {}
    label_reports = {}
    for mode, model_residual in (("cm", residual),):
        states = []
        losses[mode] = []
        for seed in (17681, 17682, 17683):
            state, loss = train_model(history, context, model_residual, target, fit_ids, seed, args.model_updates, device)
            states.append(state); losses[mode].append(loss)
        model_states[mode] = states
        labels, report = candidate_labels(
            states, history[fit_ids], context[fit_ids], data["object_quaternion"][fit_ids], device)
        fit_history, fit_context = history[fit_ids], context[fit_ids]
        actor_ids = torch.arange(len(fit_ids), device=device)
        actor_state, actor_loss = train_actor(
            fit_history, fit_context, labels, actor_ids, 18681, args.actor_updates, device)
        actor_states[mode] = actor_state; actor_losses[mode] = actor_loss
        label_reports[mode] = dict(nonzero=int((labels.abs().sum(-1) > 0).sum()),
                                   rows=len(labels), grid_size=int(len(report["grid"])))
    margins, uncertainty, calibration = {}, {}, {}
    for mode in ("cm",):
        states = model_states[mode]
        zero = torch.zeros(len(cal_ids), 3, device=device)
        h, c, r = history[cal_ids], context[cal_ids], residual[cal_ids]
        out = ensemble_predict(states, h, c, r, device)
        quaternion = data["object_quaternion"][cal_ids][None].expand(len(states), -1, -1).reshape(-1, 4)
        score, loss, clearance = consequence_score(out.flatten(0, 1), quaternion)
        score = score.reshape(len(states), -1); loss = loss.reshape(len(states), -1); clearance = clearance.reshape(len(states), -1)
        zero_out = ensemble_predict(states, h, c, zero, device)
        zscore, zloss, zclearance = consequence_score(zero_out.flatten(0, 1), quaternion)
        zscore = zscore.reshape(len(states), -1); zloss = zloss.reshape(len(states), -1); zclearance = zclearance.reshape(len(states), -1)
        target_height = world_height_score_mm(data["object_quaternion"][cal_ids], target[cal_ids, :3])
        target_score = target_height * target[cal_ids, 3] * (1 - target[cal_ids, 5])
        error = (score.mean(0) - target_score).abs()
        margins[mode] = max(.5, float(torch.quantile(error, .75)))
        uncertainty[mode] = max(.25, float(torch.quantile(score.std(0, unbiased=False), .9)))
        calibration[mode] = dict(rows=len(cal_ids), score_mae_mm=float(error.mean()),
                                 contact_brier=float((loss.mean(0) - target[cal_ids, 4]).square().mean()),
                                 clearance_brier=float((clearance.mean(0) - target[cal_ids, 5]).square().mean()),
                                 zero_score_mm=float(zscore.mean()),
                                 residual_score_mm=float(score.mean()))
    payload = dict(schema=SCHEMA, experiment_id="P-20261003-cm-residual-policy-corrected",
                   model_device=str(device), residual_scale_m=RESIDUAL_SCALE_M,
                   history_mean=norm["history_mean"].cpu(), history_std=norm["history_std"].cpu(),
                   context_mean=norm["context_mean"].cpu(), context_std=norm["context_std"].cpu(),
                   models=model_states, actors=actor_states, margins_mm=margins,
                   uncertainty_mm=uncertainty, fit_rows=int(fit_mask.sum()),
                   cal_rows=int(cal_mask.sum()), held_rows=int(held_mask.sum()),
                   model_losses=losses, actor_losses=actor_losses,
                   label_reports=label_reports, calibration=calibration,
                   source_sha256={str(path): sha(path) for path in paths},
                   git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("refusing to overwrite policy checkpoint")
    torch.save(payload, args.output)
    result = dict(run_status="COMPLETED", schema=SCHEMA, fit_rows=int(fit_mask.sum()),
                  cal_rows=int(cal_mask.sum()), held_rows=int(held_mask.sum()),
                  label_reports=label_reports, calibration=calibration,
                  checkpoint_sha256=sha(args.output), elapsed_seconds=time.monotonic() - begin,
                  source_sha256=payload["source_sha256"])
    result_path = args.output.with_suffix(".results.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-updates", type=int, default=500)
    parser.add_argument("--actor-updates", type=int, default=400)
    fit(parser.parse_args())
