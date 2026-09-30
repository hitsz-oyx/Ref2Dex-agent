#!/usr/bin/env python3
"""One fixed MLP screen; never claims policy utility or tunes on holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.task.CmResidual.six_expert_support_adapter import validate_payload

SEED = 278
FIT_SHA = "aa60b82b35d95ea6278e8cf0c386d00b5fcf6ed5fc3ec31bc7a858c0f31dbf93"
HOLD_SHA = "b49e30045ba2370546cc0c6ea8949660fbcdc7984a584fe4d67113f59ddeef80"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path, expected):
    if sha(path) != expected:
        raise ValueError("support input hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    audit = validate_payload(payload)
    rows = payload["records"]
    ids = rows["assignment"].long()
    obs = rows["pre_action_observation"].float()
    actions = rows["candidate_actions"].float()
    n = len(ids)
    x = torch.cat([obs, actions[torch.arange(n), ids],
                   torch.nn.functional.one_hot(ids, 6).float()], dim=1)
    all_x = torch.cat([obs[:, None].expand(-1, 6, -1), actions,
                       torch.eye(6)[None].expand(n, -1, -1)], dim=2)
    delta = rows["target_delta_object_local_1"].float()
    contact = rows["contact_mask_t_plus_1_to_t_plus_5"].float().mean(dim=1)
    return dict(x=x, all_x=all_x, delta=delta, contact=contact,
                axis=rows["object_lift_axis"].float(),
                router=rows["router_teacher_candidate_id"].long(),
                episodes=list(rows["episode_id"]), audit=audit)


def quantile90(values):
    # Finite-sample rank, kept independent of all holdout outcomes.
    count = len(values)
    rank = min(count, int(__import__("math").ceil((count + 1) * .90)))
    return values.sort(dim=0).values[rank - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SEED)
    started = time.monotonic()
    fit = load(args.fit, FIT_SHA)
    hold = load(args.holdout, HOLD_SHA)
    if set(fit["episodes"]) & set(hold["episodes"]):
        raise ValueError("fit/holdout episodes overlap")
    if len(set(fit["episodes"])) != len(fit["episodes"]):
        raise ValueError("fit rows are not independent episodes")
    permutation = torch.randperm(len(fit["x"]))
    calibration_count = int(__import__("math").ceil(len(permutation) * .20))
    cal_ids, opt_ids = permutation[:calibration_count], permutation[calibration_count:]
    mean = fit["x"][opt_ids].mean(dim=0)
    std = fit["x"][opt_ids].std(dim=0).clamp_min(1e-3)
    dmean = fit["delta"][opt_ids].mean(dim=0)
    dstd = fit["delta"][opt_ids].std(dim=0).clamp_min(1e-4)
    model = torch.nn.Sequential(torch.nn.Linear(mean.numel(), 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                torch.nn.Linear(128, 6))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = (fit["x"][opt_ids] - mean) / std
    target = (fit["delta"][opt_ids] - dmean) / dstd
    quantiles = torch.tensor([.1, .5, .9])
    losses = []
    for step in range(1000):
        if time.monotonic() - started > 900:
            raise TimeoutError("15-minute budget")
        pred = model(x)
        dloss = torch.nn.functional.smooth_l1_loss(pred[:, :3], target)
        error = fit["contact"][opt_ids, None] - pred[:, 3:]
        closs = torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
        loss = dloss + closs
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 999:
            losses.append(dict(step=step + 1, loss=float(loss.detach())))

    @torch.no_grad()
    def predict(values):
        prediction = model((values - mean) / std)
        return prediction[..., :3] * dstd + dmean, prediction[..., 3:].sort(dim=-1).values.clamp(0, 1)

    model.eval()
    cdelta, cq = predict(fit["x"][cal_ids])
    dradius = quantile90((cdelta - fit["delta"][cal_ids]).abs())
    correction = quantile90(cq[:, 0] - fit["contact"][cal_ids]).clamp_min(0)

    def score(data):
        delta, q = predict(data["x"])
        all_delta, all_q = predict(data["all_x"])
        lower = (all_q[..., 0] - correction).clamp_min(0)
        # A single common radius does not change relative progress ordering.
        progress = (all_delta * data["axis"][:, None]).sum(dim=-1) - (dradius * data["axis"].abs()).sum(dim=-1)[:, None]
        labels = {}
        fallbacks = {}
        for threshold in (.55, .60, .65):
            eligible = lower >= threshold
            candidate = progress.masked_fill(~eligible, -torch.inf).argmax(dim=1)
            fallback = ~eligible.any(dim=1)
            labels[threshold] = torch.where(fallback, data["router"], candidate)
            fallbacks[threshold] = fallback
        base = labels[.60]
        stability = min(float((labels[t] == base).float().mean()) for t in labels)
        result = dict(rows=len(delta), delta_rmse=float((delta - data["delta"]).square().mean().sqrt()),
                      contact_q10_lower_coverage=float((data["contact"] >= (q[:, 0] - correction).clamp_min(0)).float().mean()),
                      delta_interval_coordinate_coverage=float(((delta - data["delta"]).abs() <= dradius).float().mean()),
                      fallback_rate=float(fallbacks[.60].float().mean()),
                      changed_vs_router=float((base != data["router"]).float().mean()),
                      threshold_stability=stability, cm_on_labels=base.tolist(),
                      cm_off_labels=data["router"].tolist())
        return result

    fit_scores, hold_scores = score(fit), score(hold)
    gates = dict(contact_coverage=hold_scores["contact_q10_lower_coverage"] >= .9,
                 delta_coverage=hold_scores["delta_interval_coordinate_coverage"] >= .8,
                 fallback=hold_scores["fallback_rate"] <= .5,
                 stability=hold_scores["threshold_stability"] >= .9)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "model.pt"
    torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std,
                    delta_mean=dmean, delta_std=dstd, delta_radius=dradius,
                    contact_correction=correction, seed=SEED,
                    fit_sha256=FIT_SHA, holdout_sha256=HOLD_SHA), checkpoint)
    result = dict(schema="ref2dex.scratch_mlp_screen.v1", experiment_id="P-20260930-scratch-mlp-feasibility",
                  run_id="cpu_s278_r1", run_status="COMPLETED", gates=gates,
                  conclusion="PROMISING" if all(gates.values()) else "UNPROMISING",
                  scope="estimator feasibility only; no physical policy comparison",
                  prior_holdout_inspected=True, steps=1000, seed=SEED, cpu_threads=2,
                  optimization_rows=len(opt_ids), calibration_rows=len(cal_ids),
                  calibration_episode_ids=[fit["episodes"][i] for i in cal_ids.tolist()],
                  delta_radius=dradius.tolist(), contact_correction=float(correction),
                  training_losses=losses, fit=fit_scores, holdout=hold_scores,
                  inputs=dict(fit_path=str(args.fit), holdout_path=str(args.holdout),
                              fit_sha256=FIT_SHA, holdout_sha256=HOLD_SHA),
                  audits=dict(fit=fit["audit"], holdout=hold["audit"]),
                  checkpoint_sha256=sha(checkpoint), source_sha256=sha(Path(__file__)),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  wall_seconds=time.monotonic() - started)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("run_status", "conclusion", "gates", "wall_seconds")}))


if __name__ == "__main__":
    main()
