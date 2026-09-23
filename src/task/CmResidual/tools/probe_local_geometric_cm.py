"""Train a compact six-region geometric Cm on executed DExplore transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from src.task.CmResidual.cmlite import local_translation_target
from src.task.CmResidual.dexplore_cm_geometry import (
    dexplore_action_to_native_targets, dexplore_root_pose, native_joint_limits,
    world_to_object_frame,
)
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS, TRANSITIONS
from src.task.CmResidual.tools.probe_mixed_cm_sim_adapt import TRAIN
from src.task.CmResidual.tools.probe_mixed_cm_sim_transfer import sha256
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
CALIBRATION = ROOT / "outputs/CmResidual/agent_executed_handflow_64/report.json"
CALIBRATION_SHA256 = "76bf65d450e3cfdcb7c0a9f54d68d7324e569fab18e2b7392224bd363ede0b16"
NUM_ENVS = 64
NUM_REGIONS = 6
FEATURE_DIM = 19
REGION_FOR_LINK = torch.tensor([0] + [1] * 5 + [2] * 3 + [3] * 3 + [4] * 3 + [5] * 3)


def _surface_geometry_class():
    from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
    return _surface_geometry_class()


class SixRegionCm(nn.Module):
    """Object-local fingertip/hand interaction tokens with a compact effect head."""

    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.region_embedding = nn.Parameter(torch.zeros(NUM_REGIONS, width))
        nn.init.normal_(self.region_embedding, std=width ** -0.5)
        self.input = nn.Sequential(nn.Linear(FEATURE_DIM, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width * 2 + 6, width * 2), nn.SiLU(),
                                  nn.Linear(width * 2, width), nn.SiLU())
        self.translation = nn.Linear(width, 3)
        self.contact = nn.Linear(width, 1)
        nn.init.zeros_(self.translation.weight)
        nn.init.zeros_(self.translation.bias)

    def forward(self, region_features, object_velocity):
        if region_features.ndim != 3 or region_features.shape[1:] != (NUM_REGIONS, FEATURE_DIM):
            raise ValueError("region_features must be [B,6,19]")
        if object_velocity.shape != (len(region_features), 6):
            raise ValueError("object_velocity must be [B,6]")
        encoded = self.input(region_features) + self.region_embedding[None]
        attended = self.attention(encoded, encoded, encoded, need_weights=False)[0]
        tokens = self.norm(encoded + attended)
        pooled = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1), object_velocity), dim=-1)
        hidden = self.head(pooled)
        return {"cm_tokens": tokens,
                "delta_local": self.translation(hidden) * .01,
                "contact_logit": self.contact(hidden).squeeze(-1)}


def select_rows(path, digest, seed, each, lower, upper):
    if sha256(path) != digest:
        raise ValueError(f"transition SHA drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError("transition schema mismatch")
    mask = first_episode_mask(payload["done"], NUM_ENVS)
    mask[:NUM_ENVS] = False
    mask &= ~torch.roll(payload["done"].reshape(-1).bool(), NUM_ENVS)
    rows = {key: value[mask] for key, value in payload.items()
            if isinstance(value, torch.Tensor)}
    rows["q_prev"] = torch.roll(payload["q"], NUM_ENVS, dims=0)[mask]
    contact = (rows["hand_contact"].bool() & rows["object_contact"].bool()).reshape(-1)
    motion = local_translation_target(rows["object_state"], rows["next_object_state"]).norm(dim=-1)
    categories = (contact & (motion > .002), contact & (motion <= .002), ~contact)
    generator = torch.Generator().manual_seed(2440 + seed)
    indices = []
    for category in categories:
        choices = torch.where(category)[0]
        if len(choices) < each:
            raise ValueError(f"seed {seed} lacks {each} category samples")
        indices.append(choices[torch.randperm(len(choices), generator=generator)[:each]])
    selected = torch.cat(indices)
    result = {key: value[selected] for key, value in rows.items()}
    result["category"] = torch.arange(3).repeat_interleave(each)
    result["target"] = local_translation_target(
        result["object_state"], result["next_object_state"])
    result["pd_target"] = dexplore_action_to_native_targets(
        result["action"], result["q"], lower, upper)
    return result


@torch.no_grad()
def extract_features(rows, *, kinematics, geometry, lower, upper, calibration,
                     batch_size, shuffle_actions=False):
    region_ids = REGION_FOR_LINK[torch.as_tensor(geometry.hand_link, dtype=torch.long)]
    counts = torch.stack([(region_ids == region).sum() for region in range(NUM_REGIONS)])
    if (counts == 0).any():
        raise ValueError(f"six-region surface sampling is incomplete: {counts.tolist()}")
    shuffled = rows["action"].roll(1, dims=0)
    features, contexts = [], []
    for start in range(0, len(rows["q"]), batch_size):
        stop = start + batch_size
        q = rows["q"][start:stop]
        action = shuffled[start:stop] if shuffle_actions else rows["action"][start:stop]
        current_pose = dexplore_root_pose(rows["object_state"][start:stop])
        current_links = kinematics.forward(q[:, None])[:, 0]
        current_hand, current_normals = geometry.hand(current_links)
        pd_target = dexplore_action_to_native_targets(action, q, lower, upper)
        predicted_q = (q + calibration["alpha"] * (pd_target - q)
                       + calibration["beta"] * (q - rows["q_prev"][start:stop]))
        next_links = kinematics.forward(predicted_q[:, None])[:, 0]
        next_hand, _ = geometry.hand(next_links)
        hand = world_to_object_frame(current_hand, current_pose, vector=False)
        hand_normals = world_to_object_frame(current_normals, current_pose, vector=True)
        flow = world_to_object_frame(next_hand - current_hand, current_pose, vector=True)
        obj_world, obj_normal_world = geometry.object(current_pose)
        obj = world_to_object_frame(obj_world, current_pose, vector=False)
        obj_normals = world_to_object_frame(obj_normal_world, current_pose, vector=True)
        distances = torch.cdist(hand, obj)
        min_dist, nearest = distances.min(dim=-1)
        gather = nearest[..., None].expand(-1, -1, 3)
        nearest_obj = torch.gather(obj, 1, gather)
        nearest_normal = torch.gather(obj_normals, 1, gather)
        relative = hand - nearest_obj
        direction = relative / min_dist[..., None].clamp_min(1e-5)
        toward = -(flow * direction).sum(dim=-1)
        proximity_weight = torch.exp(-min_dist / .02)
        region_values = []
        for region in range(NUM_REGIONS):
            region_mask = region_ids == region
            region_distance = min_dist[:, region_mask]
            weights = proximity_weight[:, region_mask]
            weights /= weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            presence = torch.exp(-region_distance.amin(dim=-1) / .02)
            def aggregate(value):
                return (weights[..., None] * value[:, region_mask]).sum(dim=1) * presence[:, None]
            rel = aggregate(relative) / .05
            pos = aggregate(nearest_obj) / .1
            normal = aggregate(nearest_normal)
            hand_normal = aggregate(hand_normals)
            flow_mean = aggregate(flow) / .02
            toward_mean = (weights * toward[:, region_mask]).sum(dim=-1) * presence / .02
            flow_rms = torch.sqrt((weights * flow[:, region_mask].square().sum(-1)).sum(-1) + 1e-12)
            flow_rms = flow_rms * presence / .02
            region_feature = torch.cat((
                (region_distance.amin(dim=-1) / .05).clamp(max=4)[:, None],
                (region_distance < .02).float().mean(dim=-1, keepdim=True),
                rel, pos, normal, flow_mean, toward_mean[:, None],
                flow_rms[:, None], hand_normal), dim=-1)
            if region_feature.shape[1] != FEATURE_DIM:
                raise RuntimeError("six-region feature layout changed")
            region_values.append(region_feature)
        features.append(torch.stack(region_values, dim=1))
        velocity_world = rows["object_state"][start:stop, 7:13]
        local_linear = world_to_object_frame(velocity_world[:, None, :3], current_pose, vector=True)[:, 0]
        local_angular = world_to_object_frame(velocity_world[:, None, 3:], current_pose, vector=True)[:, 0]
        contexts.append(torch.cat((local_linear / .1, local_angular), dim=-1))
    result = {"region": torch.cat(features), "context": torch.cat(contexts),
              "target": rows["target"].float(), "category": rows["category"],
              "contact": (rows["category"] < 2).float()}
    if not all(torch.isfinite(value.float()).all() for value in result.values()):
        raise FloatingPointError("non-finite local Cm feature")
    return result


@torch.no_grad()
def metrics(model, data, batch_size=256):
    model.eval()
    predictions = []
    for start in range(0, len(data["target"]), batch_size):
        output = model(data["region"][start:start + batch_size],
                       data["context"][start:start + batch_size])
        predictions.append(output["delta_local"])
    pred = torch.cat(predictions)
    error = (pred - data["target"]).norm(dim=-1) * 1000
    zero = data["target"].norm(dim=-1) * 1000
    result = {}
    for category, name in enumerate(("moving_contact", "static_contact", "no_contact")):
        selected = data["category"] == category
        result[name] = {"count": int(selected.sum()),
                        "epe_mm": float(error[selected].mean()),
                        "zero_mm": float(zero[selected].mean()),
                        "predicted_motion_mm": float(pred[selected].norm(dim=-1).mean() * 1000)}
    return result


def fit_arm(initial, train, eval_data, *, steps, batch_size, seed):
    model = SixRegionCm()
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    categories = [torch.where(train["category"] == i)[0] for i in range(3)]
    if any(len(items) == 0 for items in categories):
        raise ValueError("training category missing")
    # Equal category quota per minibatch, same schedule for both arms.
    per = max(1, batch_size // 3)
    schedule = [torch.cat([items[torch.randint(len(items), (per,), generator=generator)]
                           for items in categories]) for _ in range(steps)]
    reports = {"initial": metrics(model, eval_data)}
    for step, ids in enumerate(schedule, start=1):
        model.train()
        output = model(train["region"][ids], train["context"][ids])
        prediction = output["delta_local"]
        target = train["target"][ids]
        translation = F.smooth_l1_loss(prediction / .01, target / .01,
                                       beta=.5, reduction="none").mean(dim=-1)
        contact_loss = F.binary_cross_entropy_with_logits(
            output["contact_logit"], train["contact"][ids])
        loss = translation.mean() + .1 * contact_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite local Cm loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (steps // 2, steps):
            reports[str(step)] = metrics(model, eval_data)
            reports[str(step)]["train_loss"] = float(loss.detach())
    return model, reports


def execute(args, manifest):
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(2440)
    if sha256(CALIBRATION) != CALIBRATION_SHA256:
        raise ValueError("calibration SHA drift")
    coefficients = json.loads(CALIBRATION.read_text())["coefficients"]
    calibration = {name: torch.tensor(coefficients[name], dtype=torch.float32)
                   for name in ("alpha", "beta")}
    hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
    kinematics = TorchInspireKinematics(hand_urdf, "cpu")
    geometry = _surface_geometry_class()(
        hand_urdf=hand_urdf, object_urdf=ASSETS / "mjcf/airplane.urdf",
        query_links=QUERY_LINKS, object_count=256, hand_count=256, seed=42, device="cpu")
    lower, upper = native_joint_limits(hand_urdf, "cpu")
    train_rows = [select_rows(path, digest, seed, args.train_per_group, lower, upper)
                  for seed, (path, digest) in TRAIN.items()]
    test_rows = [select_rows(path, digest, seed, args.test_per_group, lower, upper)
                 for seed, (path, digest) in TRANSITIONS.items()]
    def merged(items):
        return {key: torch.cat([item[key] for item in items]) for key in items[0]}
    train_rows, test_rows = merged(train_rows), merged(test_rows)
    train_cal = extract_features(train_rows, kinematics=kinematics, geometry=geometry,
                                 lower=lower, upper=upper, calibration=calibration,
                                 batch_size=32)
    test_cal = extract_features(test_rows, kinematics=kinematics, geometry=geometry,
                                lower=lower, upper=upper, calibration=calibration,
                                batch_size=32)
    train_zero = dict(train_cal)
    train_zero["region"] = train_cal["region"].clone()
    train_zero["region"][..., 11:16] = 0.0
    test_zero = dict(test_cal)
    test_zero["region"] = test_cal["region"].clone()
    test_zero["region"][..., 11:16] = 0.0
    # Region layout: distance(1), proximity(1), relative(3), position(3),
    # object normal(3), flow(3), toward(1), flow magnitude(1), hand normal(3).
    torch.manual_seed(2441)
    initial = {key: value.detach().clone() for key, value in SixRegionCm().state_dict().items()}
    model_cal, report_cal = fit_arm(initial, train_cal, test_cal,
                                    steps=args.steps, batch_size=args.batch_size, seed=2442)
    _, report_zero = fit_arm(initial, train_zero, test_zero,
                             steps=args.steps, batch_size=args.batch_size, seed=2442)
    test_shuffled = extract_features(test_rows, kinematics=kinematics, geometry=geometry,
                                     lower=lower, upper=upper, calibration=calibration,
                                     batch_size=32, shuffle_actions=True)
    report = {"schema": "ref2dex.local_geometric_cm_probe.v1",
              "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
              "args": dict(vars(args), output=str(args.output)),
              "train_sha256": manifest["train_sha256"],
              "test_sha256": manifest["test_sha256"],
              "calibration_sha256": CALIBRATION_SHA256,
              "parameter_count": sum(p.numel() for p in model_cal.parameters()),
              "feature_geometry": {"object_points": 256, "hand_points": 256,
                                   "regions": NUM_REGIONS, "features_per_region": FEATURE_DIM},
              "train_samples": len(train_cal["target"]),
              "test_samples": len(test_cal["target"]),
              "calibrated": report_cal, "zero_flow": report_zero,
              "calibrated_shuffled_action": metrics(model_cal, test_shuffled),
              "elapsed_seconds": time.monotonic() - start,
              "interpretation_limit": "single trajectory observational actions; shuffle is not physical counterfactual"}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                    report_sha256=sha256(report_path), final_step=args.steps,
                    elapsed_seconds=report["elapsed_seconds"])
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"calibrated": report_cal[str(args.steps)],
                      "zero_flow": report_zero[str(args.steps)],
                      "shuffled": report["calibrated_shuffled_action"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-group", type=int, default=128)
    parser.add_argument("--test-per-group", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    if args.output.exists() or min(args.train_per_group, args.test_per_group,
                                    args.steps, args.batch_size) < 1:
        raise ValueError("new output and positive counts required")
    args.output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 60,
                "output_budget_gb": 1,
                "stop_rule": "input SHA drift, nonfinite, leakage, resource or wall budget",
                "args": dict(vars(args), output=str(args.output)),
                "calibration_sha256": CALIBRATION_SHA256,
                "train_sha256": {str(seed): item[1] for seed, item in TRAIN.items()},
                "test_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()}}
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        execute(args, manifest)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        failure=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
