"""Matched small-data simulation adaptation: mixed-pretrained versus scratch Cm.

Only the object-effect head is evaluated; all conditioning uses pre-action
state and nominal action flow.  This is a decision Probe, not a validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_root_pose, native_joint_limits,
    world_to_object_frame,
)
from src.task.CmResidual.cmlite import local_translation_target
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS, TRANSITIONS
from src.task.CmResidual.tools.probe_mixed_cm_sim_transfer import (
    CHECKPOINT, CHECKPOINT_SHA256, object_flow_target, sha256,
)
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


ROOT = Path(__file__).resolve().parents[4]
TRAIN = {
    74: (ROOT / "outputs/Dexplore/agent_v130_s3_transfer_cmoff_s70_e160/"
         "eval_s74_e160_full/transitions.pt",
         "a34b30e635dddbe2e1d87766a0075181754c4f5ae4d4e069c388c7d3776dd520"),
    78: (ROOT / "outputs/Dexplore/agent_v135_s3_cmoff_s70_e180/"
         "eval_s78_e180_full/transitions.pt",
         "09a0aee8ae2e816c3a34b5c52cc9a7dfedea2ddc221187312be576521303e9e4"),
}
INPUT_KEYS = ("obj_points", "obj_normals", "hand_points", "hand_normals", "hand_flow")


def select_moving_contact(payload, seed: int, each: int):
    mask = first_episode_mask(payload["done"], 64)
    rows = {key: value[mask] for key, value in payload.items()
            if isinstance(value, torch.Tensor)}
    contact = (rows["hand_contact"].bool() & rows["object_contact"].bool()).reshape(-1)
    motion = local_translation_target(rows["object_state"], rows["next_object_state"]).norm(dim=-1)
    generator = torch.Generator().manual_seed(2312 + seed)
    indices = []
    for candidates in (torch.where(contact & (motion > .002))[0], torch.where(~contact)[0]):
        if len(candidates) < each:
            raise ValueError(f"seed {seed} lacks {each} required samples")
        indices.append(candidates[torch.randperm(len(candidates), generator=generator)[:each]])
    selected = torch.cat(indices)
    result = {key: value[selected] for key, value in rows.items()}
    result["contact_group"] = contact[selected]
    return result


@torch.no_grad()
def materialize(bridge, lower, upper, sources, each: int) -> dict[str, torch.Tensor]:
    chunks = {key: [] for key in (*INPUT_KEYS, "target", "contact")}
    for seed, (path, expected) in sources.items():
        if sha256(path) != expected:
            raise ValueError(f"transition SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "ref2dex.cmlite_transition.v1":
            raise ValueError(f"transition schema mismatch: {path}")
        rows = select_moving_contact(payload, seed, each)
        for start in range(0, len(rows["q"]), 2):
            stop = start + 2
            q = rows["q"][start:stop]
            action = rows["action"][start:stop]
            state = rows["object_state"][start:stop]
            current = bridge.current(q, state)
            _, _, flow_world = bridge.nominal_hand_sweep(q, action[:, None], lower, upper)
            pose = current.object_pose
            values = {
                "obj_points": world_to_object_frame(current.object_points, pose, vector=False),
                "obj_normals": world_to_object_frame(current.object_normals, pose, vector=True),
                "hand_points": world_to_object_frame(current.hand_points, pose, vector=False),
                "hand_normals": world_to_object_frame(current.hand_normals, pose, vector=True),
                "hand_flow": world_to_object_frame(flow_world[:, 0], pose, vector=True),
            }
            values["target"] = object_flow_target(
                values["obj_points"], pose,
                dexplore_root_pose(rows["next_object_state"][start:stop]))
            values["contact"] = rows["contact_group"][start:stop].bool()
            for key, value in values.items():
                if not torch.isfinite(value.float()).all():
                    raise FloatingPointError(f"non-finite {key}")
                chunks[key].append(value.cpu())
    return {key: torch.cat(value) for key, value in chunks.items()}


def object_only(model: ObjectInteractionCmModel, batch: dict[str, torch.Tensor]):
    """The V1.3 object path, excluding the unused hand reconstruction head."""
    obj = batch["obj_points"].float()
    normals = F.normalize(batch["obj_normals"].float(), dim=-1, eps=1e-6)
    hand = batch["hand_points"].float()
    hand_normals = F.normalize(batch["hand_normals"].float(), dim=-1, eps=1e-6)
    hand_flow = batch["hand_flow"].float()
    obj_feat = model.object_encoder(torch.cat((obj, normals), dim=-1))
    _, diag = model.local_interaction(
        obj_feat, obj, normals, hand, hand_normals, hand_flow)
    fused = model.fusion(torch.cat((obj_feat, diag["interaction_geo"],
                                    diag["interaction_flow"], diag["flow_mean"],
                                    diag["flow_magnitude"]), dim=-1))
    tokens = model.fusion_norm(obj_feat + fused)
    valid = diag["has_interaction"]
    slots, _, weights, _ = model.slot_attention(tokens, valid)
    compressed = model.cm_projection(slots)
    anchors = torch.einsum("bsn,bnd->bsd", weights, obj)
    anchor_normals = F.normalize(torch.einsum("bsn,bnd->bsd", weights, normals), dim=-1, eps=1e-6)
    prediction, _ = model.object_decoder(obj, normals, compressed, anchors, anchor_normals)
    return prediction * model.object_flow_target_scale, valid.any(dim=1)


@torch.no_grad()
def evaluate(model, data, batch_size=2):
    model.eval()
    totals = {group: {"count": 0, "valid": 0, "epe_mm": 0.0,
                      "gated_epe_mm": 0.0, "zero_mm": 0.0}
              for group in ("contact", "no_contact")}
    for start in range(0, len(data["contact"]), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in data.items()}
        pred, valid = object_only(model, batch)
        target = batch["target"]
        error = (pred - target).norm(dim=-1).mean(dim=-1) * 1000
        gated = (pred * valid[:, None, None] - target).norm(dim=-1).mean(dim=-1) * 1000
        zero = target.norm(dim=-1).mean(dim=-1) * 1000
        for i, contact in enumerate(batch["contact"]):
            item = totals["contact" if contact else "no_contact"]
            item["count"] += 1
            item["valid"] += int(valid[i])
            item["epe_mm"] += float(error[i])
            item["gated_epe_mm"] += float(gated[i])
            item["zero_mm"] += float(zero[i])
    for item in totals.values():
        for key in ("epe_mm", "gated_epe_mm", "zero_mm"):
            item[key] /= item["count"]
        item["valid_fraction"] = item["valid"] / item["count"]
    return totals


def run(args, manifest):
    torch.set_num_threads(2)
    torch.manual_seed(2309)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("mixed pretrained checkpoint SHA mismatch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    meta = SimpleNamespace(**payload["config"]["meta"])
    bridge = DExploreCmv2GeometryBridge(
        hand_urdf=ASSETS / "inspire_hand_new/inspire_hand_right.urdf",
        object_urdf=ASSETS / "mjcf/airplane.urdf", device="cpu", seed=42)
    lower, upper = native_joint_limits(bridge.hand_urdf, "cpu")
    train = materialize(bridge, lower, upper, TRAIN, args.train_per_group_per_seed)
    heldout = materialize(bridge, lower, upper, TRANSITIONS, args.eval_per_group_per_seed)
    pretrained = ObjectInteractionCmModel(SimpleNamespace(meta=meta, work_version="V1.3"))
    pretrained.load_state_dict(payload["model"], strict=True)
    torch.manual_seed(2310)
    scratch = ObjectInteractionCmModel(SimpleNamespace(meta=meta, work_version="V1.3"))
    # The unused hand head is present for exact architectural parity but never updated.
    models = {"pretrained": pretrained, "scratch": scratch}
    optimizers = {name: torch.optim.AdamW(
        (param for key, param in model.named_parameters() if not key.startswith("hand_decoder.")),
        lr=args.lr, weight_decay=1e-4) for name, model in models.items()}
    generator = torch.Generator().manual_seed(2311)
    contact_ids = torch.where(train["contact"])[0]
    no_contact_ids = torch.where(~train["contact"])[0]
    if not len(contact_ids) or not len(no_contact_ids):
        raise ValueError("balanced contact training split missing")
    contacts = contact_ids[torch.randint(len(contact_ids), (args.steps,), generator=generator)]
    noncontacts = no_contact_ids[torch.randint(len(no_contact_ids), (args.steps,), generator=generator)]
    schedule = torch.stack((contacts, noncontacts), dim=-1)
    reports = {name: {"initial": evaluate(model, heldout)} for name, model in models.items()}
    for name, model in models.items():
        model.train()
        optimizer = optimizers[name]
        losses = []
        for step, ids in enumerate(schedule, start=1):
            batch = {key: value[ids] for key, value in train.items()}
            pred, valid = object_only(model, batch)
            target = batch["target"]
            point_loss = F.smooth_l1_loss(pred, target, beta=.005, reduction="none").mean(dim=(1, 2))
            moving = target.norm(dim=-1).mean(dim=-1) > .001
            weight = 1.0 + 4.0 * batch["contact"].float() + 4.0 * moving.float()
            loss = (point_loss * weight).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at {name}/{step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if step in (10, 20, args.steps):
                reports[name][str(step)] = evaluate(model, heldout)
                reports[name][str(step)]["mean_train_loss"] = sum(losses) / len(losses)
                model.train()
                print(json.dumps({"arm": name, "step": step,
                                  "contact": reports[name][str(step)]["contact"]}), flush=True)
    report = {"schema": "ref2dex.mixed_cm_sim_adapt_probe.v1",
              "status": "COMPLETED", "args": dict(vars(args), output=str(args.output)),
              "checkpoint_sha256": CHECKPOINT_SHA256,
              "train_sha256": {str(seed): item[1] for seed, item in TRAIN.items()},
              "eval_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()},
              "train_samples": len(train["contact"]), "eval_samples": len(heldout["contact"]),
              "reports": reports,
              "limitation": "small observational same-trajectory probe; no physical action counterfactual or PPO utility claim"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                    report_sha256=sha256(args.output / "report.json"))
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-group-per-seed", type=int, default=16)
    parser.add_argument("--eval-per-group-per-seed", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists() or min(args.train_per_group_per_seed,
                                    args.eval_per_group_per_seed, args.steps) < 1:
        raise ValueError("new output and positive counts required")
    args.output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "run_id": args.output.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "cpu_threads": 2, "gpu_count": 0, "output_budget_mb": 10,
                "wall_budget_minutes": 60, "stop_rule": "nonfinite loss or input SHA drift",
                "args": dict(vars(args), output=str(args.output)),
                "train_sha256": {str(seed): item[1] for seed, item in TRAIN.items()},
                "eval_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()},
                "pretrained_sha256": CHECKPOINT_SHA256}
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        run(args, manifest)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        failure=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
