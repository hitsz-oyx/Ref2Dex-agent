"""Probe frozen ObjectInteractionCm V1.3 tokens with object-LOO heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.base import task_config_from_dict
from src.task.CmResidual.dexplore_cm_geometry import (
    dexplore_action_to_native_targets, dexplore_root_pose, native_joint_limits,
    world_to_object_frame,
)
from src.task.CmResidual.tools.analyze_stratified_e320_support import (
    CHECKPOINT_SHA, SOURCES,
)
from src.task.CmResidual.tools.analyze_train5_state_coverage import sha256
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / "third_party/DExplore/dexplore/data/assets"
CM_CHECKPOINT = Path(
    "/home2/wyy/oyx_ws/Ref2Dex/outputs/objectinteractioncm/"
    "object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt")
CM_SHA = "3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283"
CALIBRATION = ROOT / (
    "outputs/CmResidual/agent_intervention_handflow_mix_s145147_"
    "train_s146148_test/report.json")
CALIBRATION_SHA = "d0342ff4e17c66bb1d2e999416eecf24a0754c891e35916822f10609b42bdde3"
OBJECTS = ("airplane", "cubesmall", "mug", "toothpaste", "waterbottle")
SOURCE_FOR = {"airplane": "mixed", "cubesmall": "mixed", "mug": "mixed",
              "toothpaste": "toothpaste", "waterbottle": "waterbottle"}
NUM_ENVS = 64
HORIZON = 10
PER_CLASS = 64
HIGH_DENSITY = 10135
LOW_DENSITY = 1538


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def surface_geometry(*, hand_urdf: Path, object_urdf: Path, hand_count: int,
                     device: torch.device):
    from src.task.CmResidual.dexplore_cm_geometry import _surface_geometry_class
    return _surface_geometry_class()(
        hand_urdf=hand_urdf, object_urdf=object_urdf, query_links=QUERY_LINKS,
        object_count=1024, hand_count=hand_count, seed=42, device=device)


def load_source(name: str) -> dict:
    source = SOURCES[name]
    path = source["root"] / "transitions.pt"
    manifest = json.loads((source["root"] / "run_manifest.json").read_text())
    if (sha256(path) != source["transition_sha"] or
            manifest.get("transition_sha256") != source["transition_sha"] or
            manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
            manifest.get("input_manifest_sha256") != source["split_sha"] or
            manifest.get("run_status") != "COMPLETED"):
        raise ValueError(f"source provenance drift: {name}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError(f"transition schema drift: {name}")
    return payload


def balanced_env_sample(candidates: list[tuple[int, int]], count: int,
                        generator: torch.Generator) -> list[int]:
    by_env: dict[int, list[int]] = {}
    for index, env in candidates:
        by_env.setdefault(env, []).append(index)
    for values in by_env.values():
        order = torch.randperm(len(values), generator=generator).tolist()
        values[:] = [values[i] for i in order]
    envs = list(by_env)
    env_order = torch.randperm(len(envs), generator=generator).tolist()
    envs = [envs[i] for i in env_order]
    selected = []
    while len(selected) < count:
        progress = False
        for env in envs:
            if by_env[env]:
                selected.append(by_env[env].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise ValueError(f"only {len(selected)} candidates for quota {count}")
    return selected


def select_object_rows(payload: dict, mapping: list[str], object_name: str,
                       seed: int) -> dict[str, torch.Tensor]:
    positive: list[tuple[int, int]] = []
    failure: list[tuple[int, int]] = []
    for env in range(NUM_ENVS):
        indices = torch.arange(env, len(payload["q"]), NUM_ENVS)
        ended = torch.where(payload["done"].reshape(-1)[indices].bool())[0]
        if not len(ended):
            raise ValueError(f"environment {env} has no complete first episode")
        indices = indices[:int(ended[0])]
        if len(indices) <= HORIZON:
            continue
        data_ids = payload["data_id"].reshape(-1)[indices].long()
        if not torch.equal(data_ids, data_ids[:1].expand_as(data_ids)):
            raise ValueError("motion changed within first episode")
        motion_id = int(data_ids[0])
        if not 0 <= motion_id < len(mapping) or mapping[motion_id] != object_name:
            continue
        contact = (payload["hand_contact"].reshape(-1)[indices].bool() &
                   payload["object_contact"].reshape(-1)[indices].bool())
        z = payload["object_state"][indices, 2]
        for local in torch.where(contact[:-HORIZON])[0].tolist():
            future_contact = float(contact[local + 1:local + HORIZON + 1].float().mean())
            dz = float(z[local + HORIZON] - z[local])
            pair = (int(indices[local]), env)
            if dz >= .010 and future_contact >= .5:
                positive.append(pair)
            elif future_contact < .5 or dz <= 0:
                failure.append(pair)
    generator = torch.Generator().manual_seed(seed)
    chosen_positive = balanced_env_sample(positive, PER_CLASS, generator)
    chosen_failure = balanced_env_sample(failure, PER_CLASS, generator)
    chosen = torch.tensor(chosen_positive + chosen_failure, dtype=torch.long)
    rows = {key: value[chosen].clone() for key, value in payload.items()
            if isinstance(value, torch.Tensor)}
    rows["label"] = torch.cat((torch.ones(PER_CLASS), torch.zeros(PER_CLASS)))
    rows["source_index"] = chosen
    rows["object_id"] = torch.full((2 * PER_CLASS,), OBJECTS.index(object_name),
                                   dtype=torch.long)
    rows["available_positive"] = torch.tensor(len(positive))
    rows["available_failure"] = torch.tensor(len(failure))
    return rows


def encode(output: dict, object_points: torch.Tensor) -> torch.Tensor:
    tokens = output["cm_tokens"]
    anchors = output["cm_anchor_pos"]
    normals = F.normalize(output["cm_anchor_normal"], dim=-1, eps=1e-8)
    mass = output["cm_assignment"].sum(dim=-1)
    mask = (mass > 0) & output["sample_valid"][:, None]
    center = object_points.mean(dim=1)
    radius = (object_points - center[:, None]).square().sum(-1).mean(-1).sqrt().clamp_min(1e-8)
    relative = (anchors - center[:, None]) / radius[:, None, None]
    log_mass = torch.log1p(mass) / torch.log(torch.tensor(1025., device=mass.device))
    features = torch.cat((tokens, relative, normals, log_mass[..., None],
                          mask[..., None].to(tokens.dtype)), dim=-1)
    features = torch.where(mask[..., None], features, torch.zeros_like(features))
    if features.shape[1:] != (16, 40) or not torch.isfinite(features).all():
        raise RuntimeError("invalid canonical ObjectInteractionCm context")
    return features.flatten(1)


@torch.inference_mode()
def extract_object(rows: dict[str, torch.Tensor], object_name: str, *, model,
                   kinematics, lower, upper, calibration, device, batch_size: int) -> dict:
    hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
    object_urdf = ASSETS / f"mjcf/{object_name}.urdf"
    low = surface_geometry(hand_urdf=hand_urdf, object_urdf=object_urdf,
                           hand_count=LOW_DENSITY, device=device)
    high = surface_geometry(hand_urdf=hand_urdf, object_urdf=object_urdf,
                            hand_count=HIGH_DENSITY, device=device)
    permutation = torch.randperm(len(rows["q"]), generator=torch.Generator().manual_seed(
        249100 + OBJECTS.index(object_name)))
    shuffled_action = rows["action"][permutation]
    result = {key: [] for key in ("aware_high", "blind_high", "shuffle_high", "aware_low")}
    validity = {key: [] for key in result}
    for start in range(0, len(rows["q"]), batch_size):
        stop = min(start + batch_size, len(rows["q"]))
        q = rows["q"][start:stop].to(device)
        velocity = rows["dof_vel"][start:stop].to(device)
        action = rows["action"][start:stop].to(device)
        shuffled = shuffled_action[start:stop].to(device)
        state = rows["object_state"][start:stop].to(device)
        pose = dexplore_root_pose(state)
        links = kinematics.forward(q[:, None])[:, 0]
        low_hand_world, low_normal_world = low.hand(links)
        high_hand_world, high_normal_world = high.hand(links)
        object_world, object_normal_world = low.object(pose)
        obj = world_to_object_frame(object_world, pose, vector=False)
        obj_normals = world_to_object_frame(object_normal_world, pose, vector=True)
        low_hand = world_to_object_frame(low_hand_world, pose, vector=False)
        low_normals = world_to_object_frame(low_normal_world, pose, vector=True)
        high_hand = world_to_object_frame(high_hand_world, pose, vector=False)
        high_normals = world_to_object_frame(high_normal_world, pose, vector=True)
        pd = dexplore_action_to_native_targets(action, q, lower, upper)
        shuffled_pd = dexplore_action_to_native_targets(shuffled, q, lower, upper)
        predicted_q = (q + calibration[:, 0] * (pd - q) +
                       calibration[:, 1] * velocity + calibration[:, 2])
        shuffled_q = (q + calibration[:, 0] * (shuffled_pd - q) +
                      calibration[:, 1] * velocity + calibration[:, 2])
        next_links = kinematics.forward(predicted_q[:, None])[:, 0]
        shuffled_links = kinematics.forward(shuffled_q[:, None])[:, 0]
        next_low_world, _ = low.hand(next_links)
        shuffled_low_world, _ = low.hand(shuffled_links)
        next_high_world, _ = high.hand(next_links)
        shuffled_high_world, _ = high.hand(shuffled_links)
        low_flow = world_to_object_frame(next_low_world - low_hand_world, pose, vector=True)
        low_shuffle_flow = world_to_object_frame(
            shuffled_low_world - low_hand_world, pose, vector=True)
        high_flow = world_to_object_frame(next_high_world - high_hand_world, pose, vector=True)
        high_shuffle_flow = world_to_object_frame(
            shuffled_high_world - high_hand_world, pose, vector=True)
        high_knn = torch.topk(torch.cdist(obj, high_hand), k=32, largest=False).indices
        low_knn = torch.topk(torch.cdist(obj, low_hand), k=32, largest=False).indices
        common = {"obj_points": obj, "obj_normals": obj_normals,
                  "hand_points": low_hand, "hand_normals": low_normals,
                  "hand_valid_mask": torch.ones(low_hand.shape[:2], dtype=torch.bool,
                                                 device=device)}
        variants = {
            "aware_high": {"hand_flow": low_flow, "knn_hand_points": high_hand,
                           "knn_hand_normals": high_normals, "knn_hand_flow": high_flow,
                           "knn_hand_valid_mask": torch.ones(high_hand.shape[:2], dtype=torch.bool,
                                                             device=device),
                           "knn_edge_indices": high_knn},
            "blind_high": {"hand_flow": torch.zeros_like(low_flow),
                           "knn_hand_points": high_hand, "knn_hand_normals": high_normals,
                           "knn_hand_flow": torch.zeros_like(high_flow),
                           "knn_hand_valid_mask": torch.ones(high_hand.shape[:2], dtype=torch.bool,
                                                             device=device),
                           "knn_edge_indices": high_knn},
            "shuffle_high": {"hand_flow": low_shuffle_flow,
                             "knn_hand_points": high_hand, "knn_hand_normals": high_normals,
                             "knn_hand_flow": high_shuffle_flow,
                             "knn_hand_valid_mask": torch.ones(high_hand.shape[:2], dtype=torch.bool,
                                                               device=device),
                             "knn_edge_indices": high_knn},
            "aware_low": {"hand_flow": low_flow, "knn_edge_indices": low_knn},
        }
        for name, extra in variants.items():
            output = model({**common, **extra})
            result[name].append(encode(output, obj).cpu())
            validity[name].append(output["sample_valid"].cpu())
    return {**{name: torch.cat(value) for name, value in result.items()},
            "validity": {name: torch.cat(value) for name, value in validity.items()}}


def auc(target: torch.Tensor, score: torch.Tensor) -> float:
    positive = score[target.bool()]
    negative = score[~target.bool()]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both classes")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + .5 * (comparisons == 0).float()).mean())


class LinearHead(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.output = nn.Linear(dimension, 1)

    def forward(self, value):
        return self.output(value).squeeze(-1)


def loo_scores(features: dict[str, torch.Tensor], labels: torch.Tensor,
               object_ids: torch.Tensor, steps: int) -> dict:
    reports = {}
    for representation, value in features.items():
        folds = {}
        for heldout_id, heldout_name in enumerate(OBJECTS):
            train = object_ids != heldout_id
            test = ~train
            mean = value[train].mean(0)
            std = value[train].std(0).clamp_min(.01)
            x_train = (value[train] - mean) / std
            x_test = (value[test] - mean) / std
            torch.manual_seed(249200 + heldout_id)
            head = LinearHead(value.shape[1])
            optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-2)
            for _ in range(steps):
                loss = F.binary_cross_entropy_with_logits(head(x_train), labels[train])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                test_score = head(x_test)
                train_score = head(x_train)
            folds[heldout_name] = {"auc": auc(labels[test], test_score),
                                   "train_auc": auc(labels[train], train_score),
                                   "test_count": int(test.sum()),
                                   "positive": int(labels[test].sum())}
        reports[representation] = {
            "folds": folds,
            "mean_auc": sum(item["auc"] for item in folds.values()) / len(folds),
            "folds_at_least_0_60": sum(item["auc"] >= .60 for item in folds.values()),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--head-steps", type=int, default=400)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.microbatch <= 4 or not 1 <= args.head_steps <= 1000:
        parser.error("new output, microbatch 1..4 and head steps 1..1000 required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        parser.error("CUDA_VISIBLE_DEVICES must pin the requested physical GPU")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "schema": "ref2dex.cmv13_object_loo_manifest.v1", "run_status": "STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(), "run_id": args.output.name,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        "physical_gpu": args.gpu, "gpu_count": 1, "cpu_threads": 2,
        "checkpoint": str(CM_CHECKPOINT), "checkpoint_sha256": CM_SHA,
        "policy_checkpoint_sha256": CHECKPOINT_SHA,
        "calibration": str(CALIBRATION), "calibration_sha256": CALIBRATION_SHA,
        "source_transition_sha256": {name: value["transition_sha"]
                                      for name, value in SOURCES.items()},
        "objects": OBJECTS, "per_object_per_class": 2 if args.smoke else PER_CLASS,
        "interaction_density": {"high": HIGH_DENSITY, "low": LOW_DENSITY,
                                "knn_k": 32, "radius_m": .02},
        "microbatch": args.microbatch, "head_steps": args.head_steps,
        "wall_budget_minutes": 60, "output_budget_mb": 1024,
        "stop_rule": "provenance/schema/support/non-finite/GPU conflict/wall budget",
        "interpretation_limit": "representation signal only; not policy utility or causality",
    }
    write_json(manifest_path, manifest)
    try:
        if sha256(CM_CHECKPOINT) != CM_SHA or sha256(CALIBRATION) != CALIBRATION_SHA:
            raise ValueError("Cm checkpoint or calibration drift")
        gpu_used = int(subprocess.check_output(
            ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True).strip())
        if gpu_used > 1024:
            raise RuntimeError(f"physical GPU{args.gpu} occupied before run: {gpu_used} MiB")
        torch.set_num_threads(2)
        device = torch.device("cuda:0")
        checkpoint = torch.load(CM_CHECKPOINT, map_location="cpu", weights_only=False)
        cfg = task_config_from_dict(checkpoint["config"])
        # The full task config owns ``meta``. Passing only cfg.model silently
        # falls back to K=8/radius=5cm even though the checkpoint embeds K=32/2cm.
        model = ObjectInteractionCmModel(cfg).to(device).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        model.requires_grad_(False)
        if model.local_interaction.knn_k != 32 or abs(model.local_interaction.radius_m - .02) > 1e-9:
            raise ValueError("embedded V1.3 KNN configuration was not applied")
        hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
        kinematics = TorchInspireKinematics(hand_urdf, device)
        lower, upper = native_joint_limits(hand_urdf, device)
        coefficients = json.loads(CALIBRATION.read_text())["coefficients"]["action_velocity"]
        calibration = torch.tensor(coefficients, dtype=torch.float32, device=device)
        sources = {name: load_source(name) for name in set(SOURCE_FOR.values())}
        all_rows, all_features = [], {name: [] for name in
                                      ("aware_high", "blind_high", "shuffle_high", "aware_low")}
        validity = {}
        for object_id, object_name in enumerate(OBJECTS):
            source_name = SOURCE_FOR[object_name]
            rows = select_object_rows(sources[source_name], SOURCES[source_name]["mapping"],
                                      object_name, seed=249000 + object_id)
            if args.smoke:
                keep = torch.tensor([0, PER_CLASS])
                rows = {key: value[keep] if value.ndim and len(value) == 2 * PER_CLASS else value
                        for key, value in rows.items()}
            extracted = extract_object(rows, object_name, model=model,
                                       kinematics=kinematics, lower=lower, upper=upper,
                                       calibration=calibration, device=device,
                                       batch_size=args.microbatch)
            for name in all_features:
                all_features[name].append(extracted[name])
            validity[object_name] = {name: float(value.float().mean())
                                     for name, value in extracted["validity"].items()}
            all_rows.append(rows)
            print(json.dumps({"object": object_name, "samples": len(rows["label"]),
                              "valid_fraction": validity[object_name]}, sort_keys=True), flush=True)
            if time.monotonic() - started > 3600:
                raise TimeoutError("wall budget exceeded")
        features = {name: torch.cat(value) for name, value in all_features.items()}
        labels = torch.cat([rows["label"] for rows in all_rows])
        object_ids = torch.cat([rows["object_id"] for rows in all_rows])
        raw_parts = []
        for rows in all_rows:
            raw_parts.append(torch.cat((rows["q"], rows["dof_vel"], rows["object_state"],
                                        rows["action"]), dim=-1).float())
        features["raw"] = torch.cat(raw_parts)
        torch.save({"schema": "ref2dex.cmv13_object_loo_features.v1",
                    "features": features, "label": labels, "object_id": object_ids},
                   args.output / "features.pt")
        reports = {} if args.smoke else loo_scores(features, labels, object_ids, args.head_steps)
        if args.smoke:
            gate = False
        else:
            aware = reports["aware_high"]
            gate = bool(
                aware["mean_auc"] >= .65 and aware["folds_at_least_0_60"] >= 4 and
                aware["mean_auc"] - reports["raw"]["mean_auc"] >= .05 and
                aware["mean_auc"] - reports["blind_high"]["mean_auc"] >= .05 and
                aware["mean_auc"] - reports["shuffle_high"]["mean_auc"] >= .05)
        report = {
            "schema": "ref2dex.cmv13_object_loo_representation.v1",
            "classification": "Engineering smoke" if args.smoke else "Probe",
            "representations": reports, "sample_valid_fraction": validity,
            "continue_to_matched_ppo_probe": gate,
            "density_delta_mean_auc": (None if args.smoke else
                                       reports["aware_high"]["mean_auc"] -
                                       reports["aware_low"]["mean_auc"]),
            "gate": {"aware_mean_auc_min": .65, "margin_over_raw_min": .05,
                     "margin_over_blind_min": .05, "margin_over_shuffle_min": .05,
                     "folds_auc_0_60_min": 4},
            "elapsed_seconds": time.monotonic() - started,
            "limits": ["Object-LOO observational outcome prediction is not a physical counterfactual.",
                       "The 10135-point runtime surface approximates the V1.3 maximum unique-KNN density."],
        }
        report_path = args.output / "report.json"
        write_json(report_path, report)
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        report_sha256=sha256(report_path),
                        features_sha256=sha256(args.output / "features.pt"))
        print(json.dumps({"continue_to_matched_ppo_probe": gate,
                          "mean_auc": {name: value["mean_auc"]
                                       for name, value in reports.items()}}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
