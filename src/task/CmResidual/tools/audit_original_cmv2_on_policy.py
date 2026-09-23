"""Compare frozen original Cmv2 and CmLite on fixed self-trained PPO transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from src.task.CmResidual.cm_v2_adapter import FrozenCmv2Adapter
from src.task.CmResidual.cm_v2_action_evaluator import effect_metrics, object_pose_delta_to_xi
from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_root_pose, native_joint_limits,
)
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask


ROOT = Path(__file__).resolve().parents[4]
CMV2 = ROOT / "outputs/CmResidual/agent_v149_cmv2_input/original_v13.pt"
CMLITE = ROOT / "outputs/CmLite/V1.37/balanced_s1x24_s3_e160_e180/best.pt"
CMV2_SHA = "371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591"
CMLITE_SHA = "396f5e0c92ecf67b2a096568c1a538b7c86d40779a594729f786df6c64ed204a"
EXTERNAL = Path("/home2/wyy/oyx_ws/Ref2Dex/third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/latest.pt")
TRANSITIONS = {
    95: (ROOT / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/"
         "eval_s95_e260_full_v142_cm_audit/transitions.pt",
         "ae4cd504912ba5abcf86d75519c2b8f372e84c0fa23a279827369f47165098bd"),
    96: (ROOT / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/"
         "eval_s96_e260_full_v142_cm_audit/transitions.pt",
         "bc656eaed82b714cf1da1115e6d800261917e0f80a74dcc8823e9c1fa006e360"),
}
ASSETS = ROOT / "third_party/DExplore/dexplore/data/assets"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sync() -> None:
    torch.cuda.synchronize()


def select(payload: dict, seed: int, each: int) -> dict[str, torch.Tensor]:
    mask = first_episode_mask(payload["done"], 64)
    rows = {key: value[mask] for key, value in payload.items()
            if isinstance(value, torch.Tensor)}
    contact = (rows["hand_contact"].reshape(-1).bool() &
               rows["object_contact"].reshape(-1).bool())
    generator = torch.Generator().manual_seed(149 + seed)
    indices = []
    for choice in (True, False):
        candidates = torch.where(contact == choice)[0]
        if len(candidates) < each:
            raise ValueError(f"seed {seed} lacks {each} contact={choice} rows")
        indices.append(candidates[torch.randperm(len(candidates), generator=generator)[:each]])
    selected = torch.cat(indices)
    data = {key: value[selected] for key, value in rows.items()}
    data["contact_group"] = contact[selected]
    shuffled = data["action"].clone()
    for start in (0, each):
        order = torch.randperm(each, generator=generator)
        shuffled[start:start + each] = data["action"][start:start + each][order]
    data["shuffled_action"] = shuffled
    data["selected_first_episode_indices"] = selected
    return data


def evaluate_seed(data: dict[str, torch.Tensor], *, bridge: DExploreCmv2GeometryBridge,
                  cmv2: FrozenCmv2Adapter, lite: FrozenCmLite,
                  lower: torch.Tensor, upper: torch.Tensor,
                  microbatch: int) -> dict:
    count = data["q"].shape[0]
    device = lower.device
    rows = {"gt_local": [], "cmv2_local": [], "cmv2_shuffled_local": [],
            "lite_local": [], "lite_shuffled_local": [], "cmv2_rotation": [],
            "gt_rotation": [], "token_active": []}
    timing = {key: [] for key in ("geometry_ms", "cmv2_ms", "shuffled_geometry_ms",
                                   "shuffled_cmv2_ms", "cmlite_ms")}
    for start in range(0, count, microbatch):
        stop = min(start + microbatch, count)
        q = data["q"][start:stop].to(device)
        action = data["action"][start:stop].to(device)
        shuffled = data["shuffled_action"][start:stop].to(device)
        state = data["object_state"][start:stop].to(device)
        next_state = data["next_object_state"][start:stop].to(device)
        gt = object_pose_delta_to_xi(dexplore_root_pose(state),
                                     dexplore_root_pose(next_state))
        sync()
        t0 = time.perf_counter()
        geometry = bridge.current(q, state)
        _, _, flow = bridge.nominal_hand_sweep(q, action[:, None], lower, upper)
        sync()
        t1 = time.perf_counter()
        prediction = cmv2.predict_effect_only(
            geometry.object_points, geometry.object_normals,
            geometry.hand_points, geometry.hand_normals,
            flow[:, 0], 1.0 / 30.0, interaction_object_chunk=32)
        sync()
        t2 = time.perf_counter()
        _, _, shuffled_flow = bridge.nominal_hand_sweep(q, shuffled[:, None], lower, upper)
        sync()
        t3 = time.perf_counter()
        shuffled_prediction = cmv2.predict_effect_only(
            geometry.object_points, geometry.object_normals,
            geometry.hand_points, geometry.hand_normals,
            shuffled_flow[:, 0], 1.0 / 30.0, interaction_object_chunk=32)
        sync()
        t4 = time.perf_counter()
        lite_prediction = lite.predict(q, action, state)
        lite_shuffled = lite.predict(q, shuffled, state)
        sync()
        t5 = time.perf_counter()
        for key, value in (
            ("gt_local", gt[:, :3]),
            ("gt_rotation", gt[:, 3:]),
            ("cmv2_local", prediction["delta_xi_root"][:, :3]),
            ("cmv2_rotation", prediction["delta_xi_root"][:, 3:]),
            ("cmv2_shuffled_local", shuffled_prediction["delta_xi_root"][:, :3]),
            ("lite_local", lite_prediction["delta_local"]),
            ("lite_shuffled_local", lite_shuffled["delta_local"]),
            ("token_active", prediction["token_mask"].any(-1))):
            if not torch.isfinite(value.float()).all():
                raise FloatingPointError(f"non-finite {key}")
            rows[key].append(value.cpu())
        for key, duration in (("geometry_ms", t1-t0), ("cmv2_ms", t2-t1),
                              ("shuffled_geometry_ms", t3-t2),
                              ("shuffled_cmv2_ms", t4-t3), ("cmlite_ms", t5-t4)):
            timing[key].append(duration * 1000)
    rows = {key: torch.cat(value) for key, value in rows.items()}
    errors = {"zero": rows["gt_local"].norm(dim=-1),
              "cmv2": (rows["cmv2_local"] - rows["gt_local"]).norm(dim=-1),
              "cmv2_shuffled": (rows["cmv2_shuffled_local"] - rows["gt_local"]).norm(dim=-1),
              "cmlite": (rows["lite_local"] - rows["gt_local"]).norm(dim=-1),
              "cmlite_shuffled": (rows["lite_shuffled_local"] - rows["gt_local"]).norm(dim=-1)}
    report = {"samples": count, "groups": {}, "timing_ms_per_microbatch": {
        key: {"median": statistics.median(value), "mean": statistics.mean(value)}
        for key, value in timing.items()},
        "microbatch": microbatch,
        "cmv2_params": sum(p.numel() for p in cmv2.model.parameters()),
        "cmlite_params": sum(p.numel() for p in lite.model.parameters())}
    for name, group in (("contact", data["contact_group"].bool()),
                        ("no_contact", ~data["contact_group"].bool()),
                        ("all", torch.ones(count, dtype=torch.bool))):
        item = {"samples": int(group.sum()),
                "token_active_fraction": float(rows["token_active"][group].float().mean()),
                "mean_cmv2_predicted_action_change_mm": float((
                    rows["cmv2_local"] - rows["cmv2_shuffled_local"]).norm(dim=-1)[group].mean() * 1000),
                "mean_cmlite_predicted_action_change_mm": float((
                    rows["lite_local"] - rows["lite_shuffled_local"]).norm(dim=-1)[group].mean() * 1000),
                "cmv2_rotation_error_rad": float(effect_metrics(
                    torch.cat((rows["cmv2_local"], rows["cmv2_rotation"]), -1)[group],
                    torch.cat((rows["gt_local"], rows["gt_rotation"]), -1)[group]
                )["rotation_error_rad"].mean())}
        item["translation_epe_mm"] = {
            key: float(value[group].mean() * 1000) for key, value in errors.items()}
        item["cmv2_improvement_vs_zero_fraction"] = 1 - item["translation_epe_mm"]["cmv2"] / item["translation_epe_mm"]["zero"]
        item["cmv2_shuffle_penalty_fraction"] = item["translation_epe_mm"]["cmv2_shuffled"] / item["translation_epe_mm"]["cmv2"] - 1
        item["cmlite_improvement_vs_zero_fraction"] = 1 - item["translation_epe_mm"]["cmlite"] / item["translation_epe_mm"]["zero"]
        report["groups"][name] = item
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / "outputs/CmResidual" / ("agent_v149_cmv2_audit_smoke"
                                            if args.smoke else "agent_v149_cmv2_audit")
    if output.exists():
        raise FileExistsError(output)
    if (not CMV2.is_symlink() or CMV2.resolve() != EXTERNAL.resolve() or
            sha256(CMV2) != CMV2_SHA or sha256(CMLITE) != CMLITE_SHA):
        raise ValueError("frozen Cmv2/CmLite checkpoint source or SHA mismatch")
    for path, expected in TRANSITIONS.values():
        if sha256(path) != expected:
            raise ValueError(f"transition SHA mismatch: {path}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise ValueError("V1.49 must use only physical GPU5")
    gpu_used = subprocess.check_output(
        ["nvidia-smi", "--id=5", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    if int(gpu_used) > 1024:
        raise RuntimeError(f"GPU5 occupied before V1.49: {gpu_used} MiB")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": output.name, "work_version": "V1.49",
                "git_commit": revision, "physical_gpu": 5,
                "cmv2_checkpoint_sha256": CMV2_SHA,
                "cmlite_checkpoint_sha256": CMLITE_SHA,
                "transition_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()},
                "samples_per_seed_per_group": 2 if args.smoke else 64,
                "seed": 149, "microbatch": 2 if args.smoke else 8,
                "interaction_object_chunk": 32,
                "output_budget_gb": 1,
                "stop_rule": "strict load/finite/sampling failure or GPU resource conflict",
                "official_policy_checkpoint": None}
    write(output / "run_manifest.json", manifest)
    try:
        device = torch.device("cuda:0")
        bridge = DExploreCmv2GeometryBridge(
            hand_urdf=ASSETS / "inspire_hand_new/inspire_hand_right.urdf",
            object_urdf=ASSETS / "mjcf/airplane.urdf", device=device, seed=42)
        lower, upper = native_joint_limits(bridge.hand_urdf, device)
        cmv2 = FrozenCmv2Adapter(CMV2, CMV2_SHA, device)
        lite = FrozenCmLite(str(CMLITE), device, CMLITE_SHA)
        reports = {}
        for seed, (path, _) in TRANSITIONS.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema") != "ref2dex.cmlite_transition.v1":
                raise ValueError("transition schema mismatch")
            data = select(payload, seed, 2 if args.smoke else 64)
            reports[str(seed)] = evaluate_seed(data, bridge=bridge, cmv2=cmv2,
                                               lite=lite, lower=lower, upper=upper,
                                               microbatch=2 if args.smoke else 8)
            print(json.dumps({"seed": seed, "contact": reports[str(seed)]["groups"]["contact"]},
                             sort_keys=True), flush=True)
        gate = (not args.smoke and all(
            reports[str(seed)]["groups"]["contact"]["cmv2_improvement_vs_zero_fraction"] >= 0.20 and
            reports[str(seed)]["groups"]["contact"]["cmv2_shuffle_penalty_fraction"] >= 0.10 and
            reports[str(seed)]["groups"]["contact"]["token_active_fraction"] >= 0.10
            for seed in TRANSITIONS))
        result = {"schema": "ref2dex.v149_original_cmv2_audit.v1",
                  "run_status": "COMPLETED", "work_version": "V1.49",
                  "is_smoke": args.smoke, "reports": reports,
                  "original_cmv2_offline_signal_gate": gate,
                  "interpretation_limit": "observational shuffled actions are not physical counterfactuals"}
        write(output / "audit.json", result)
        manifest.update(run_status="COMPLETED", completed_at=now(),
                        audit_json=str(output / "audit.json"),
                        audit_sha256=sha256(output / "audit.json"),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())
        write(output / "run_manifest.json", manifest)
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=now(),
                        failure=f"{type(error).__name__}: {error}")
        write(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
