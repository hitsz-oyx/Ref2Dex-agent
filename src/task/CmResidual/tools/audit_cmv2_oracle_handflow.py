"""Diagnose Cmv2's nominal hand-flow mismatch using offline oracle next-q."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from src.task.CmResidual.cm_v2_adapter import FrozenCmv2Adapter
from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, dexplore_root_pose, native_joint_limits,
)
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import (
    ASSETS, CMV2, CMV2_SHA, EXTERNAL, ROOT, TRANSITIONS,
    object_pose_delta_to_xi, select, sha256,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sync() -> None:
    torch.cuda.synchronize()


def evaluate_seed(data: dict[str, torch.Tensor], *, bridge: DExploreCmv2GeometryBridge,
                  cmv2: FrozenCmv2Adapter, lower: torch.Tensor,
                  upper: torch.Tensor, microbatch: int) -> dict:
    count = data["q"].shape[0]
    rows = {key: [] for key in (
        "gt", "nominal", "oracle", "zero_hand_flow", "nominal_flow_norm",
        "oracle_flow_norm", "nominal_active", "oracle_active", "zero_active")}
    timing = {key: [] for key in ("geometry_ms", "nominal_model_ms", "oracle_model_ms",
                                   "zero_hand_flow_model_ms")}
    for start in range(0, count, microbatch):
        stop = min(start + microbatch, count)
        q = data["q"][start:stop].to(lower.device)
        next_q = data["next_q"][start:stop].to(lower.device)
        action = data["action"][start:stop].to(lower.device)
        state = data["object_state"][start:stop].to(lower.device)
        next_state = data["next_object_state"][start:stop].to(lower.device)
        gt = object_pose_delta_to_xi(dexplore_root_pose(state),
                                     dexplore_root_pose(next_state))[:, :3]
        sync()
        t0 = time.perf_counter()
        geometry = bridge.current(q, state)
        _, _, nominal_flow = bridge.nominal_hand_sweep(q, action[:, None], lower, upper)
        next_links = bridge.kinematics.forward(next_q[:, None])[:, 0]
        next_points, _ = bridge.geometry.hand(next_links)
        oracle_flow = next_points - geometry.hand_points
        nominal_flow = nominal_flow[:, 0]
        if nominal_flow.shape != oracle_flow.shape or not torch.isfinite(oracle_flow).all():
            raise ValueError("oracle and nominal hand flows must match and be finite")
        sync()
        t1 = time.perf_counter()
        inputs = (geometry.object_points, geometry.object_normals,
                  geometry.hand_points, geometry.hand_normals)
        results = {}
        for name, flow in (("nominal", nominal_flow), ("oracle", oracle_flow),
                           ("zero_hand_flow", torch.zeros_like(nominal_flow))):
            output = cmv2.predict_effect_only(*inputs, flow, 1.0 / 30.0,
                                               interaction_object_chunk=32)
            sync()
            t2 = time.perf_counter()
            results[name] = output
            timing[f"{name}_model_ms"].append((t2 - t1) * 1000)
            t1 = t2
        timing["geometry_ms"].append((t1 - t0) * 1000 - sum(
            timing[f"{name}_model_ms"][-1] for name in results))
        values = {"gt": gt,
                  "nominal_flow_norm": nominal_flow.norm(dim=-1),
                  "oracle_flow_norm": oracle_flow.norm(dim=-1)}
        for name, output in results.items():
            values[name] = output["delta_xi_root"][:, :3]
            values["zero_active" if name == "zero_hand_flow" else f"{name}_active"] = (
                output["token_mask"].any(-1))
        for name, value in values.items():
            if not torch.isfinite(value.float()).all():
                raise FloatingPointError(f"non-finite {name}")
            rows[name].append(value.cpu())
    rows = {name: torch.cat(values) for name, values in rows.items()}
    report = {"samples": count, "microbatch": microbatch,
              "timing_ms_per_microbatch": {
                  key: {"median": statistics.median(value), "mean": statistics.mean(value)}
                  for key, value in timing.items()}, "groups": {}}
    for name, group in (("contact", data["contact_group"].bool()),
                        ("no_contact", ~data["contact_group"].bool()),
                        ("all", torch.ones(count, dtype=torch.bool))):
        selected = {key: value[group] for key, value in rows.items()}
        target = selected["gt"]
        epe = {"zero_object": target.norm(dim=-1).mean() * 1000}
        for kind in ("nominal", "oracle", "zero_hand_flow"):
            epe[kind] = (selected[kind] - target).norm(dim=-1).mean() * 1000
        nominal_norm = selected["nominal_flow_norm"].reshape(-1)
        oracle_norm = selected["oracle_flow_norm"].reshape(-1)
        report["groups"][name] = {
            "samples": int(group.sum()),
            "translation_epe_mm": {key: float(value) for key, value in epe.items()},
            "mean_predicted_translation_mm": {
                kind: float(selected[kind].norm(dim=-1).mean() * 1000)
                for kind in ("nominal", "oracle", "zero_hand_flow")},
            "hand_flow_mm": {
                "nominal_mean": float(nominal_norm.mean() * 1000),
                "nominal_p95": float(torch.quantile(nominal_norm, 0.95) * 1000),
                "oracle_mean": float(oracle_norm.mean() * 1000),
                "oracle_p95": float(torch.quantile(oracle_norm, 0.95) * 1000),
            },
            "token_active_fraction": {
                kind: float(selected["zero_active" if kind == "zero_hand_flow"
                                     else f"{kind}_active"].float().mean())
                for kind in ("nominal", "oracle", "zero_hand_flow")},
            "oracle_epe_reduction_vs_nominal_fraction": float(
                1 - epe["oracle"] / epe["nominal"]),
            "oracle_improvement_vs_zero_object_fraction": float(
                1 - epe["oracle"] / epe["zero_object"]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / "outputs/CmResidual" / ("agent_v150_oracle_handflow_smoke"
                                            if args.smoke else "agent_v150_oracle_handflow")
    if output.exists():
        raise FileExistsError(output)
    if (not CMV2.is_symlink() or CMV2.resolve() != EXTERNAL.resolve() or
            sha256(CMV2) != CMV2_SHA):
        raise ValueError("original Cmv2 checkpoint input drifted")
    for path, expected in TRANSITIONS.values():
        if sha256(path) != expected:
            raise ValueError(f"transition SHA mismatch: {path}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise ValueError("V1.50 must use only physical GPU5")
    used = subprocess.check_output(
        ["nvidia-smi", "--id=5", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    if int(used) > 1024:
        raise RuntimeError(f"GPU5 occupied before V1.50: {used} MiB")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    output.mkdir(parents=True)
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": output.name, "work_version": "V1.50",
                "git_commit": revision, "physical_gpu": 5,
                "cmv2_checkpoint_sha256": CMV2_SHA,
                "transition_sha256": {str(seed): item[1] for seed, item in TRANSITIONS.items()},
                "samples_per_seed_per_group": 2 if args.smoke else 64,
                "seed": 149, "microbatch": 2 if args.smoke else 8,
                "interaction_object_chunk": 32,
                "oracle_uses_future_next_q": True,
                "oracle_permitted_online": False,
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
        reports = {}
        for seed, (path, _) in TRANSITIONS.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema") != "ref2dex.cmlite_transition.v1":
                raise ValueError("transition schema mismatch")
            data = select(payload, seed, 2 if args.smoke else 64)
            reports[str(seed)] = evaluate_seed(data, bridge=bridge, cmv2=cmv2,
                                               lower=lower, upper=upper,
                                               microbatch=2 if args.smoke else 8)
            print(json.dumps({"seed": seed, "contact": reports[str(seed)]["groups"]["contact"]},
                             sort_keys=True), flush=True)
        gate = (not args.smoke and all(
            reports[str(seed)]["groups"]["contact"]["oracle_epe_reduction_vs_nominal_fraction"] >= 0.50 and
            reports[str(seed)]["groups"]["contact"]["oracle_improvement_vs_zero_object_fraction"] >= 0.20
            for seed in TRANSITIONS))
        result = {"schema": "ref2dex.v150_oracle_handflow_audit.v1",
                  "run_status": "COMPLETED", "work_version": "V1.50",
                  "is_smoke": args.smoke, "reports": reports,
                  "handflow_mismatch_strong_gate": gate,
                  "interpretation_limit": "oracle uses future next_q and is prohibited online"}
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
