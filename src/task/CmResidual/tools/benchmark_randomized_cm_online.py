"""Latency smoke for two-candidate six-region Cm on live-style GPU tensors."""
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

from src.task.CmResidual.cmlite import local_to_world_translation
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_cm_cate_ranking import (
    CALIBRATION, CALIBRATION_SHA256, CHECKPOINT_ROOT, CHECKPOINT_SHA256,
)
from src.task.CmResidual.tools.probe_intervention_handflow import INPUTS, sha256
from src.task.CmResidual.tools.probe_local_geometric_cm import (
    SixRegionCm, _surface_geometry_class, extract_features,
)
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
INPUT_KEY = "test_s148_d01"


def candidate_rows(source, count, device):
    payload = torch.load(INPUTS[INPUT_KEY][0], map_location="cpu", weights_only=False)
    values = payload["records"]
    chosen = (values["assignment"].reshape(-1) != 0).nonzero(as_tuple=False).reshape(-1)[:count]
    if len(chosen) != count:
        raise ValueError("not enough randomized current states")
    q = values["q"][chosen].to(device)
    velocity = values["dof_vel"][chosen].to(device)
    obj = values["object_state"][chosen].to(device)
    base = values["base_action"][chosen].to(device)
    minus, plus = base.clone(), base.clone()
    minus[:, 2] = (minus[:, 2] - .1).clamp(-1, 1)
    plus[:, 2] = (plus[:, 2] + .1).clamp(-1, 1)
    rows = {"q": torch.cat((q, q)), "dof_vel": torch.cat((velocity, velocity)),
            "object_state": torch.cat((obj, obj)),
            "action": torch.cat((minus, plus)),
            "target": torch.zeros(count * 2, 3, device=device),
            "category": torch.zeros(count * 2, dtype=torch.long, device=device)}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or not 10 <= args.repeats <= 100:
        raise ValueError("new output and 10-100 repeats required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(args.gpu_index):
        raise RuntimeError("one explicit matching physical GPU required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"physical GPU{visible} occupied: {memory}MiB")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "run_id": args.output.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "physical_gpu": args.gpu_index, "max_gpu_count": 1,
        "input_sha256": INPUTS[INPUT_KEY][1],
        "calibration_sha256": CALIBRATION_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256["geometric"],
        "repeats": args.repeats, "wall_budget_minutes": 5,
        "output_budget_mb": 1,
        "stop_rule": "input/checkpoint drift, occupied GPU, non-finite score or wall budget",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        if (sha256(INPUTS[INPUT_KEY][0]) != INPUTS[INPUT_KEY][1] or
                sha256(CALIBRATION) != CALIBRATION_SHA256 or
                sha256(CHECKPOINT_ROOT / "geometric.pt") != CHECKPOINT_SHA256["geometric"]):
            raise ValueError("input or model drift")
        device = torch.device("cuda:0")
        coefficients = torch.tensor(json.loads(CALIBRATION.read_text())
                                    ["coefficients"]["action_velocity"], device=device)
        calibration = {"alpha": coefficients[:, 0], "velocity": coefficients[:, 1],
                       "bias": coefficients[:, 2]}
        hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
        kinematics = TorchInspireKinematics(hand_urdf, device)
        geometry = _surface_geometry_class()(
            hand_urdf=hand_urdf, object_urdf=ASSETS / "mjcf/airplane.urdf",
            query_links=QUERY_LINKS, object_count=256, hand_count=256,
            seed=42, device=device)
        lower, upper = native_joint_limits(hand_urdf, device)
        model = SixRegionCm().to(device).eval()
        model.load_state_dict(torch.load(CHECKPOINT_ROOT / "geometric.pt",
                                         map_location=device, weights_only=False)["model"])
        reports = {}
        with torch.no_grad():
            for count in (16, 64):
                rows = candidate_rows(INPUT_KEY, count, device)
                times = []
                for repeat in range(args.repeats + 5):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    data = extract_features(rows, kinematics=kinematics, geometry=geometry,
                                            lower=lower, upper=upper, calibration=calibration,
                                            batch_size=count * 2)
                    local = model(data["region"], data["context"])["delta_local"]
                    world = local_to_world_translation(rows["object_state"], local)
                    effect = (world[count:, 2] - world[:count, 2]) * 1000
                    torch.cuda.synchronize()
                    if not torch.isfinite(effect).all():
                        raise FloatingPointError("non-finite online Cm candidate score")
                    if repeat >= 5:
                        times.append((time.perf_counter() - started) * 1000)
                sorted_times = sorted(times)
                reports[str(count)] = {
                    "environments": count, "candidates_per_env": 2,
                    "median_ms": statistics.median(times),
                    "p95_ms": sorted_times[int(.95 * (len(times) - 1))],
                    "max_ms": max(times),
                    "score_mean_mm": float(effect.mean()),
                }
                print(json.dumps(reports[str(count)], sort_keys=True), flush=True)
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps({"schema": "ref2dex.cm_online_latency.v1",
                                           "run_status": "COMPLETED", "reports": reports},
                                          indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED",
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
