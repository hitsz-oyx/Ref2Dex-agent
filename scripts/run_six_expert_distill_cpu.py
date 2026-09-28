#!/usr/bin/env python3
"""CPU preflight and reduced-state imitation probe for T-20260928.

This script is deliberately offline.  It reads frozen checkpoints and existing
transition exports, writes only a new task output directory, and never starts
Isaac Gym, PPO, a collector, or a GPU process.  The reduced 49-D state probe is
an implementation wiring check; it is not deployment-equivalent to the 1442-D
observation policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn


TASK_ID = "T-20260928-p0-six-expert-distillation"
EXPERIMENT_ID = "P-20260928-six-expert-distillation"
REPO = Path(__file__).resolve().parents[1]
BASELINE = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline")
ROUTE = REPO / "src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json"
C1_RESULTS = REPO / "docs/experiments/validations/VAL-20260926-observation-six-expert-c1-results.json"

TRANSITIONS = {
    "balanced_e360": BASELINE / "outputs/Dexplore/agent_crossobject_train5_balanced_s179_e360/eval_s178_e360_full_coverage_train5/transitions.pt",
    "source_e260": BASELINE / "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/eval_s95_e260_full_v142_cm_audit/transitions.pt",
    "train5_e320": BASELINE / "outputs/Dexplore/agent_crossobject_train5_s179_e320/eval_s178_e320_full_coverage_train5/transitions.pt",
}
REQUIRED_TRANSITION_KEYS = {
    "q": (18,),
    "dof_vel": (18,),
    "object_state": (13,),
    "action": (18,),
    "next_q": (18,),
    "next_object_state": (13,),
    "done": (1,),
    "progress": (1,),
    "data_id": (1,),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_meta(path: Path, expected: str) -> dict:
    result = {"path": str(path), "exists": path.is_file(), "expected_sha256": expected}
    if not path.is_file():
        return result | {"sha256": None, "match": False}
    result["sha256"] = sha256(path)
    result["match"] = result["sha256"] == expected
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = payload.get("model", {})
    result["epoch"] = payload.get("epoch")
    result["model_keys"] = sorted(model)
    result["model_shapes"] = {key: list(value.shape) for key, value in model.items() if hasattr(value, "shape")}
    rms = payload.get("running_mean_std", {})
    result["running_mean_shape"] = list(rms["running_mean"].shape) if "running_mean" in rms else None
    result["running_var_shape"] = list(rms["running_var"].shape) if "running_var" in rms else None
    return result


def resolve_checkpoint(relative: str) -> Path:
    candidates = [REPO / relative, BASELINE / relative]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def transition_meta(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result | {"schema": None, "rows": 0, "valid": False, "missing": sorted(REQUIRED_TRANSITION_KEYS)}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result["schema"] = payload.get("schema")
    result["rows"] = int(payload["action"].shape[0]) if "action" in payload else 0
    result["sha256"] = sha256(path)
    missing = sorted(set(REQUIRED_TRANSITION_KEYS) - set(payload))
    shape_errors = {}
    finite = True
    for key, suffix in REQUIRED_TRANSITION_KEYS.items():
        value = payload.get(key)
        if value is None:
            continue
        if tuple(value.shape[1:]) != suffix:
            shape_errors[key] = [list(value.shape[1:]), list(suffix)]
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            finite = False
    result["missing"] = missing
    result["shape_errors"] = shape_errors
    result["finite"] = finite
    result["data_ids"] = sorted(int(value) for value in payload["data_id"].unique().tolist()) if "data_id" in payload else []
    result["full_observation_present"] = "observation" in payload
    result["valid"] = bool(result["schema"] == "ref2dex.cmlite_transition.v1" and not missing and not shape_errors and finite)
    return result


def build_preflight(output: Path) -> dict:
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    expert_meta = {}
    for name, spec in route["experts"].items():
        expert_meta[name] = checkpoint_meta(resolve_checkpoint(spec["checkpoint"]), spec["sha256"])
    model_keys = [tuple(item.get("model_keys", [])) for item in expert_meta.values()]
    model_shapes = [item.get("model_shapes", {}) for item in expert_meta.values()]
    architecture_match = len(set(model_keys)) == 1 and len(set(json.dumps(item, sort_keys=True) for item in model_shapes)) == 1
    data_meta = {name: transition_meta(path) for name, path in TRANSITIONS.items()}
    coverage = {name: name in data_meta and data_meta[name].get("valid", False) for name in route["experts"]}
    missing_roles = sorted(name for name, present in coverage.items() if not present)
    full_observation = all(item.get("full_observation_present", False) for item in data_meta.values() if item.get("valid"))
    contract = {
        "six_checkpoints_present_and_hash_matched": all(item.get("match", False) for item in expert_meta.values()),
        "six_checkpoint_architecture_match": architecture_match,
        "transition_schema_valid": all(item.get("valid", False) for item in data_meta.values()),
        "full_1442_observation_in_transition_exports": full_observation,
        "per_expert_action_exports_for_all_six_roles": not missing_roles,
        "missing_expert_roles": missing_roles,
    }
    cpu_contract = {
        "input": {"q": 18, "dof_vel": 18, "object_state": 13, "total": 49},
        "target": 18,
        "student": {"type": "mlp", "hidden": [128, 128], "activation": "relu"},
        "split": "deterministic first 80 percent per existing transition file; remainder held out",
        "device": "cpu",
        "deployment_equivalent": False,
    }
    gpu_go = all(contract.values())
    preflight = {
        "schema": "ref2dex.distillation_preflight.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at_unix": time.time(),
        "repository": str(REPO),
        "branch": os.popen("git branch --show-current").read().strip(),
        "head": os.popen("git rev-parse HEAD").read().strip(),
        "main_head_observed": os.popen("git rev-parse main").read().strip(),
        "sync_status": "BLOCKED_READ_ONLY_LINKED_GIT_METADATA",
        "sync_error": "git merge --ff-only main failed while writing ORIG_HEAD; no files changed",
        "route_config": {"path": str(ROUTE), "sha256": sha256(ROUTE), "experts": route["experts"], "object_route": route["object_route"]},
        "experts": expert_meta,
        "rollout_data": data_meta,
        "contract": contract,
        "cpu_wiring_contract": cpu_contract,
        "decision": {"cpu_offline_probe": "GO_REDUCED_STATE_ONLY", "gpu_short_probe": "GO" if gpu_go else "NO_GO", "label_before_gpu": "UNCLEAR"},
        "stop_reason": "full distillation contract is incomplete: transition exports lack observation and per-expert action coverage is 3/6",
        "output_dir": str(output),
    }
    json_dump(output / "preflight.json", preflight)
    return preflight


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(49, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 18))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def cpu_probe(output: Path) -> dict:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.manual_seed(20260928)
    random.seed(20260928)
    xs, ys, tags = [], [], []
    source_stats = {}
    max_rows = 8192
    for tag, path in TRANSITIONS.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing = sorted(set(REQUIRED_TRANSITION_KEYS) - set(payload))
        if missing:
            source_stats[tag] = {
                "rows_used": 0,
                "rows_available": int(payload["action"].shape[0]) if "action" in payload else 0,
                "data_ids": sorted(int(v) for v in payload["data_id"].unique().tolist()) if "data_id" in payload else [],
                "usable": False,
                "missing": missing,
            }
            continue
        rows = min(max_rows, int(payload["action"].shape[0]))
        index = torch.linspace(0, payload["action"].shape[0] - 1, rows).round().long()
        x = torch.cat([payload["q"], payload["dof_vel"], payload["object_state"]], dim=1).float()[index]
        y = payload["action"].float()[index]
        xs.append(x)
        ys.append(y)
        tags.extend([tag] * rows)
        source_stats[tag] = {"rows_used": rows, "rows_available": int(payload["action"].shape[0]), "data_ids": sorted(int(v) for v in payload["data_id"].unique().tolist()), "usable": True}
    x = torch.cat(xs)
    y = torch.cat(ys)
    n = x.shape[0]
    split = int(n * 0.8)
    order = torch.arange(n)
    train_idx, test_idx = order[:split], order[split:]
    mean = x[train_idx].mean(0)
    std = x[train_idx].std(0).clamp_min(1e-6)
    model = Student()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(30):
        permutation = train_idx[torch.randperm(len(train_idx))]
        for batch in permutation.split(256):
            prediction = model((x[batch] - mean) / std)
            loss = loss_fn(prediction, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model((x - mean) / std)
    def metrics(indices: torch.Tensor) -> dict:
        target = y[indices]
        pred = prediction[indices]
        mse = torch.mean((pred - target) ** 2).item()
        cosine = torch.nn.functional.cosine_similarity(pred, target, dim=1).mean().item()
        return {"rows": int(len(indices)), "mse": mse, "rmse": math.sqrt(mse), "cosine": cosine, "target_norm": target.norm(dim=1).mean().item(), "prediction_norm": pred.norm(dim=1).mean().item()}
    result = {
        "schema": "ref2dex.distillation_cpu_probe.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "device": "cpu",
        "rows_total": int(n),
        "train": metrics(train_idx),
        "held_out": metrics(test_idx),
        "constant_mean_baseline_held_out_mse": torch.mean((y[test_idx] - y[train_idx].mean(0)) ** 2).item(),
        "source_data": source_stats,
        "student_capacity": {"input_dim": 49, "hidden": [128, 128], "output_dim": 18, "parameters": sum(p.numel() for p in model.parameters())},
        "interpretation": "CPU wiring only; reduced q+dof_vel+object_state input, not a 1442-D deployment student and not evidence for a scientific grasp claim",
    }
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "config": result["student_capacity"]}, output / "student_cpu.pt")
    json_dump(output / "cpu_probe.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    preflight = build_preflight(args.output)
    result = cpu_probe(args.output)
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    run_manifest = {
        "schema": "ref2dex.distillation_run_manifest.v1",
        "task_id": TASK_ID,
        "experiment_id": EXPERIMENT_ID,
        "run_id": args.output.name,
        "run_status": "COMPLETED",
        "device": "cpu",
        "gpu_count": 0,
        "ppo": False,
        "online": False,
        "collector": False,
        "input_hashes": {
            "route_config": preflight["route_config"]["sha256"],
            "checkpoints": {name: item.get("sha256") for name, item in preflight["experts"].items()},
            "transitions": {name: item.get("sha256") for name, item in preflight["rollout_data"].items()},
        },
        "output_hashes": output_hashes,
        "metrics_file": "cpu_probe.json",
        "preflight_file": "preflight.json",
        "stop_reason": preflight["stop_reason"],
    }
    json_dump(args.output / "run_manifest.json", run_manifest)
    summary = {"preflight_decision": preflight["decision"], "cpu_probe": result, "run_manifest": run_manifest, "created_files": sorted(str(p) for p in args.output.iterdir())}
    json_dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
