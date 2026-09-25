"""Evaluate a frozen, simulator-object-ID route over self-trained PPO actors."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - import before torch
import torch

import evaluate as original


ROOT = Path(__file__).resolve().parents[3]
CONFIG = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class RoutedPlayer(original.EvalPlayer):
    def __init__(self, config):
        super().__init__(config)
        self.expert_models = {}
        self.route_by_motion = None

    def restore(self, filename):
        super().restore(filename)
        target_keys = set(self.model.state_dict())
        for name, spec in CONFIG["experts"].items():
            path = ROOT / spec["checkpoint"]
            payload = torch.load(path, map_location=self.device, weights_only=False)
            model = copy.deepcopy(self.model)
            # rl_games may compile the training actor while the evaluation
            # actor remains eager (or vice versa). Normalize only that wrapper.
            state = {}
            for key, value in payload["model"].items():
                bare_key = key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key
                target_key = bare_key if bare_key in target_keys else "_orig_mod." + bare_key
                state[target_key] = value
            if set(state) != target_keys:
                raise ValueError(f"expert architecture differs: {name}")
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            rms = copy.deepcopy(self.running_mean_std)
            rms.load_state_dict(payload["running_mean_std"], strict=True)
            rms.eval()
            self.expert_models[name] = (model, rms)
        task = self.env.task
        names = list(self.expert_models)
        route = CONFIG["object_route"]
        motion_objects = [task.object_name[int(object_id)] for object_id in task.object_id]
        unknown = set(motion_objects) - set(route)
        if unknown:
            raise ValueError(f"unmapped objects: {sorted(unknown)}")
        self.route_by_motion = torch.tensor(
            [names.index(route[obj]) for obj in motion_objects],
            device=self.device, dtype=torch.long)
        self.expert_names = names
        print("REF2DEX_OBJECT_ROUTE " + json.dumps({
            "motion_objects": motion_objects,
            "motion_experts": [route[obj] for obj in motion_objects],
        }, sort_keys=True), flush=True)

    @torch.no_grad()
    def get_action(self, obs_dict, is_determenistic=False):
        if self.route_by_motion is None:
            raise RuntimeError("object route not initialized")
        choice = self.route_by_motion[self.env.task.data_id.long()]
        source_model, source_rms = self.model, self.running_mean_std
        selected = None
        try:
            for index in choice.unique().tolist():
                self.model, self.running_mean_std = self.expert_models[
                    self.expert_names[index]]
                candidate = super().get_action(obs_dict, is_determenistic)
                if selected is None:
                    selected = torch.zeros_like(candidate)
                mask = choice == index
                selected[mask] = candidate[mask]
        finally:
            self.model, self.running_mean_std = source_model, source_rms
        if selected is None or not torch.isfinite(selected).all():
            raise FloatingPointError("invalid routed action")
        return selected


def main() -> None:
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--route-config", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    config_path = args.route_config.resolve()
    CONFIG = json.loads(config_path.read_text())
    for name, spec in CONFIG["experts"].items():
        path = ROOT / spec["checkpoint"]
        if not path.is_file() or sha256(path) != spec["sha256"]:
            raise ValueError(f"expert checkpoint drift: {name}")
    if set(CONFIG["object_route"].values()) - set(CONFIG["experts"]):
        raise ValueError("route refers to missing expert")
    output = Path(remaining[remaining.index("--output") + 1]).resolve()
    if output.exists() or output.parent.exists():
        raise FileExistsError("new output directory required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical GPU required")
    used = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    memory = {int(line.split(",")[0]): int(line.split(",")[1])
              for line in used.splitlines()}[int(visible)]
    if memory > 1024:
        raise RuntimeError(f"GPU{visible} occupied: {memory} MiB")
    output.parent.mkdir(parents=True)
    manifest_path = output.parent / "run_manifest.json"
    manifest = {"run_status": "STARTED", "created_at": now(),
                "run_id": output.parent.name, "work_version": "multitrajectory-object-router-probe",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=ROOT, text=True).strip(),
                "route_config": str(config_path), "route_config_sha256": sha256(config_path),
                "physical_gpu": int(visible), "command": [sys.executable, *sys.argv],
                "checkpoint_roles": "self_trained_only", "cm_enabled": False,
                "budget": {"gpu_count": 1, "wall_minutes": 10, "output_mb": 50},
                "stop_rule": "input drift, GPU conflict, invalid route or incomplete evaluation"}
    write(manifest_path, manifest)
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = RoutedPlayer
    try:
        original.main()
        result = json.loads(output.read_text())
        summary = result["summary"]
        if summary["num_episodes"] != 64 or not summary["early_termination_disabled"]:
            raise ValueError("incomplete first-episode evaluation")
        manifest.update(run_status="COMPLETED", summary=summary)
        print("REF2DEX_ROUTER_RESULT " + json.dumps({
            "run_id": output.parent.name,
            "lift_success_rate": summary["lift_success_rate"],
        }, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = now()
        write(manifest_path, manifest)


if __name__ == "__main__":
    main()
