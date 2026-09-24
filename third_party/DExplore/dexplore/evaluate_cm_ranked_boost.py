"""Matched actor/base, always-boost, and frozen-Cm boost evaluation arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - must precede torch
import torch

import evaluate_paired as pinned
from src.task.CmResidual.cm_online_boost import apply_wrist_z_boost
from src.task.CmResidual.cmlite import local_to_world_translation
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits
from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_cm_cate_ranking import (
    CALIBRATION, CALIBRATION_SHA256, CHECKPOINT_ROOT, CHECKPOINT_SHA256,
)
from src.task.CmResidual.tools.probe_local_geometric_cm import (
    SixRegionCm, _surface_geometry_class, extract_features,
)
from src.task.CmResidual.v118_planner import QUERY_LINKS, TorchInspireKinematics


original = pinned.original
BASE_PLAYER = original.EvalPlayer
CONFIG = None
THRESHOLD_MM = 12.060623168945312
FIRST_STEP, LAST_STEP, STRIDE = 50, 150, 2
DELTA_Z = .1


class BoostPlayer(BASE_PLAYER):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("Cm boost configuration missing")
        task = self.env.task
        self.boost_step = 0
        self.boost_scored_steps = 0
        self.boost_contact = torch.zeros((), dtype=torch.long, device=task._dof_pos.device)
        self.boost_selected = torch.zeros((), dtype=torch.long, device=task._dof_pos.device)
        self.boost_error = None
        self.mode = CONFIG["mode"]
        if self.mode == "cm":
            device = task._dof_pos.device
            coefficients = torch.tensor(json.loads(CALIBRATION.read_text())
                                        ["coefficients"]["action_velocity"], device=device)
            self.calibration = {"alpha": coefficients[:, 0],
                                "velocity": coefficients[:, 1],
                                "bias": coefficients[:, 2]}
            hand_urdf = ASSETS / "inspire_hand_new/inspire_hand_right.urdf"
            self.kinematics = TorchInspireKinematics(hand_urdf, device)
            self.geometry = _surface_geometry_class()(
                hand_urdf=hand_urdf, object_urdf=ASSETS / "mjcf/airplane.urdf",
                query_links=QUERY_LINKS, object_count=256, hand_count=256,
                seed=42, device=device)
            self.lower, self.upper = native_joint_limits(hand_urdf, device)
            self.cm = SixRegionCm().to(device).eval()
            self.cm.load_state_dict(torch.load(CHECKPOINT_ROOT / "geometric.pt",
                                               map_location=device, weights_only=False)["model"])

    @torch.no_grad()
    def _cm_selected(self, task, action, contact):
        ids = contact.nonzero(as_tuple=False).reshape(-1)
        result = torch.zeros(len(action), dtype=torch.bool, device=action.device)
        if not len(ids):
            return result
        count = len(ids)
        baseline = action[ids].detach()
        minus, plus = baseline.clone(), baseline.clone()
        minus[:, 2] = (minus[:, 2] - DELTA_Z).clamp(-1, 1)
        plus[:, 2] = (plus[:, 2] + DELTA_Z).clamp(-1, 1)
        q = task._dof_pos[ids]
        velocity = task._dof_vel[ids]
        obj = task._target_states[ids]
        rows = {"q": torch.cat((q, q)),
                "dof_vel": torch.cat((velocity, velocity)),
                "object_state": torch.cat((obj, obj)),
                "action": torch.cat((minus, plus)),
                "target": torch.zeros(count * 2, 3, device=q.device),
                "category": torch.zeros(count * 2, dtype=torch.long, device=q.device)}
        features = extract_features(rows, kinematics=self.kinematics,
                                    geometry=self.geometry, lower=self.lower,
                                    upper=self.upper, calibration=self.calibration,
                                    batch_size=count * 2)
        local = self.cm(features["region"], features["context"])["delta_local"]
        world = local_to_world_translation(rows["object_state"], local)
        score = (world[count:, 2] - world[:count, 2]) * 1000
        if not torch.isfinite(score).all():
            raise FloatingPointError("non-finite online Cm treatment score")
        result[ids] = score > THRESHOLD_MM
        self.boost_scored_steps += 1
        return result

    def env_step(self, env, action):
        self.boost_step += 1
        if (self.mode == "base" or not FIRST_STEP <= self.boost_step <= LAST_STEP or
                self.boost_step % STRIDE):
            return BASE_PLAYER.env_step(self, env, action)
        task = env.task
        contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                   (task._tar_contact_forces.norm(dim=-1) > .1) &
                   (task.reset_buf.reshape(-1) == 0))
        self.boost_contact += contact.sum()
        if self.mode == "always":
            chosen = contact
        else:
            chosen = self._cm_selected(task, action, contact)
        boosted = apply_wrist_z_boost(action, chosen, DELTA_Z)
        self.boost_selected += chosen.sum()
        return BASE_PLAYER.env_step(self, env, boosted)

    def run(self):
        status = "COMPLETED"
        try:
            return super().run()
        except BaseException as error:
            status = "FAILED"
            self.boost_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            summary = {"run_status": status, "mode": self.mode,
                       "global_steps": self.boost_step,
                       "contact_eligible": int(self.boost_contact),
                       "boost_selected": int(self.boost_selected),
                       "cm_scored_steps": self.boost_scored_steps,
                       "failure": self.boost_error}
            CONFIG["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_CM_BOOST " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--boost-mode", choices=("base", "always", "cm"), required=True)
    parser.add_argument("--boost-summary", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.boost_summary.exists():
        raise FileExistsError(args.boost_summary)
    checkpoint = Path(pinned.argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(pinned.argument_value(remaining, "--motion_file")).resolve()
    if (checkpoint != pinned.CHECKPOINT.resolve() or
            pinned.sha256(checkpoint) != pinned.CHECKPOINT_SHA256 or
            motion_root != pinned.MOTION_ROOT.resolve() or
            pinned.sha256(pinned.MOTION_MANIFEST) != pinned.MOTION_MANIFEST_SHA256):
        raise ValueError("pinned self-trained actor or motion source drift")
    for name, digest in (("calibration", CALIBRATION_SHA256),
                         ("geometric", CHECKPOINT_SHA256["geometric"])):
        source = CALIBRATION if name == "calibration" else CHECKPOINT_ROOT / "geometric.pt"
        if pinned.sha256(source) != digest:
            raise ValueError(f"{name} model input drift")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"physical GPU{visible} occupied: {memory}MiB")
    num_envs = int(pinned.argument_value(remaining, "--num_envs"))
    evaluation_output = Path(pinned.argument_value(remaining, "--output")).resolve()
    if not 1 <= num_envs <= 64:
        raise ValueError("online Probe limited to 1-64 env")
    args.boost_summary.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.boost_summary.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {
        "run_status": "STARTED", "run_id": args.boost_summary.parent.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=pinned.ROOT, text=True).strip(),
        "mode": args.boost_mode, "physical_gpu": int(visible), "max_gpu_count": 1,
        "evaluation_output": str(evaluation_output),
        "num_envs": num_envs, "checkpoint_sha256": pinned.CHECKPOINT_SHA256,
        "motion_manifest_sha256": pinned.MOTION_MANIFEST_SHA256,
        "calibration_sha256": CALIBRATION_SHA256,
        "geometric_sha256": CHECKPOINT_SHA256["geometric"],
        "window": [FIRST_STEP, LAST_STEP], "stride": STRIDE,
        "delta_z_action": DELTA_Z, "score_threshold_mm": THRESHOLD_MM,
        "wall_budget_minutes": 20, "output_budget_mb": 100,
        "stop_rule": "input drift, occupied GPU, non-finite score, evaluation failure or wall budget",
        "command": [sys.executable, *sys.argv],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"mode": args.boost_mode, "summary": args.boost_summary.resolve()}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = BoostPlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        summary_sha256=pinned.sha256(args.boost_summary),
                        evaluation_sha256=pinned.sha256(evaluation_output))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
