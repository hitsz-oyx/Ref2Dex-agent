#!/usr/bin/env python3
"""Export first-episode source trajectories, or evaluate a matched BC actor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
sys.path.insert(0, str(ROOT))
from isaacgym import gymapi  # import before torch
import torch
import evaluate as original
from src.task.CmResidual.cm_bottleneck_student import BottleneckStudent, PhysicalFeatures

EXPECTED_SOURCE = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
ARGS = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StudentPlayer(original.EvalPlayer):
    def restore(self, filename):
        super().restore(filename)
        self.student = None
        self.last_observation = None
        self.last_source_action = None
        if ARGS.student is not None:
            checkpoint = torch.load(ARGS.student, map_location=self.device, weights_only=False)
            self.student = BottleneckStudent().to(self.device)
            self.student.load_state_dict(checkpoint["state_dict"], strict=True)
            self.student.eval().requires_grad_(False)
            self.features = PhysicalFeatures(checkpoint["physical_checkpoint"], checkpoint["arm"]).to(self.device)
            self.input_mean = checkpoint["mean"].to(self.device)
            self.input_std = checkpoint["std"].to(self.device)

    @torch.no_grad()
    def get_action(self, obs_dict, is_determenistic=False):
        # Same deterministic self-trained source and Cm compute for every arm.
        source_action = super().get_action(obs_dict, True)
        obs = obs_dict["obs"]
        self.last_observation = obs.detach().clone()
        self.last_source_action = source_action.detach().clone()
        if self.student is None:
            return source_action
        extra = self.features(obs, source_action)
        values = torch.cat([obs, extra], dim=1)
        action = self.student(((values - self.input_mean) / self.input_std).clamp(-10, 10)).clamp(-1, 1)
        if not torch.isfinite(action).all():
            raise FloatingPointError("nonfinite student action")
        return action

    def _record_transition(self, **values):
        values["observation"] = self.last_observation
        values["source_action"] = self.last_source_action
        values["env_id"] = torch.arange(self.last_observation.shape[0], device=self.device)[:, None]
        super()._record_transition(**values)


def main():
    global ARGS
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--student-sha256")
    parser.add_argument("--run-dir", type=Path, required=True)
    ARGS, remaining = parser.parse_known_args()
    if ARGS.run_dir.exists():
        raise FileExistsError(ARGS.run_dir)
    source = Path(remaining[remaining.index("--checkpoint") + 1])
    if sha(source) != EXPECTED_SOURCE:
        raise ValueError("self-trained source hash drift")
    if (ARGS.student is None) != (ARGS.student_sha256 is None):
        raise ValueError("student/hash must be paired")
    if ARGS.student is not None and sha(ARGS.student) != ARGS.student_sha256:
        raise ValueError("student hash drift")
    if "--disable-early-termination" not in remaining:
        raise ValueError("full first episodes required")
    ARGS.run_dir.mkdir(parents=True)
    manifest = dict(run_status="STARTED", created_at=datetime.now(timezone.utc).isoformat(),
                    command=[sys.executable, *sys.argv], source_sha256=EXPECTED_SOURCE,
                    student_sha256=ARGS.student_sha256, physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    path = ARGS.run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = StudentPlayer
    try:
        original.main()
        output = Path(remaining[remaining.index("--output") + 1])
        result = json.loads(output.read_text())
        if result["summary"]["num_episodes"] != 64:
            raise ValueError("incomplete physical evaluation")
        manifest.update(run_status="COMPLETED", summary=result["summary"])
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
