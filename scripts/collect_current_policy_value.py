#!/usr/bin/env python3
"""CPU preflight contract for the r7 current-policy value diagnostic.

``--preflight`` deliberately imports only the standard library and PyTorch.  A
future, separately authorized ``--collect`` invocation is the only path that
may import Isaac Gym.  The old r7 environment runner is read-only evidence;
its collect mode adds assignment noise, so this entry point never delegates
to that mode without the explicit actor-only sampling contract below.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


WORKER_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(os.environ.get("REF2DEX_SOURCE_ROOT", "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent")).resolve()
R7_ROOT = MAIN_ROOT / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7"
SOURCE_RUNNER = MAIN_ROOT / "scripts/run_cm_physical_value_environment.py"
SOURCE_ORIGIN_COMMIT = "f3ae0dfd6072f0d3cde8112f71574828b6a5a4bc"
SOURCE_COMMIT = SOURCE_ORIGIN_COMMIT
SOURCE_RUNNER_SHA256 = "1fc914f75b5579078eea533660a57f2c820414995da18cd7304858054ab26a57"
ENV_CONFIG = R7_ROOT / "environment.yaml"
TRAIN_CONFIG = R7_ROOT / "training.yaml"
MOTION_ROOT = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions")
MOTION_HASHES = {
    "s3_airplane_lift": "64d18ee69ea3fcb7ed76300aa2f5004383b7c01b29a30bc533f6960e3db5be1c",
    "s7_airplane_lift_Retake": "31cc18d4719e9e8d02f3c6d142d943a1eeedcdde3472ff690f0171090b374fc6",
    "s9_airplane_lift": "27e96c54756b84502cd56c6f801852b809de378a45b69f8de24f4b9899305c32",
}
CHECKPOINTS = {
    286: R7_ROOT / "train_cm_value_s286/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000420.pth",
    287: R7_ROOT / "train_cm_value_s287/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000420.pth",
}
EVAL_MANIFESTS = {
    286: R7_ROOT / "eval_cm_value_t286_e160_s289/run_manifest.json",
    287: R7_ROOT / "eval_cm_value_t287_e160_s289/run_manifest.json",
}
CHECKPOINT_SHA256 = {
    286: "0fe81f67f7b95356253edbdc755ccbc475ea77fb8a472ee99bb565dd6f327cf0",
    287: "afdb2cdf56d3350e18862d782d777e9aeb8a8133bf6690c440b43b9c8d5698bf",
}
ENV_SHA256 = "f66b3d16d685678ec24e757972b1521de96df6d51970a7abca2185fb376a37b9"
TRAIN_SHA256 = "a9b46c7bad03df3518cca07734fcb1a9d78f448ccf951ed119c999ff97eebf38"
DIAGNOSTIC_SEED = 290
NATIVE_EVAL_SEED = 289
ENVS_PER_CHECKPOINT = 96
MOTION_COUNTS = {0: 32, 1: 32, 2: 32}
MAX_ROWS_TOTAL = 192_000
MAX_ROWS_PER_CHECKPOINT = 96_000
MAX_OUTPUT_BYTES = 2 * 1024**3
MAX_WALL_SECONDS = 20 * 60
MAX_STEPS_PER_EPISODE = 700
STATE_DIM = 55
ACTION_DIM = 18
CONTEXT_DIM = 605
SCHEMA = "ref2dex.physical_value.v1"


def _torch():
    """Import torch only after the runtime wrapper has imported Isaac Gym."""
    import torch  # local import is intentional: --collect uses Isaac first
    return torch


@dataclass(frozen=True)
class PanelSpec:
    seed: int
    checkpoint: Path
    checkpoint_sha256: str
    native_manifest: Path
    episode_id_prefix: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flag(command: list[str], name: str) -> str:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"native manifest lacks {name}") from error


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise RuntimeError("PyYAML is required for CPU config preflight") from error
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _validate_native_panel(seed: int) -> dict[str, Any]:
    checkpoint = CHECKPOINTS[seed]
    manifest_path = EVAL_MANIFESTS[seed]
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing native checkpoint/manifest for {seed}")
    actual_sha = sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256[seed]:
        raise ValueError(f"checkpoint hash drift for seed {seed}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_status") != "COMPLETED" or manifest.get("mode") != "evaluate":
        raise ValueError(f"native evaluation manifest is not completed/evaluate: {manifest_path}")
    command = manifest.get("command")
    if not isinstance(command, list):
        raise ValueError("native manifest command missing")
    expected = {
        "--checkpoint": str(checkpoint),
        "--cfg_env": str(ENV_CONFIG),
        "--cfg_train": str(TRAIN_CONFIG),
        "--motion_file": str(MOTION_ROOT),
        "--num_envs": str(ENVS_PER_CHECKPOINT),
        "--seed": str(NATIVE_EVAL_SEED),
    }
    for flag, value in expected.items():
        if _flag(command, flag) != value:
            raise ValueError(f"native manifest {flag} mismatch for seed {seed}")
    if manifest.get("input_sha256") != actual_sha:
        raise ValueError(f"native manifest checkpoint SHA mismatch for seed {seed}")
    return {
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "native_physical_gpu": manifest.get("physical_gpu"),
        "native_completed_at": manifest.get("completed_at"),
    }


def _validate_motion_inputs() -> dict[str, str]:
    if not MOTION_ROOT.is_dir():
        raise FileNotFoundError(MOTION_ROOT)
    actual = {}
    for name, expected in MOTION_HASHES.items():
        path = MOTION_ROOT / name / "interaction_hand_inspire.pt"
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"motion hash drift: {path}")
        actual[name] = sha256(path)
    return actual


def sample_gaussian_action(mean: torch.Tensor, sigma: torch.Tensor,
                           generator: torch.Generator) -> torch.Tensor:
    """r7 PPO player sampling: Gaussian draw followed by original action clip."""
    torch = _torch()
    if mean.shape != sigma.shape or mean.shape[-1] != ACTION_DIM:
        raise ValueError("actor mean/sigma shape mismatch")
    if not torch.isfinite(mean).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("nonfinite or nonpositive Gaussian parameters")
    sampled = mean + sigma * torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
    return sampled.clamp(-1.0, 1.0)


def validate_episode_rows(rows: dict[str, Any]) -> dict[str, Any]:
    """Validate the fields Episodes.from_data needs without importing Isaac."""
    torch = _torch()
    required = {
        "state", "next_state", "context", "next_context", "action", "previous_action",
        "reward", "reward_components", "done", "terminate", "timeout", "episode_id",
        "env_id", "step", "motion_id", "start_frame", "checkpoint_seed",
        "value_at_state", "next_value", "global_tick", "horizon_phase",
    }
    missing = sorted(required - set(rows))
    if missing:
        raise ValueError(f"missing transition fields: {missing}")
    n = len(rows["step"])
    if not n:
        raise ValueError("empty transition batch")
    tensor_fields = ["state", "next_state", "context", "next_context", "action",
                     "previous_action", "reward", "reward_components", "done",
                     "terminate", "timeout", "episode_id", "env_id", "step",
                     "motion_id", "start_frame", "checkpoint_seed", "value_at_state",
                     "next_value", "global_tick", "horizon_phase"]
    for key in tensor_fields:
        value = rows[key]
        if len(value) != n:
            raise ValueError(f"length mismatch: {key}")
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite field: {key}")
    for key in ("episode_id", "env_id", "step", "motion_id", "start_frame", "checkpoint_seed",
                "global_tick", "horizon_phase"):
        if not isinstance(rows[key], torch.Tensor) or rows[key].dtype != torch.int64:
            raise ValueError(f"{key} must be int64")
    if rows["state"].shape != (n, STATE_DIM) or rows["next_state"].shape != (n, STATE_DIM):
        raise ValueError("state shape mismatch")
    if rows["context"].shape != (n, CONTEXT_DIM) or rows["next_context"].shape != (n, CONTEXT_DIM):
        raise ValueError("context shape mismatch")
    if rows["action"].shape != (n, ACTION_DIM) or rows["previous_action"].shape != (n, ACTION_DIM):
        raise ValueError("action shape mismatch")
    if not torch.allclose(rows["reward"], rows["reward_components"].sum(-1), atol=1e-5):
        raise ValueError("reward components do not sum to reward")
    if (rows["terminate"].bool() & ~rows["done"].bool()).any() or (rows["timeout"].bool() & ~rows["done"].bool()).any():
        raise ValueError("done reason without done")
    if ((rows["terminate"].bool() | rows["timeout"].bool()) != rows["done"].bool()).any():
        raise ValueError("done reason is incomplete")
    episode = rows["episode_id"].long()
    unique = torch.unique(episode, sorted=True)
    episode_count = 0
    for eid in unique.tolist():
        indices = (episode == eid).nonzero(as_tuple=False).reshape(-1)
        steps = rows["step"][indices].long()
        if steps[0].item() != 0 or not torch.equal(steps, torch.arange(len(indices))):
            raise ValueError(f"non-contiguous episode {eid}")
        if rows["done"][indices[:-1]].bool().any() or not rows["done"][indices[-1]].bool().item():
            raise ValueError(f"episode {eid} is not exactly terminated at its last row")
        if rows["env_id"][indices].unique().numel() != 1 or rows["motion_id"][indices].unique().numel() != 1:
            raise ValueError(f"episode {eid} changed env/motion")
        episode_count += 1
    return {"rows": n, "episodes": episode_count, "episode_ids": unique.tolist()}


def synthetic_contract_fixture() -> dict[str, Any]:
    """Terminal/reset and actor behavior fixture; CPU-only and deterministic."""
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(290)
    mean = torch.zeros(4, ACTION_DIM)
    sigma = torch.full_like(mean, 0.25)
    action = sample_gaussian_action(mean, sigma, generator)
    if not torch.isfinite(action).all() or action.abs().max() > 1:
        raise AssertionError("Gaussian action fixture violated clip")
    state = torch.zeros(4, STATE_DIM)
    next_state = state.clone()
    next_state[1, 0] = 7  # terminal post-state is retained, not reset in place
    next_state[3, 0] = 9
    context = torch.zeros(4, CONTEXT_DIM)
    reward_components = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
                                      [2., 0., 0., 0., 0.], [0., 0., 1., 0., 0.]])
    rows = {
        "state": state, "next_state": next_state, "context": context,
        "next_context": context.clone(), "action": action,
        "previous_action": torch.zeros_like(action), "reward": reward_components.sum(-1),
        "reward_components": reward_components, "done": torch.tensor([False, True, False, True]),
        "terminate": torch.tensor([False, True, False, True]),
        "timeout": torch.zeros(4, dtype=torch.bool), "episode_id": torch.tensor([100, 100, 101, 101]),
        "env_id": torch.tensor([0, 0, 1, 1]), "step": torch.tensor([0, 1, 0, 1]),
        "motion_id": torch.tensor([0, 0, 1, 1]), "start_frame": torch.zeros(4, dtype=torch.long),
        "checkpoint_seed": torch.tensor([286, 286, 287, 287]),
        "value_at_state": torch.tensor([0.5, 0.25, 0.75, 0.125]),
        "next_value": torch.tensor([0.25, 0.0, 0.125, 0.0]),
        "global_tick": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "horizon_phase": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }
    result = validate_episode_rows(rows)
    return {"validation": result, "sampled_action_sha256": hashlib.sha256(action.numpy().tobytes()).hexdigest(),
            "reset_boundary_preserved": bool(next_state[1, 0].item() == 7 and next_state[3, 0].item() == 9)}


def _source_contract(source_text: str) -> dict[str, Any]:
    old_noise = "action = action + noise[:, None] * torch.randn" in source_text
    old_sampling = "self.get_action(obs, ARGS.mode == \"evaluate\")" in source_text
    if not old_noise or not old_sampling:
        raise ValueError("unexpected r7 runner source; refuse silent policy drift")
    return {
        "old_runner_collect_injects_noise": old_noise,
        "old_runner_sampling_dispatch": "collect passes deterministic=False; evaluate passes deterministic=True",
        "new_entry_sampling": {"deterministic": False, "enable_eps_greedy": False,
                               "gaussian": True, "action_clip": [-1.0, 1.0], "extra_noise": False},
        "preprocessing": "DExplore player _preproc_obs with checkpoint running_mean_std; no replacement normalization",
        "correction": "new entry must not delegate to old PhysicalPlayer.run collect branch",
    }


def preflight(output_root: Path | None = None) -> dict[str, Any]:
    if not SOURCE_RUNNER.is_file() or sha256(SOURCE_RUNNER) != SOURCE_RUNNER_SHA256:
        raise ValueError("source runner commit/hash drift")
    current_head = subprocess.check_output(
        ["git", "-C", str(MAIN_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.call(
        ["git", "-C", str(MAIN_ROOT), "merge-base", "--is-ancestor", SOURCE_ORIGIN_COMMIT, current_head],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) != 0:
        raise ValueError("source origin commit is not an ancestor of current HEAD")
    for path, expected in ((ENV_CONFIG, ENV_SHA256), (TRAIN_CONFIG, TRAIN_SHA256)):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"config hash drift: {path}")
    motions = _validate_motion_inputs()
    panels = [_validate_native_panel(seed) for seed in (286, 287)]
    train_cfg = _load_yaml(TRAIN_CONFIG)
    if train_cfg["params"]["config"].get("enable_eps_greedy") is not False:
        raise ValueError("training config enable_eps_greedy is not false")
    if float(train_cfg["params"]["config"].get("gamma")) != 0.99:
        raise ValueError("training gamma drift")
    source_text = SOURCE_RUNNER.read_text()
    contract = _source_contract(source_text)
    fixture = synthetic_contract_fixture()
    if output_root is not None and output_root.exists():
        raise FileExistsError(f"new unique output root required: {output_root}")
    plan = {
        "diagnostic_seed": DIAGNOSTIC_SEED, "envs_per_checkpoint": ENVS_PER_CHECKPOINT,
        "motion_counts": MOTION_COUNTS, "start_frame": 0,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "max_rows_total": MAX_ROWS_TOTAL, "max_rows_per_checkpoint": MAX_ROWS_PER_CHECKPOINT,
        "max_output_bytes": MAX_OUTPUT_BYTES, "max_wall_seconds": MAX_WALL_SECONDS,
        "episode_id": "checkpoint-specific prefix s{training_seed}_e420_diag290_env{env_id}; globally disjoint",
        "storage": "Episodes-compatible tensors; export only after all 96 episodes are complete and finite",
        "commands": [render_command(seed, output_root or (WORKER_ROOT / "src/task/CmResidual/research/physical_value/output/P-20261001-current-policy-value-r7")) for seed in (286, 287)],
    }
    return {
        "status": "READY_FOR_ROOT_RUNTIME_TASK", "schema": SCHEMA,
        "source_origin_commit": SOURCE_ORIGIN_COMMIT, "source_current_head": current_head,
        "source_commit": SOURCE_COMMIT, "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "environment_sha256": ENV_SHA256, "training_sha256": TRAIN_SHA256,
        "motion_hashes": motions, "panels": panels, "source_contract": contract,
        "plan": plan, "cpu_fixture": fixture, "isaac_imported": False,
        "gpu_started": False, "collection_started": False, "training_started": False,
    }


def render_command(seed: int, output_root: Path) -> str:
    return (f"python3 scripts/collect_current_policy_value.py --collect --checkpoint-seed {seed} "
            f"--run-root {output_root.resolve()}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-json", type=Path)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--mock", action="store_true", help="run the Isaac-free contract harness")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--checkpoint-seed", type=int, choices=(286, 287))
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args(argv)
    if args.collect:
        # Keep this thin entry Isaac-free: the runtime wrapper imports Isaac
        # before torch in its own process.  The mock path exercises the same
        # downstream Episodes contract without touching a simulator.
        runtime = WORKER_ROOT / "scripts/run_current_policy_value_environment.py"
        if args.checkpoint_seed is None or args.run_root is None:
            parser.error("--collect requires --checkpoint-seed and --run-root")
        command = [sys.executable, str(runtime), "--collect",
                   "--checkpoint-seed", str(args.checkpoint_seed),
                   "--run-root", str(args.run_root)]
        if args.mock:
            command.append("--mock")
        elif args.gpu_index is not None:
            command.extend(["--gpu-index", str(args.gpu_index)])
        return os.spawnv(os.P_WAIT, sys.executable, command)
    if not args.preflight:
        parser.error("--preflight is required for this task")
    result = preflight(args.run_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.preflight_json:
        if args.preflight_json.exists():
            raise FileExistsError(args.preflight_json)
        args.preflight_json.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_json.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
