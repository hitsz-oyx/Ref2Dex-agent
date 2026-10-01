#!/usr/bin/env python3
"""Lazy-Isaac runtime for the current-policy value diagnostic.

The argument parser and CPU mock stay Isaac-free.  The real ``--collect``
branch imports Isaac Gym before importing torch, then reuses the read-only HF08
runner's environment construction while replacing only its player loop.  The
replacement removes the old assignment noise and records the actor forward's
value, phase, and complete transition contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any


WORKER_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("REF2DEX_SOURCE_ROOT", "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent")).resolve()
SOURCE_RUNNER = SOURCE_ROOT / "scripts/run_cm_physical_value_environment.py"
OUTPUT_BASE = WORKER_ROOT / "src/task/CmResidual/research/physical_value/output"
RUNTIME_MAX_OUTPUT_BYTES = 2 * 1024**3
MAX_ROWS_TOTAL = 192_000
MAX_ROWS_PER_CHECKPOINT = 96_000
MAX_WALL_SECONDS = 1200
MAX_STEPS = 700
ENVS = 96
DIAGNOSTIC_SEED = 290
CHECKPOINTS = {
    286: ("0fe81f67f7b95356253edbdc755ccbc475ea77fb8a472ee99bb565dd6f327cf0",
          "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/train_cm_value_s286/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000420.pth"),
    287: ("afdb2cdf56d3350e18862d782d777e9aeb8a8133bf6690c440b43b9c8d5698bf",
          "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/train_cm_value_s287/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000420.pth"),
}
MOTION_ROOT = "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions"
ENV_CONFIG = "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/environment.yaml"
TRAIN_CONFIG = "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/training.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _validate_value_config(config: Any) -> None:
    """Accept rl_games mappings and DefaultRewardsShaper objects at unit scale."""
    if not isinstance(config, Mapping):
        raise TypeError("player config must be a Mapping")
    if config.get("normalize_value", False):
        raise ValueError("normalize_value must remain false")
    shaper = config.get("reward_shaper", 1)
    if isinstance(shaper, Mapping):
        scale = shaper.get("scale_value", 1)
    else:
        scale = getattr(shaper, "scale_value", None)
    if scale is None or float(scale) != 1.0:
        raise ValueError("reward scale must remain 1")


def _player_obs(player: Any, observation: Any) -> Any:
    """Apply installed preprocessing to tensor or the env's outer ``{obs: tensor}``."""
    if isinstance(observation, Mapping) and set(observation) == {"obs"}:
        observation = observation["obs"]
    return player._preproc_obs(observation)


def _clip_player_action(player: Any, action: Any) -> Any:
    """Match rl_games PpoPlayerContinuous clamp then action-space rescaling."""
    import torch
    clipped = torch.clamp(action, -1.0, 1.0)
    if not getattr(player, "clip_actions", True):
        return action
    low = getattr(player, "actions_low", None)
    high = getattr(player, "actions_high", None)
    if low is None or high is None:
        return clipped
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    return clipped * d + m


def _native_actor_forward(player: Any, observation: Any) -> tuple[Any, Any, Any]:
    """Use the installed ModelA2CContinuousLogStd player call exactly once.

    The model performs running-mean/std normalization and converts raw logstd to
    sigma before its one Gaussian draw.
    """
    import torch
    obs = _player_obs(player, observation)
    input_dict = {"is_train": False, "prev_actions": None, "obs": obs,
                  "rnn_states": player.states}
    with torch.no_grad():
        result = player.model(input_dict)
    for key in ("actions", "values", "rnn_states"):
        if key not in result:
            raise KeyError(f"native player result lacks {key}")
    player.states = result["rnn_states"]
    action = _clip_player_action(player, result["actions"])
    value = result["values"].reshape(-1)
    if not torch.isfinite(action).all() or not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite native actor action/value")
    return action, value, result


def _native_critic_value(player: Any, observation: Any) -> Any:
    """Evaluate the native critic with identical normalization and no sampling."""
    import torch
    obs = _player_obs(player, observation)
    model = player.model
    # Critic inspection is inference only; do not retain an autograd graph.
    with torch.no_grad():
        return _native_critic_from_preprocessed(player, obs)


def _native_critic_from_preprocessed(player: Any, obs: Any) -> Any:
    import torch
    model = player.model
    normalized = model.norm_obs(obs) if hasattr(model, "norm_obs") else obs
    critic = getattr(model, "eval_critic", None)
    if critic is not None:
        value = critic(normalized)
    else:
        network = getattr(model, "a2c_network", None)
        critic = getattr(network, "eval_critic", None) if network is not None else None
        if critic is not None:
            value = critic(normalized)
        elif network is not None:
            # rl_games 1.x exposes the critic only through the raw network
            # tuple; invoking it is deterministic and does not draw actions.
            raw = network({"obs": normalized, "rnn_states": player.states})
            if not isinstance(raw, (tuple, list)) or len(raw) < 3:
                raise TypeError("native network has no critic-only path")
            value = raw[2]
        else:
            raise TypeError("native model has no critic-only path")
    value = value.reshape(-1)
    if getattr(player, "normalize_value", False):
        raise ValueError("normalize_value must remain false")
    if not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite native next critic value")
    return value


def _load_contract():
    path = WORKER_ROOT / "scripts/collect_current_policy_value.py"
    spec = importlib.util.spec_from_file_location("current_policy_value_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_run_root(root: Path, *, mock: bool = False) -> Path:
    root = root.absolute()
    if mock:
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        return root
    base = OUTPUT_BASE.resolve()
    if root.parent.resolve() != base or root.is_symlink():
        raise ValueError("run root must be a direct, non-symlink child of task research output")
    if root.exists() and not root.is_dir():
        raise ValueError("run root is not a directory")
    return root


def _prepare_root(root: Path, seed: int, contract: Any) -> tuple[Path, dict[str, Any]]:
    root = _safe_run_root(root)
    root_manifest = root / "runtime_root.json"
    parent = {
        "schema": "ref2dex.current_policy_value.runtime_root.v1",
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "source_runner_sha256": contract.SOURCE_RUNNER_SHA256,
        "environment_sha256": contract.ENV_SHA256,
        "training_sha256": contract.TRAIN_SHA256,
        "motion_hashes": contract.MOTION_HASHES,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_output_bytes": RUNTIME_MAX_OUTPUT_BYTES,
        "max_rows_total": MAX_ROWS_TOTAL,
    }
    if root.exists():
        if not root_manifest.is_file():
            raise FileExistsError("existing root lacks matching runtime provenance")
        previous = json.loads(root_manifest.read_text())
        for key, value in parent.items():
            if previous.get(key) != value:
                raise ValueError("runtime root provenance drift")
        parent = previous
    else:
        parent["workflow_started_epoch"] = time.time()
        root.mkdir()
        root_manifest.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n")
    run_dir = root / f"checkpoint_s{seed}_e420"
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    provenance = dict(parent, checkpoint_seed=seed,
                      checkpoint_sha256=contract.CHECKPOINT_SHA256[seed])
    return run_dir, provenance


def _gpu_ownership(index: int) -> dict[str, Any]:
    try:
        table = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi ownership query failed") from error
    devices = {}
    for line in table.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3:
            devices[int(fields[0])] = {"uuid": fields[1], "memory_used_mib": int(fields[2])}
    if index not in devices:
        raise RuntimeError(f"GPU index {index} not found")
    uuid = devices[index]["uuid"]
    owners = []
    for line in apps.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[0] == uuid:
            owners.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2] if len(fields) > 2 else ""})
    if owners or devices[index]["memory_used_mib"] > 2048:
        raise RuntimeError(f"GPU ownership conflict: {devices[index]}, owners={owners}")
    return {"physical_gpu": index, "gpu_uuid": uuid, "memory_used_mib": devices[index]["memory_used_mib"], "compute_owners": owners}


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _existing_rows(root: Path) -> int:
    total = 0
    for result_path in root.glob("checkpoint_s*_e420/results.json"):
        result = json.loads(result_path.read_text())
        if result.get("run_status") == "COMPLETED":
            total += int(result.get("rows", 0))
    return total


def _mock_collect(run_root: Path, seed: int) -> dict[str, Any]:
    """Exercise the real runtime loop with an Isaac-free vector-environment double."""
    import types
    torch = __import__("torch")
    contract = _load_contract()
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from src.task.CmResidual.physical_value_contract import HoldTracker, validate_rows
    from src.task.CmResidual.physical_value_data import Episodes

    root = _safe_run_root(run_root, mock=True)
    run_dir = root / f"checkpoint_s{seed}_e420"
    run_dir.mkdir()
    n_env = ENVS
    motion = torch.arange(n_env, dtype=torch.long).remainder(3)

    class MockTask:
        dt = 1 / 30
        num_envs = n_env

        def __init__(self):
            self._dof_pos = torch.zeros(n_env, 18)
            self._target_states = torch.zeros(n_env, 43)
            self._target_states[:, 39] = 1
            self.start_times = torch.zeros(n_env, dtype=torch.long)
            self.data_id = motion.clone()
            self.progress_buf = torch.zeros(n_env, dtype=torch.long)
            self._hybrid_init_prob = 1.0
            self.motion_file = ["s3_airplane_lift", "s7_airplane_lift_Retake", "s9_airplane_lift"]

    class MockEnv:
        def __init__(self, task):
            self.task = task

    class MockNetwork:
        def __call__(self, inputs):
            batch = inputs["obs"].shape[0]
            mu = torch.zeros(batch, 18)
            # Native ModelA2CContinuousLogStd expects raw logstd.
            logstd = torch.full_like(mu, torch.log(torch.tensor(.2)))
            value = torch.full((batch, 1), .5)
            return mu, logstd, value, inputs.get("rnn_states")

        def eval_critic(self, obs):
            return torch.full((obs.shape[0], 1), .5)

    class MockPlayer:
        def __init__(self, task):
            self.env = MockEnv(task)
            self.device = torch.device("cpu")
            self.states = None
            self.is_rnn = False
            self.has_batch_dimension = True
            self.normalize_value = False
            self.config = {"normalize_value": False, "reward_shaper": {"scale_value": 1}}
            from rl_games.algos_torch.models import ModelA2CContinuousLogStd
            self.model = ModelA2CContinuousLogStd.Network(
                MockNetwork(), obs_shape=(1442,), normalize_value=False,
                normalize_input=False, value_size=1)

        def get_batch_size(self, obs, default):
            return int(obs.shape[0]) if hasattr(obs, "shape") else default

        def _preproc_obs(self, obs):
            return obs.float()

        def env_reset(self, env_ids=None):
            if env_ids is not None:
                ids = torch.as_tensor(env_ids, dtype=torch.long)
                if ids.numel():
                    self.env.task.progress_buf[ids] = 0
            return {"obs": torch.zeros(n_env, 1442)}

        def env_step(self, env, action):
            task = env.task
            task.progress_buf += 1
            task._target_states[:, 2] = .03
            done = task.progress_buf >= 33
            return torch.zeros(n_env, 1442), torch.ones(n_env), done, {"terminate": done.clone()}

    class MockBridge:
        def __init__(self, **kwargs):
            pass

        def current(self, q, target):
            points = torch.zeros(q.shape[0], 1, 3)
            return types.SimpleNamespace(hand_points=points, object_points=points)

    mock_source = types.ModuleType("ref2dex_hf08_physical_runner")
    mock_source.torch = torch
    mock_source.ARGS = types.SimpleNamespace(
        seed_namespace=seed, run_dir=run_dir,
        run_root=root, workflow_started_epoch=time.time(), workflow_rows_before=0,
        checkpoint_sha256=contract.CHECKPOINT_SHA256[seed])
    mock_source.ROOT = SOURCE_ROOT
    mock_source.SCHEMA = contract.SCHEMA
    mock_source.HoldTracker = HoldTracker
    mock_source.DExploreCmv2GeometryBridge = MockBridge
    mock_source.ApproachConfig = lambda: types.SimpleNamespace()
    mock_source.sampled_surface_gap = lambda hand, obj, approach: torch.zeros(n_env)
    mock_source.snapshot = lambda task, tracker: torch.cat(
        (task.progress_buf.float()[:, None].expand(-1, 39),
         torch.ones(n_env, 1), torch.zeros(n_env, 15)), dim=1)
    mock_source.context = lambda task, tracker: torch.zeros(n_env, 605)
    mock_source.contacts = lambda task: torch.ones(n_env, 1)

    def reward(task, tracker, base, before_z, gap_before, gap_after, gamma, approach):
        components = torch.zeros(n_env, 5)
        components[:, 0] = base
        return base, components

    mock_source.shared_reward = reward
    mock_source.validate_rows = validate_rows
    previous = sys.modules.get("ref2dex_hf08_physical_runner")
    sys.modules["ref2dex_hf08_physical_runner"] = mock_source
    try:
        torch.manual_seed(290)
        _runtime_player_run(MockPlayer(MockTask()))
    finally:
        if previous is None:
            sys.modules.pop("ref2dex_hf08_physical_runner", None)
        else:
            sys.modules["ref2dex_hf08_physical_runner"] = previous

    result = json.loads((run_dir / "results.json").read_text())
    Episodes([run_dir], gamma=.99)
    return {"run_dir": str(run_dir), "rows": result["rows"],
            "episodes_compatible": True, "phase_max": 32, "bytes": _dir_bytes(run_dir)}


def _load_hf08_source():
    spec = importlib.util.spec_from_file_location("ref2dex_hf08_physical_runner", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_player_run(self):
    """HF08 PhysicalPlayer loop with actor-only sampling and full provenance."""
    source = sys.modules["ref2dex_hf08_physical_runner"]
    contract = _load_contract()
    torch = source.torch
    args = source.ARGS
    task = self.env.task
    task._enable_early_termination = False
    task._adaptive_kappa_enabled = False
    task._hybrid_init_prob = 1.0
    if abs(task.dt - 1 / 30) > 1e-8 or task.num_envs != ENVS:
        raise ValueError("HF08 runtime environment contract drift")
    _validate_value_config(self.config)
    asset = source.ROOT / "third_party/DExplore/dexplore/data/assets"
    bridge = source.DExploreCmv2GeometryBridge(
        hand_urdf=asset / "inspire_hand_new/inspire_hand_right.urdf",
        object_urdf=asset / "mjcf/airplane.urdf", device=task._dof_pos.device)
    approach = source.ApproachConfig()
    num = task.num_envs
    self.has_batch_dimension = True
    obs = self.env_reset(torch.arange(num, device=self.device))
    if self.get_batch_size(obs["obs"], 1) != num:
        raise ValueError("player batch size differs from simulator")
    if self.is_rnn:
        self.init_rnn()
    ids = torch.arange(num, device=self.device)
    tracker = source.HoldTracker(num, self.device)
    tracker.reset(ids, task._target_states[:, 2])
    start_frame = task.start_times.clone()
    motion = task.data_id.clone()
    if start_frame.ne(0).any() or torch.bincount(motion, minlength=3).tolist() != [32, 32, 32]:
        raise ValueError("frame-0 or 32/32/32 motion contract drift")
    episode_namespace = (int(args.seed_namespace) << 32) | (DIAGNOSTIC_SEED << 16)
    episode = torch.arange(num, device=self.device, dtype=torch.long) + episode_namespace
    step = torch.zeros(num, dtype=torch.long, device=self.device)
    previous_action = torch.zeros(num, 18, device=self.device)
    finished = torch.zeros(num, dtype=torch.bool, device=self.device)
    initial_object_z = task._target_states[:, 2].clone()
    max_lift = torch.zeros(num, device=self.device)
    lift_sum = torch.zeros(num, device=self.device)
    contact_sum = torch.zeros(num, device=self.device)
    completed = []
    parts: dict[str, list[torch.Tensor]] = {}
    shards = []
    rows = 0
    global_tick = 0
    started = time.monotonic()
    workflow_started_epoch = float(getattr(args, "workflow_started_epoch", time.time()))
    workflow_rows_before = int(getattr(args, "workflow_rows_before", 0))
    done_indices: list[int] = []

    def actor_forward(observation):
        # The installed player model owns running-mean/std normalization,
        # logstd -> exp conversion, one Gaussian draw, and rnn state updates.
        action, value, _ = _native_actor_forward(self, observation)
        return action, value

    def critic_value(observation):
        # Critic-only evaluation shares native normalization and consumes no
        # random numbers; tensor and dict observations are both supported.
        return _native_critic_value(self, observation)

    def gap():
        geometry = bridge.current(task._dof_pos, task._target_states)
        return source.sampled_surface_gap(geometry.hand_points, geometry.object_points, approach)

    def check_budget(extra_rows=0):
        elapsed = time.time() - workflow_started_epoch
        if elapsed > MAX_WALL_SECONDS or time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("runtime wall budget")
        if (rows + extra_rows > MAX_ROWS_PER_CHECKPOINT or
                workflow_rows_before + rows + extra_rows > MAX_ROWS_TOTAL):
            raise RuntimeError("runtime row budget")
        if _dir_bytes(args.run_root) > RUNTIME_MAX_OUTPUT_BYTES:
            raise RuntimeError("runtime output budget")

    def flush():
        nonlocal parts
        if not parts:
            return
        payload = {key: torch.cat(value) for key, value in parts.items()}
        payload.update(schema=source.SCHEMA, gamma=.99, control_dt=task.dt,
                       motion_names=list(task.motion_file), source_sha256=args.checkpoint_sha256)
        contract.validate_episode_rows(payload)
        source.validate_rows(payload)
        estimate = sum(v.numel() * v.element_size() for v in payload.values() if isinstance(v, torch.Tensor)) + 1_000_000
        check_budget()
        if _dir_bytes(args.run_root) + estimate > RUNTIME_MAX_OUTPUT_BYTES:
            raise RuntimeError("runtime output budget before export")
        path = args.run_dir / ("transitions_%03d.pt" % len(shards))
        torch.save(payload, path)
        shards.append({"path": str(path), "sha256": _sha256(path), "rows": len(payload["state"])})
        if _dir_bytes(args.run_root) > RUNTIME_MAX_OUTPUT_BYTES:
            raise RuntimeError("runtime output budget after export")
        parts = {}

    for tick in range(MAX_STEPS):
        obs = self.env_reset(torch.as_tensor(done_indices, dtype=torch.long, device=self.device))
        if done_indices:
            reset_ids = torch.tensor(done_indices, device=self.device)
            tracker.reset(reset_ids, task._target_states[reset_ids, 2])
            step[reset_ids] = 0; previous_action[reset_ids] = 0
            start_frame[reset_ids] = task.start_times[reset_ids]; motion[reset_ids] = task.data_id[reset_ids]
        before = source.snapshot(task, tracker); ctx = source.context(task, tracker)
        phase = task.progress_buf.clone()
        action, value_at_state = actor_forward(obs)
        gap_before = gap()
        obs_after, base_reward, done, info = self.env_step(self.env, action)
        done = done.to(self.device).bool().reshape(-1)
        terminate = info["terminate"].to(self.device).bool().reshape(-1)
        reward, components = source.shared_reward(task, tracker, base_reward.to(self.device), before[:, 38], gap_before, gap(), .99, approach)
        after = source.snapshot(task, tracker); next_ctx = source.context(task, tracker)
        next_value = critic_value(obs_after)
        next_value = torch.where(done, torch.zeros_like(next_value), next_value)
        active = ~finished
        active_rows = int(active.sum())
        check_budget(active_rows)
        values = dict(state=before, next_state=after, context=ctx, next_context=next_ctx,
                      action=action, previous_action=previous_action.clone(), reward=reward,
                      reward_components=components, done=done, terminate=terminate,
                      timeout=done & ~terminate, episode_id=episode.clone(), env_id=ids,
                      step=step.clone(), progress=phase, start_frame=start_frame.clone(),
                      motion_id=motion.clone(), checkpoint_seed=torch.full_like(ids, int(args.seed_namespace)),
                      value_at_state=value_at_state, next_value=next_value,
                      global_tick=torch.full_like(ids, global_tick), horizon_phase=torch.full_like(ids, global_tick % 32))
        for key, value in values.items():
            parts.setdefault(key, []).append(value[active].detach().cpu())
        rows += active_rows
        # A global simulator tick is shared by the active environments; it is
        # not a row counter.  Keeping this at one increment per env step makes
        # horizon_phase an actual 32-step phase.
        global_tick += 1
        max_lift = torch.maximum(max_lift, task._target_states[:, 2] - initial_object_z)
        lift_sum += task._target_states[:, 2] - initial_object_z
        contact_sum += source.contacts(task).bool().all(-1).float()
        step += 1; previous_action = action.detach().clone()
        terminal_ids = (done & active).nonzero(as_tuple=False).reshape(-1)
        for env_id in terminal_ids.tolist():
            completed.append({"env_id": env_id, "episode_id": int(episode[env_id]), "motion_id": int(motion[env_id]),
                              "start_frame": int(start_frame[env_id]), "steps": int(step[env_id]),
                              "terminate": bool(terminate[env_id]), "value_terminal_mask": 0,
                              "stable_success": bool(tracker.stable[env_id]),
                              "drop_after_success": bool(tracker.drop_after_success[env_id]),
                              "max_hold_seconds": float(tracker.max_run[env_id]),
                              "max_lift_meters": float(max_lift[env_id]),
                              "mean_lift_meters": float(lift_sum[env_id] / step[env_id]),
                              "contact_fraction": float(contact_sum[env_id] / step[env_id])})
        finished |= done
        if len(parts.get("state", [])) and sum(len(x) for x in parts["state"]) >= 32768:
            flush()
        if finished.all():
            break
        done_indices = done.nonzero(as_tuple=False).reshape(-1).tolist()
    else:
        raise RuntimeError("incomplete first episodes before max steps")
    flush()
    if len(completed) != num or rows > MAX_ROWS_PER_CHECKPOINT:
        raise RuntimeError("incomplete or over-budget first-episode panel")
    result = {"schema": source.SCHEMA, "mode": "collect", "run_status": "COMPLETED", "rows": rows,
              "complete_episodes": len(completed), "shards": shards, "per_episode": completed,
              "diagnostic_distribution": "frame0 fixed diagnostic sub-distribution; not full training reset distribution",
              "policy": {"gaussian": True, "enable_eps_greedy": False, "extra_noise": False, "teacher": False,
                         "weights_updated": False, "clip": [-1.0, 1.0]},
              "value_fields": {"value_at_state": "same actor forward critic", "next_value": "critic eval, zero on done",
                               "global_tick": "shared simulator env-step tick", "horizon_phase": "global_tick mod 32"},
              "elapsed_seconds": time.monotonic() - started}
    if _dir_bytes(args.run_root) + len(json.dumps(result)) + 4096 > RUNTIME_MAX_OUTPUT_BYTES:
        raise RuntimeError("runtime output budget before results export")
    (args.run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if _dir_bytes(args.run_root) > RUNTIME_MAX_OUTPUT_BYTES:
        raise RuntimeError("runtime output budget after results export")


def _real_collect(run_root: Path, seed: int, gpu_index: int) -> int:
    contract = _load_contract()
    run_dir, provenance = _prepare_root(run_root, seed, contract)
    run_dir.mkdir()
    manifest_path = run_dir / "runtime_manifest.json"
    runtime_manifest = {"schema": "ref2dex.current_policy_value.runtime.v2",
                        "run_status": "STARTED", "provenance": provenance,
                        "requested_gpu_index": gpu_index,
                        "workflow_started_epoch": provenance["workflow_started_epoch"],
                        "started_at": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(runtime_manifest, indent=2) + "\n")
    try:
        workflow_started = float(provenance["workflow_started_epoch"])
        if time.time() - workflow_started > MAX_WALL_SECONDS:
            raise TimeoutError("workflow wall budget expired before GPU initialization")
        rows_before = _existing_rows(run_root)
        if rows_before >= MAX_ROWS_TOTAL or _dir_bytes(run_root) > RUNTIME_MAX_OUTPUT_BYTES:
            raise RuntimeError("workflow rows/output budget already exhausted")
        ownership = _gpu_ownership(gpu_index)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # Import order is intentional: Isaac Gym before torch in this process.
        from isaacgym import gymapi  # noqa: F401
        import torch  # noqa: F401
        # The source-only contract fixture imports torch, so it must run only
        # after Isaac has been imported in the real runtime process.
        report = contract.preflight(None)
        if report["panels"][[p["seed"] for p in report["panels"]].index(seed)]["checkpoint_sha256"] != contract.CHECKPOINT_SHA256[seed]:
            raise ValueError("checkpoint input drift")
        source = _load_hf08_source()
        source.PhysicalPlayer.run = _runtime_player_run
        checkpoint_sha, checkpoint_path = CHECKPOINTS[seed]
        # Do not call source.main(): it has a historical SOURCE_SHA collect
        # guard for a different experiment.  Invoke the pinned DExplore
        # evaluate main directly with the verified current-policy checkpoint.
        source.ARGS = SimpleNamespace(
            mode="collect", run_dir=run_dir, run_root=run_root,
            checkpoint_sha256=checkpoint_sha, rows=MAX_ROWS_PER_CHECKPOINT,
            wall_seconds=MAX_WALL_SECONDS, assignment_seed=DIAGNOSTIC_SEED,
            seed_namespace=seed, max_steps=MAX_STEPS,
            workflow_started_epoch=workflow_started, workflow_rows_before=rows_before,
        )
        source.original.EvalPlayer = source.PhysicalPlayer
        argv = [str(SOURCE_RUNNER), "--task", "Dexplore_Inspire", "--cfg_env", ENV_CONFIG,
                "--cfg_train", TRAIN_CONFIG, "--checkpoint", checkpoint_path,
                "--motion_file", MOTION_ROOT, "--headless", "--num_envs", str(ENVS),
                "--seed", str(DIAGNOSTIC_SEED), "--sim_device", "cuda:0", "--rl_device", "cuda:0",
                "--graphics_device_id", "0", "--disable-early-termination",
                "--output", str(run_dir / "unused.json"), "--output_path", str(run_dir / "player"),
                "--play"]
        old_argv = sys.argv; sys.argv = argv
        try:
            source.original.main()
        finally:
            sys.argv = old_argv
        runtime_manifest.update(run_status="COMPLETED", gpu_ownership=ownership,
                                source_runner_sha256=contract.SOURCE_RUNNER_SHA256,
                                completed_at=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(runtime_manifest, indent=2) + "\n")
        return 0
    except BaseException as error:
        runtime_manifest.update(run_status="FAILED",
                                failure=f"{type(error).__name__}: {error}",
                                completed_at=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(runtime_manifest, indent=2) + "\n")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--checkpoint-seed", type=int, choices=(286, 287))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--gpu-index", type=int)
    args = parser.parse_args(argv)
    if args.preflight:
        contract = _load_contract()
        result = contract.preflight(args.run_root)
        print(json.dumps(result, indent=2, sort_keys=True) + "\n", end="")
        return 0
    if not args.collect or args.checkpoint_seed is None or args.run_root is None:
        parser.error("--collect requires --checkpoint-seed and --run-root")
    if args.mock:
        result = _mock_collect(args.run_root, args.checkpoint_seed)
        print(json.dumps(result, sort_keys=True) + "\n", end="")
        return 0
    if args.gpu_index is None:
        parser.error("real --collect requires --gpu-index")
    return _real_collect(args.run_root, args.checkpoint_seed, args.gpu_index)


if __name__ == "__main__":
    raise SystemExit(main())
