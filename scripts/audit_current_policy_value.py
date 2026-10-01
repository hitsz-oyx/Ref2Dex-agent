#!/usr/bin/env python3
"""Factual current-policy value diagnostics (CPU only).

The collector contract adds ``value_at_state`` (the frozen PPO critic output
from the same action forward), ``global_tick`` and the usual terminal fields
to complete ``physical_value.v1`` episodes.  This module computes two
diagnostic targets without training: complete Monte Carlo returns and
GAE(lambda) targets truncated at global horizon boundaries.  It never creates
missing critic values or crosses an episode/checkpoint boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch


SOURCE = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent")
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

GAMMA = 0.99
TAU = 0.95
HORIZON = 32
STATE_DIM = 55
PRIMARY_STEPS = 45


class DiagnosticError(RuntimeError):
    """Input cannot support a factual target without an invented value."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tensor(data: Mapping[str, Any], key: str) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise DiagnosticError(f"missing_required_field:{key}")
    return value


def _checkpoint_tensor(data: Mapping[str, Any], n: int) -> torch.Tensor:
    """Return explicit checkpoint IDs, or an explicit single-checkpoint ID."""
    for key in ("checkpoint_seed", "checkpoint_id", "checkpoint", "checkpoint_sha256", "model_id"):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            if len(value) != n:
                raise DiagnosticError(f"length_mismatch:{key}")
            return value
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise DiagnosticError(f"length_mismatch:{key}")
            # String checkpoint IDs are represented deterministically without
            # using a hash as scientific evidence; they only partition rows.
            mapping = {item: i for i, item in enumerate(dict.fromkeys(value))}
            return torch.tensor([mapping[item] for item in value], dtype=torch.long)
    return torch.zeros(n, dtype=torch.long)


def validate_diagnostic_schema(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate complete Episodes plus the collector's factual-value fields."""
    required = ("state", "next_state", "reward", "done", "terminate", "timeout",
                "episode_id", "step", "motion_id", "value_at_state", "global_tick")
    tensors = {key: _tensor(data, key) for key in required}
    n = len(tensors["reward"])
    if n == 0:
        raise DiagnosticError("empty_episodes")
    for key, value in tensors.items():
        if len(value) != n:
            raise DiagnosticError(f"length_mismatch:{key}")
    if tensors["state"].ndim != 2 or tensors["state"].shape[1] < STATE_DIM:
        raise DiagnosticError("state_shape")
    if tensors["next_state"].shape != tensors["state"].shape:
        raise DiagnosticError("next_state_shape")
    if tensors["value_at_state"].ndim > 2 or tensors["value_at_state"].numel() != n:
        raise DiagnosticError("value_at_state_shape")
    if tensors["episode_id"].dtype not in (torch.int64, torch.long):
        raise DiagnosticError("episode_id_must_be_int64")
    for key in ("step", "motion_id", "global_tick"):
        if tensors[key].dtype not in (torch.int64, torch.long):
            raise DiagnosticError(f"{key}_must_be_int64")
    for key in ("done", "terminate", "timeout"):
        if tensors[key].dtype not in (torch.bool, torch.uint8, torch.int8, torch.int64, torch.long):
            raise DiagnosticError(f"{key}_must_be_boolean_like")
    finite_keys = ("state", "next_state", "reward", "value_at_state")
    for key in finite_keys:
        if not torch.isfinite(tensors[key].float()).all():
            raise DiagnosticError(f"nonfinite:{key}")
    done = tensors["done"].bool()
    terminate = tensors["terminate"].bool()
    timeout = tensors["timeout"].bool()
    if (terminate & ~done).any() or (timeout & ~done).any() or ((terminate | timeout) != done).any():
        raise DiagnosticError("terminal_reason_inconsistent")
    checkpoints = _checkpoint_tensor(data, n)
    return {"n": n, "checkpoint_id": checkpoints, "provenance":
            "explicit_checkpoint_field" if any(k in data for k in ("checkpoint_seed", "checkpoint_id", "checkpoint", "checkpoint_sha256", "model_id"))
            else "implicit_single_checkpoint"}


def _groups(data: Mapping[str, Any], checkpoint_id: torch.Tensor) -> list[tuple[tuple[int, int], list[int]]]:
    episode = data["episode_id"].tolist()
    step = data["step"].tolist()
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (ckpt, eid) in enumerate(zip(checkpoint_id.tolist(), episode)):
        groups[(int(ckpt), int(eid))].append(index)
    result = []
    for key, indices in groups.items():
        indices.sort(key=lambda i: int(step[i]))
        expected = list(range(len(indices)))
        actual = [int(step[i]) for i in indices]
        if actual != expected:
            raise DiagnosticError(f"noncontiguous_episode_steps:{key}")
        done = data["done"].bool()
        if not bool(done[indices[-1]]) or bool(done[indices[:-1]].any()):
            raise DiagnosticError(f"episode_not_complete:{key}")
        result.append((key, indices))
    result.sort(key=lambda item: item[0])
    return result


def complete_mc_targets(data: Mapping[str, Any], groups: list[tuple[tuple[int, int], list[int]]]) -> torch.Tensor:
    reward = data["reward"].float().reshape(-1)
    done = data["done"].bool()
    targets = torch.full((len(reward),), float("nan"))
    for _, indices in groups:
        running = torch.tensor(0.0)
        for index in reversed(indices):
            running = reward[index] + GAMMA * (1.0 - float(done[index])) * running
            targets[index] = running
    return targets


def frozen_critic_lambda_targets(data: Mapping[str, Any], groups: list[tuple[tuple[int, int], list[int]]],
                                 checkpoint_id: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute PPO-style GAE targets with explicit rollout truncation.

    ``next_value`` comes from the next step of the same episode/checkpoint,
    even when that step starts a new rollout.  The recursive lambda carry is
    cut when ``global_tick // HORIZON`` changes.  Terminal rows, including
    timeout rows, use a zero bootstrap through ``done``.
    """
    n = len(data["reward"])
    reward = data["reward"].float().reshape(-1)
    value = data["value_at_state"].float().reshape(-1)
    done = data["done"].bool()
    tick = data["global_tick"].long()
    step = data["step"].long()
    episode = data["episode_id"].long()
    lookup = {(int(checkpoint_id[i]), int(episode[i]), int(step[i])): i for i in range(n)}
    next_value = torch.full((n,), float("nan"))
    next_index = torch.full((n,), -1, dtype=torch.long)
    missing = []
    for i in range(n):
        if bool(done[i]):
            next_value[i] = 0.0
            continue
        key = (int(checkpoint_id[i]), int(episode[i]), int(step[i]) + 1)
        j = lookup.get(key)
        if j is None:
            missing.append(i)
        else:
            next_index[i] = j
            next_value[i] = value[j]
            if int(tick[j]) != int(tick[i]) + 1:
                raise DiagnosticError(f"global_tick_step_mismatch:{i}->{j}")
    # ``value_at_next_state`` is the legacy spelling.  New collector output
    # uses ``next_value``.  Validate either spelling against the factual
    # same-episode/checkpoint successor above; never silently ignore one.
    # When both are present, checking each against the same successor also
    # enforces that the two collector fields agree.
    for field_name in ("value_at_next_state", "next_value"):
        if field_name not in data:
            continue
        provided_next = data[field_name]
        if not isinstance(provided_next, torch.Tensor):
            raise DiagnosticError(f"invalid_{field_name}")
        provided_next = provided_next.float().reshape(-1)
        if len(provided_next) != n or not torch.isfinite(provided_next).all():
            raise DiagnosticError(f"invalid_{field_name}")
        for i in range(n):
            if bool(done[i]):
                if abs(float(provided_next[i])) > 1e-6:
                    raise DiagnosticError(f"terminal_{field_name}_not_zero:{i}")
            elif int(next_index[i]) >= 0 and abs(float(provided_next[i] - next_value[i])) > 1e-5:
                raise DiagnosticError(f"{field_name}_crosscheck_mismatch:{i}")
    if missing:
        raise DiagnosticError(f"missing_nonterminal_next_critic_values:{len(missing)}")
    advantage = torch.full((n,), float("nan"))
    target = torch.full((n,), float("nan"))
    for _, indices in groups:
        gae = torch.tensor(0.0)
        for index in reversed(indices):
            delta = reward[index] + GAMMA * (1.0 - float(done[index])) * next_value[index] - value[index]
            j = int(next_index[index])
            same_rollout = (j >= 0 and int(tick[index]) % HORIZON != HORIZON - 1)
            if bool(done[index]) or not same_rollout:
                gae = delta
            else:
                gae = delta + GAMMA * TAU * gae
            advantage[index] = gae
            target[index] = value[index] + gae
    return target, {"missing_nonterminal_next_values": 0,
                    "rollout_boundary_rows": int(sum(1 for i in range(n)
                        if bool(done[i]) or int(next_index[i]) < 0 or int(tick[i]) % HORIZON == HORIZON - 1)),
                    "terminal_rows_masked": int(done.sum())}


def _phase_labels(data: Mapping[str, Any]) -> torch.Tensor:
    next_state = data["next_state"]
    episode = data["episode_id"]
    pair = next_state[:, 49:51].bool().all(-1)
    run = next_state[:, 51]
    ever = next_state[:, 52] > .5
    dropped = next_state[:, 54] > .5
    previous = torch.zeros_like(dropped)
    previous[1:] = dropped[:-1] & (episode[1:] == episode[:-1])
    edge = dropped & ~previous
    phase = torch.full((len(next_state),), 4, dtype=torch.long)
    phase[(~pair) & (~ever)] = 0
    phase[pair & (run < 1.0) & ~edge] = 1
    phase[(run >= 1.0) & ~edge] = 2  # diagnostic one-second held phase
    phase[edge] = 3
    return phase


def _primary_labels(data: Mapping[str, Any], groups: list[tuple[tuple[int, int], list[int]]]) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], bool]]:
    success, dropped = {}, {}
    for key, indices in groups:
        before, after = data["state"], data["next_state"]
        initial = float(before[indices[0], 38])
        run_steps, stable, drop_after = 0, False, False
        for i in indices:
            z = float(after[i, 38]); pair = bool(after[i, 49:51].bool().all())
            held = (z - initial >= .03) and pair
            previous_success = stable
            run_steps = run_steps + 1 if held else 0
            stable = stable or run_steps >= PRIMARY_STEPS
            falling = (z - initial < .02) or float(after[i, 53]) >= 1.0 - 1e-6
            drop_after = drop_after or (previous_success and falling)
        success[key], dropped[key] = bool(stable), bool(drop_after)
    return success, dropped


def _metric(values: torch.Tensor, target: torch.Tensor, episode_ids: list[int]) -> dict[str, Any]:
    if len(values) == 0:
        return {"rows": 0, "episodes": 0, "rmse": None, "mae": None, "bias": None,
                "judgement": "UNDETERMINED_EMPTY"}
    error = values - target
    out = {"rows": int(len(values)), "episodes": len(set(episode_ids)),
           "rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean()),
           "bias": float(error.mean())}
    if out["episodes"] < 5:
        out["judgement"] = "UNDETERMINED_RARE_SLICE"
    return out


@torch.no_grad()
def evaluate_saved_pv_value(data: Mapping[str, Any], model_checkpoint: Path,
                            online_checkpoints: Mapping[int, Path], batch_size: int = 128) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the saved ``physical_value.value`` network on factual states.

    The network output is already in reward/return units.  ``return_scale`` is
    used by the training loss denominator only and is deliberately not applied
    here.  Histories are rebuilt with episode *and* checkpoint guards so a
    duplicated episode ID cannot borrow another checkpoint's context.
    """
    from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork
    from src.task.CmResidual.physical_value_data import HISTORY
    payload = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    features = Features(**{k: v.to("cpu") for k, v in payload["stats"].items()}, device="cpu").to("cpu")
    context_dim = int(payload["context_dim"])
    n = len(data["reward"])
    # The saved physical V consumes a history of previous actions.  Filling
    # this with zeros would fabricate an unobserved factual context, so the
    # checkpoint-inference path requires the collector field explicitly.
    previous_action = data.get("previous_action")
    if not isinstance(previous_action, torch.Tensor):
        raise DiagnosticError("previous_action_required_for_saved_pv_v")
    if previous_action.ndim != 2 or tuple(previous_action.shape) != (n, 18):
        raise DiagnosticError("previous_action_shape_for_saved_pv_v")
    if not torch.isfinite(previous_action.float()).all():
        raise DiagnosticError("nonfinite:previous_action")
    checkpoints = _checkpoint_tensor(data, n)
    predictions = torch.full((n,), float("nan"))
    metadata = {"model_checkpoint_sha256": sha256(model_checkpoint), "return_scale_not_applied": True,
                "checkpoints": {}}
    episode = data["episode_id"].long(); step = data["step"].long()
    for checkpoint_seed, path in online_checkpoints.items():
        online = torch.load(path, map_location="cpu", weights_only=False)
        physical = online.get("physical_value", {})
        if physical.get("arm") != "cm_value":
            raise DiagnosticError(f"online_checkpoint_arm:{checkpoint_seed}")
        state = physical.get("value")
        if not isinstance(state, dict):
            raise DiagnosticError(f"online_value_state_missing:{checkpoint_seed}")
        network = OutcomeNetwork(context_dim, 0, 1, 9283).to("cpu")
        network.load_state_dict(state)
        network.eval()
        selected = (checkpoints == int(checkpoint_seed)).nonzero().flatten()
        if len(selected) == 0:
            continue
        predictions_for_seed = []
        for batch_indices in selected.split(batch_size):
            idx = batch_indices.long()
            history_index = idx[:, None] - torch.arange(HISTORY - 1, -1, -1)[None]
            safe = history_index.clamp_min(0)
            mask = (history_index >= 0)
            mask &= episode[safe] == episode[idx, None]
            mask &= checkpoints[safe] == checkpoints[idx, None]
            mask &= step[safe] == (step[idx, None] - (HISTORY - 1 - torch.arange(HISTORY)[None]))
            hs = data["state"][safe] * mask[:, :, None]
            ha = previous_action[safe] * mask[:, :, None]
            hm = mask[:, :, None].float()
            history = features.history(hs, ha, hm)
            context = features.context(data["context"][idx])
            predictions_for_seed.append(network(history, context).squeeze(-1).cpu())
        predictions[selected] = torch.cat(predictions_for_seed)
        tensor_values = [v for v in state.values() if isinstance(v, torch.Tensor)]
        metadata["checkpoints"][str(checkpoint_seed)] = {
            "path": str(path), "sha256": sha256(path), "epoch": online.get("epoch"), "frame": online.get("frame"),
            "finite": bool(all(torch.isfinite(v).all() for v in tensor_values)), "rows": int(len(selected)),
            "value_parameter_count": int(sum(v.numel() for v in tensor_values)),
            "optimizer_step": sorted({float(v["step"]) for v in physical.get("value_optimizer", {}).get("state", {}).values()
                                       if isinstance(v, dict) and isinstance(v.get("step"), torch.Tensor)}),
        }
    if not torch.isfinite(predictions).all():
        missing = int((~torch.isfinite(predictions)).sum())
        raise DiagnosticError(f"missing_saved_pv_predictions:{missing}")
    return predictions, metadata


def run_diagnostic(data: Mapping[str, Any], pv_value_at_state: torch.Tensor | None = None) -> dict[str, Any]:
    """Run factual targets and stratified errors on an in-memory Episodes map."""
    contract = validate_diagnostic_schema(data)
    groups = _groups(data, contract["checkpoint_id"])
    mc = complete_mc_targets(data, groups)
    lam, lambda_meta = frozen_critic_lambda_targets(data, groups, contract["checkpoint_id"])
    phase = _phase_labels(data)
    primary, primary_drop = _primary_labels(data, groups)
    native = data.get("_native_per_episode")
    native_checked = 0
    if native:
        for (checkpoint, episode_id), value in primary.items():
            if episode_id not in native:
                raise DiagnosticError(f"native_per_episode_missing:{episode_id}")
            expected = native[episode_id]
            if bool(value) != bool(expected["stable_success"]) or bool(primary_drop[(checkpoint, episode_id)]) != bool(expected["drop_after_success"]):
                raise DiagnosticError(f"native_primary_mismatch:{checkpoint}:{episode_id}")
            native_checked += 1
    value = data["value_at_state"].float().reshape(-1)
    episode = data["episode_id"].long()
    motion = data["motion_id"].long()
    group_lookup = {}
    for key, indices in groups:
        for i in indices:
            group_lookup[i] = key
    def strata(prediction, name_to_mask):
        output = {}
        for name, mask in name_to_mask.items():
            ids = episode[mask].tolist()
            output[name] = {"mc": _metric(prediction[mask], mc[mask], ids),
                            "lambda": _metric(prediction[mask], lam[mask], ids)}
        return output
    all_mask = torch.ones(len(value), dtype=torch.bool)
    success_mask = torch.tensor([primary[group_lookup[i]] for i in range(len(value))])
    drop_mask = torch.tensor([primary_drop[group_lookup[i]] for i in range(len(value))])
    masks = {
        "overall": {"all": all_mask},
        "checkpoint": {str(int(c)): contract["checkpoint_id"] == c
                       for c in torch.unique(contract["checkpoint_id"]).tolist()},
        "primary": {"primary_success": success_mask, "primary_failure": ~success_mask,
                     "drop_after_primary": drop_mask, "no_drop_after_primary": ~drop_mask},
        "motion": {str(m): motion == m for m in sorted(set(motion.tolist()))},
        "phase": {name: phase == index for index, name in enumerate(("pre_contact", "contact", "held_1s", "drop", "other"))},
    }
    predictions = {"ppo_critic": value}
    if pv_value_at_state is not None:
        pv_value_at_state = pv_value_at_state.float().reshape(-1)
        if len(pv_value_at_state) != len(value) or not torch.isfinite(pv_value_at_state).all():
            raise DiagnosticError("invalid_saved_pv_value_predictions")
        predictions["saved_pv_v"] = pv_value_at_state
    metrics = {model_name: {stratum: strata(prediction, stratum_masks)
                            for stratum, stratum_masks in masks.items()}
               for model_name, prediction in predictions.items()}
    episode_rows = []
    for key, indices in groups:
        idx = torch.tensor(indices, dtype=torch.long)
        episode_rows.append({"checkpoint": key[0], "episode_id": key[1], "motion_id": int(motion[indices[0]]),
                             "rows": len(indices), "primary_success": primary[key],
                             "drop_after_primary": primary_drop[key], "mc_rmse": float((value[idx] - mc[idx]).square().mean().sqrt()),
                             "lambda_rmse": float((value[idx] - lam[idx]).square().mean().sqrt())})
        if pv_value_at_state is not None:
            episode_rows[-1]["saved_pv_v_mc_rmse"] = float((pv_value_at_state[idx] - mc[idx]).square().mean().sqrt())
            episode_rows[-1]["saved_pv_v_lambda_rmse"] = float((pv_value_at_state[idx] - lam[idx]).square().mean().sqrt())
    return {
        "status": "COMPLETED", "contract": {"gamma": GAMMA, "tau": TAU, "horizon": HORIZON,
            "terminal_bootstrap": "done mask zero, including timeout", "target_origin": "frozen critic factual diagnostic; not historical labels",
            "checkpoint_provenance": contract["provenance"]},
        "rows": len(value), "episodes": len(groups), "lambda_meta": lambda_meta,
        "native_per_episode_checked": native_checked,
        "lambda_vs_mc": {
            name: _metric(lam[mask], mc[mask], episode[mask].tolist())
            for name, mask in {"all": all_mask, **masks["checkpoint"]}.items()
        },
        "metrics": metrics, "episode_metrics": episode_rows,
        "primary_episode_counts": {"success": int(sum(primary.values())), "drop_after_primary": int(sum(primary_drop.values()))},
    }


def load_collections(collections: list[Path]) -> dict[str, Any]:
    from src.task.CmResidual.physical_value_data import Episodes
    dataset = Episodes(collections, gamma=GAMMA)
    if int(dataset.excluded_rows) != 0:
        raise DiagnosticError(f"excluded_rows_nonzero:{dataset.excluded_rows}")
    data = dataset.data
    # Episodes retains extra tensor fields.  Do not fill missing collector fields.
    native = {}
    for collection in collections:
        result = json.loads((Path(collection) / "results.json").read_text())
        for row in result.get("per_episode", []):
            native[int(row["episode_id"])] = {
                "stable_success": bool(row["stable_success"]),
                "drop_after_success": bool(row["drop_after_success"]),
            }
    data = dict(data)
    data["_native_per_episode"] = native
    data["_episodes_object"] = dataset
    return data


def synthetic_smoke() -> dict[str, Any]:
    """Small hand-computable smoke covering terminal, timeout, boundary, split."""
    n = 8
    state = torch.zeros(n, STATE_DIM); next_state = torch.zeros_like(state)
    state[:, 39] = 1; next_state[:, 39] = 1
    episode = torch.tensor([11, 11, 11, 11, 22, 22, 22, 22], dtype=torch.long)
    step = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    ticks = torch.tensor([30, 31, 32, 33, 80, 81, 82, 83], dtype=torch.long)
    rewards = torch.tensor([1., 2., 3., 4., 1., 2., 3., 4.])
    values = torch.tensor([10., 20., 30., 40., 10., 20., 30., 40.])
    done = torch.tensor([False, False, False, True, False, False, False, True])
    timeout = torch.tensor([False, False, False, True, False, False, False, False])
    terminate = done & ~timeout
    data = {"state": state, "next_state": next_state, "reward": rewards, "done": done,
            "terminate": terminate, "timeout": timeout, "episode_id": episode, "step": step,
            "motion_id": torch.zeros(n, dtype=torch.long), "value_at_state": values,
            "global_tick": ticks, "checkpoint_seed": torch.full((n,), 286, dtype=torch.long),
            "value_at_next_state": torch.tensor([20., 30., 40., 0., 20., 30., 40., 0.]),
            "next_value": torch.tensor([20., 30., 40., 0., 20., 30., 40., 0.])}
    # Inject an independent saved-V prediction to exercise the separate V
    # metric path without claiming this fixture is a policy calibration run.
    result = run_diagnostic(data, pv_value_at_state=data["value_at_state"] + 1.0)
    # MC terminal checks and boundary cut: tick 31 -> 32 crosses the 32-step
    # rollout boundary, so lambda at row 1 does not carry row 2's advantage.
    groups = _groups(data, torch.zeros(n, dtype=torch.long))
    mc = complete_mc_targets(data, groups)
    assert torch.allclose(mc[:4], torch.tensor([9.801496, 8.8904, 6.96, 4.]), atol=1e-5)
    assert result["lambda_meta"]["terminal_rows_masked"] == 2
    assert result["lambda_meta"]["rollout_boundary_rows"] >= 3
    assert "saved_pv_v" in result["metrics"]
    assert result["metrics"]["saved_pv_v"]["overall"]["all"]["mc"]["bias"] != result["metrics"]["ppo_critic"]["overall"]["all"]["mc"]["bias"]
    data["context"] = torch.zeros(n, 435)
    data["previous_action"] = torch.zeros(n, 18)
    model_root = SOURCE / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7"
    online_path = next((model_root / "train_cm_value_s286").glob("**/GRAB_00000420.pth"))
    _, saved_v_smoke = evaluate_saved_pv_value(data, model_root / "models/tier_1000000.pt", {286: online_path})
    return {"status": "COMPLETED", "scope": "synthetic engineering smoke only", "diagnostic": result,
            "saved_v_checkpoint_smoke": saved_v_smoke,
            "checks": ["hand_computable_mc", "terminal_timeout_zero_bootstrap", "rollout_boundary_cut", "episode_split", "saved_pv_v_inference"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--model-checkpoint", type=Path,
                        default=SOURCE / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/models/tier_1000000.pt")
    parser.add_argument("--online-checkpoint", action="append", default=[], metavar="SEED=PATH",
                        help="saved cm_value checkpoint mapping, e.g. 286=/path/GRAB_00000420.pth")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    try:
        if args.synthetic or not args.collections:
            smoke = synthetic_smoke()
            report = {"run_status": "COMPLETED", "scope": "synthetic engineering smoke only; no real data claim",
                      "synthetic": smoke}
        else:
            data = load_collections(args.collections)
            mappings = {}
            for item in args.online_checkpoint:
                if "=" not in item:
                    raise DiagnosticError("online_checkpoint_requires_seed_equals_path")
                seed, path = item.split("=", 1)
                mappings[int(seed)] = Path(path)
            if not mappings:
                raise DiagnosticError("online_checkpoint_mapping_required_for_factual_v")
            saved_v, saved_meta = evaluate_saved_pv_value(data, args.model_checkpoint, mappings)
            diagnostic = run_diagnostic(data, pv_value_at_state=saved_v)
            diagnostic["saved_pv_v"] = saved_meta
            report = {"run_status": "COMPLETED", "scope": "factual current-policy diagnostic; no utility/counterfactual/convergence claim",
                      "diagnostic": diagnostic}
    except DiagnosticError as error:
        report = {"run_status": "NEEDS_HELP", "scope": "input contract cannot determine target without invented values",
                  "reason": str(error)}
    elapsed = time.monotonic() - started
    report.update({"cpu_only": {"device": "cpu", "torch_threads": 2, "gpu": 0, "training": False, "collection": False},
                   "elapsed_seconds": elapsed, "command": sys.argv,
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    report["output_bytes"] = args.output.stat().st_size
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"status": report["run_status"], "elapsed_seconds": elapsed,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
