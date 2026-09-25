"""Probe a short-history, probabilistic Cm on randomized follow-up data.

The data in this probe come from sequentially randomized simulator interventions.
They identify conditional population effects on reached states; they do not create
same-state counterfactual pairs.  The model keeps separate heads for physical
outcomes so that no hand-written scalar value target is needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference


ROOT = Path(__file__).resolve().parents[4]
NUM_ENVS = 64
FOLLOWUP_HORIZON = 5
OBS_DIM = 18 + 18 + 13
ACTION_DIM = 18
STEP_DIM = OBS_DIM + ACTION_DIM
CONTINUOUS_TARGETS = (
    "one_step_dz_mm",
    "followup_dz_mm",
    "contact_fraction",
    "supported_dz_mm",
)
BINARY_TARGETS = ("final_contact",)
TRAIN_SEEDS = (151, 152, 153, 154)
TEST_SEEDS = (155, 156)
DATA = {
    151: (ROOT / "outputs/CmResidual/agent_randomized_followup_s151_d01_h5_n64/transitions.pt",
          "151825821ba8c43762d71c40c4dd20886d83d4f66c1c9168564a002658ac2b05"),
    152: (ROOT / "outputs/CmResidual/agent_randomized_followup_s152_d01_h5_n64/transitions.pt",
          "2d34b3adb6de8c61da7ce817b2fcf4fa898d513bf9cadf6385214eb103826a15"),
    153: (ROOT / "outputs/CmResidual/agent_randomized_followup_s153_d01_h5_n64/transitions.pt",
          "9e2d96ac37b6bbafc393215274360038692511487a755899ce04e5dd81ee948b"),
    154: (ROOT / "outputs/CmResidual/agent_randomized_followup_s154_d01_h5_n64/transitions.pt",
          "9120d8e52e6b7b24bd4d66876c82938a1db3761fb399e5232a05947b42b9e210"),
    155: (ROOT / "outputs/CmResidual/agent_randomized_followup_s155_d01_h5_n64/transitions.pt",
          "5b1e64f0ced1aad299db0adf5445521829c7ade19999c5ec3892c2007efc0f7c"),
    156: (ROOT / "outputs/CmResidual/agent_randomized_followup_s156_d01_h5_n64/transitions.pt",
          "0dbf7ae466eace755ea41ae44a6aff610faa2c09a7e3091804b0fa4a197692f5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1)


def _validate_blocks(records: dict[str, torch.Tensor]) -> tuple[int, int]:
    count = len(records["assignment"])
    if count == 0 or count % NUM_ENVS:
        raise ValueError("follow-up records must contain complete environment blocks")
    steps = _flat(records["global_step"]).numpy()
    blocks = steps.reshape(-1, NUM_ENVS)
    if not np.all(blocks == blocks[:, :1]):
        raise ValueError("global_step is not constant within environment blocks")
    if not np.all(np.diff(blocks[:, 0]) > 0):
        raise ValueError("global_step blocks are not strictly increasing")
    return count // NUM_ENVS, count


def _row_features(records: dict[str, torch.Tensor]) -> torch.Tensor:
    q = records["q"].float()
    object_state = records["object_state"].float()
    q_relative = q.clone()
    q_relative[:, :3] -= object_state[:, :3]
    observation = torch.cat((q_relative, records["dof_vel"].float(), object_state), dim=-1)
    return torch.cat((observation, records["executed_action"].float()), dim=-1)


def load_rows(seed: int, history_len: int) -> dict[str, torch.Tensor]:
    path, expected_sha = DATA[seed]
    if sha256(path) != expected_sha:
        raise ValueError(f"input SHA drift for seed {seed}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.randomized_action_followup.v1":
        raise ValueError(f"unexpected schema for seed {seed}")
    if payload.get("run_status") != "COMPLETED" or payload.get("followup_horizon") != FOLLOWUP_HORIZON:
        raise ValueError(f"incomplete or incompatible follow-up run for seed {seed}")
    records = payload["records"]
    time_count, count = _validate_blocks(records)
    assignment = _flat(records["assignment"]).to(torch.int64)
    pre_contact = _flat(records["pre_contact"]).bool()
    selected = (assignment != 0) & pre_contact
    if not selected.any():
        raise ValueError(f"no randomized contact rows for seed {seed}")

    all_features = _row_features(records)
    # Rows are [time, environment].  Padding at the beginning repeats the
    # first observed state and is explicitly marked by history_mask.
    flat_index = torch.arange(count, dtype=torch.long)
    time_index = flat_index // NUM_ENVS
    env_index = flat_index % NUM_ENVS
    selected_index = torch.nonzero(selected, as_tuple=False).reshape(-1)
    selected_time = time_index[selected_index]
    selected_env = env_index[selected_index]
    offsets = torch.arange(history_len, dtype=torch.long)
    history_time = selected_time[:, None] - (history_len - 1 - offsets[None, :])
    history_mask = (history_time >= 0).float()
    history_time = history_time.clamp_min(0)
    history_index = history_time * NUM_ENVS + selected_env[:, None]
    history = all_features[history_index]

    object_state = records["object_state"].float()
    next_object_state = records["next_object_state"].float()
    followup_object_state = records["followup_object_state"].float()
    followup_fraction = _flat(records["followup_contact_count"]).float() / FOLLOWUP_HORIZON
    followup_dz = (followup_object_state[:, 2] - object_state[:, 2]) * 1000.0
    one_step_dz = (next_object_state[:, 2] - object_state[:, 2]) * 1000.0
    selected_float = selected_index
    outcomes = torch.stack((
        one_step_dz[selected_float],
        followup_dz[selected_float],
        followup_fraction[selected_float],
        (followup_dz * followup_fraction)[selected_float],
    ), dim=-1)
    final_contact = _flat(records["followup_contact"]).float()[selected_float]
    base_action = records["base_action"].float()[selected_float]
    actual_action = records["executed_action"].float()[selected_float]
    delta = float(payload["delta_z_action"])
    steps = _flat(records["global_step"])[selected_float].to(torch.int64)
    if not torch.allclose(actual_action[:, 2] - base_action[:, 2],
                          assignment[selected_float].float() * delta, atol=1e-5):
        raise ValueError(f"current action does not match assignment for seed {seed}")
    if not torch.isfinite(history).all() or not torch.isfinite(outcomes).all():
        raise FloatingPointError(f"non-finite input or target for seed {seed}")
    return {
        "history": history,
        "history_mask": history_mask,
        "base_action": base_action,
        "actual_action": actual_action,
        "outcomes": outcomes,
        "final_contact": final_contact,
        "assignment": assignment[selected_float],
        "step": steps,
        "env_id": env_index[selected_float],
        "delta": torch.tensor(delta),
        "seed": torch.tensor(seed),
        "time_count": torch.tensor(time_count),
    }


def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in parts[0]:
        if key in {"delta", "seed", "time_count"}:
            continue
        result[key] = torch.cat([part[key] for part in parts], dim=0)
    return result


def _replace_current_action(rows: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    history = rows["history"].clone()
    history[:, -1, OBS_DIM:] = action
    return history


def shuffled_history(rows: dict[str, torch.Tensor], seed: int) -> torch.Tensor:
    """Break current action/outcome correspondence within each time stratum."""
    generator = torch.Generator().manual_seed(seed)
    result = rows["history"].clone()
    deltas = rows["actual_action"] - rows["base_action"]
    shuffled_action = rows["base_action"].clone()
    for step in torch.unique(rows["step"]):
        indices = torch.nonzero(rows["step"] == step, as_tuple=False).reshape(-1)
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        shuffled_action[indices] += deltas[permutation]
    result[:, -1, OBS_DIM:] = shuffled_action
    return result


class HistoryValueCm(nn.Module):
    """A compact probabilistic multi-target Cm."""

    def __init__(self, history_len: int, width: int = 96, continuous_count: int = 4):
        super().__init__()
        input_dim = history_len * (STEP_DIM + 1)
        self.history_len = history_len
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.continuous_mean = nn.Linear(width, continuous_count)
        self.continuous_logvar = nn.Linear(width, continuous_count)
        self.final_contact_logit = nn.Linear(width, 1)

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if history.ndim != 3 or history.shape[1] != self.history_len or history.shape[2] != STEP_DIM:
            raise ValueError("history has an unexpected shape")
        if history_mask.shape != history.shape[:2]:
            raise ValueError("history mask has an unexpected shape")
        inputs = torch.cat((history, history_mask.unsqueeze(-1)), dim=-1).flatten(1)
        hidden = self.net(inputs)
        return {
            "continuous_mean": self.continuous_mean(hidden),
            "continuous_logvar": self.continuous_logvar(hidden).clamp(-5.0, 4.0),
            "final_contact_logit": self.final_contact_logit(hidden).squeeze(-1),
        }


def _variant_history(rows: dict[str, torch.Tensor], variant: str, *, shuffle_seed: int) -> torch.Tensor:
    if variant == "history_action":
        return rows["history"]
    if variant == "current_action":
        return rows["history"][:, -1:]
    if variant == "history_blind":
        return _replace_current_action(rows, rows["base_action"])
    if variant == "history_shuffled":
        return shuffled_history(rows, shuffle_seed)
    raise ValueError(f"unknown model variant: {variant}")


def _variant_mask(rows: dict[str, torch.Tensor], variant: str) -> torch.Tensor:
    if variant == "current_action":
        return rows["history_mask"][:, -1:]
    return rows["history_mask"]


def fit_stats(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = history.reshape(-1, history.shape[-1])
    return flat.mean(0), flat.std(0).clamp_min(1e-3)


def normalized(history: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (history - mean) / std


def fit_targets(rows: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mean = rows["outcomes"].mean(0)
    std = rows["outcomes"].std(0).clamp_min(1e-3)
    return mean, std


def fit_model(model: HistoryValueCm, history: torch.Tensor, mask: torch.Tensor,
              rows: dict[str, torch.Tensor], feature_mean: torch.Tensor,
              feature_std: torch.Tensor, target_mean: torch.Tensor,
              target_std: torch.Tensor, *, steps: int, batch_size: int,
              seed: int) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    count = len(history)
    normalized_history = normalized(history, feature_mean, feature_std)
    target = (rows["outcomes"] - target_mean) / target_std
    final_loss = float("nan")
    for _ in range(steps):
        ids = torch.randint(count, (min(batch_size, count),), generator=generator)
        output = model(normalized_history[ids], mask[ids])
        logvar = output["continuous_logvar"]
        error = output["continuous_mean"] - target[ids]
        continuous_nll = .5 * (error.square() * torch.exp(-logvar) + logvar).mean()
        binary_loss = F.binary_cross_entropy_with_logits(
            output["final_contact_logit"], rows["final_contact"][ids])
        loss = continuous_nll + binary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite history Cm loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        final_loss = float(loss.detach())
    return {"final_loss": final_loss}


@torch.no_grad()
def predict(model: HistoryValueCm, history: torch.Tensor, mask: torch.Tensor,
            feature_mean: torch.Tensor, feature_std: torch.Tensor,
            target_mean: torch.Tensor, target_std: torch.Tensor) -> dict[str, torch.Tensor]:
    output = model(normalized(history, feature_mean, feature_std), mask)
    return {
        "outcomes": output["continuous_mean"] * target_std + target_mean,
        "continuous_logvar": output["continuous_logvar"],
        "final_contact": torch.sigmoid(output["final_contact_logit"]),
    }


def prediction_metrics(prediction: dict[str, torch.Tensor], rows: dict[str, torch.Tensor],
                       target_std: torch.Tensor) -> dict[str, float]:
    errors = prediction["outcomes"] - rows["outcomes"]
    result = {f"rmse_{name}": float(errors[:, i].square().mean().sqrt())
              for i, name in enumerate(CONTINUOUS_TARGETS)}
    result["brier_final_contact"] = float(
        (prediction["final_contact"] - rows["final_contact"]).square().mean())
    normalized_error = errors / target_std
    result["mean_squared_error_normalized"] = float(normalized_error.square().mean())
    result["mean_nll_normalized"] = float(
        .5 * (normalized_error.square() + prediction["continuous_logvar"]
              + normalized_error.square() *
              (torch.exp(-prediction["continuous_logvar"]) - 1.0)).mean())
    return result


def _weighted_effect(outcome: np.ndarray, assignment: np.ndarray,
                     step: np.ndarray) -> float:
    return float(weighted_step_difference(outcome, assignment, step)[0])


def rank_report(score: np.ndarray, outcome: np.ndarray, assignment: np.ndarray,
                step: np.ndarray, env_id: np.ndarray, *, bootstraps: int,
                seed: int) -> dict[str, object]:
    q25, q75 = np.quantile(score, [.25, .75])
    low = score <= q25
    high = score >= q75
    result: dict[str, object] = {
        "score_q25": float(q25), "score_q75": float(q75),
        "low_count": int(low.sum()), "high_count": int(high.sum()),
    }
    if q75 - q25 < 1e-7 or min(low.sum(), high.sum()) < 20:
        result["status"] = "NO_SCORE_SPREAD"
        return result
    full = np.arange(len(score))

    def group_ate(indices: np.ndarray, group: np.ndarray) -> float:
        selected = group[indices]
        if selected.sum() == 0:
            raise ValueError("empty score group")
        return _weighted_effect(outcome[indices][selected], assignment[indices][selected],
                                step[indices][selected])

    low_ate = group_ate(full, low)
    high_ate = group_ate(full, high)
    clusters = [np.flatnonzero(env_id == identifier) for identifier in np.unique(env_id)]
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(bootstraps):
        selected_clusters = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in selected_clusters])
        try:
            draws.append(group_ate(indices, high) - group_ate(indices, low))
        except ValueError:
            continue
    if len(draws) < max(100, int(.8 * bootstraps)):
        raise ValueError("too few valid cluster bootstrap draws")
    result.update({
        "status": "COMPLETED",
        "actual_ate_low": low_ate,
        "actual_ate_high": high_ate,
        "actual_high_minus_low": high_ate - low_ate,
        "cluster_95ci": np.quantile(draws, [.025, .975]).tolist(),
        "effective_bootstraps": len(draws),
    })
    return result


def candidate_history(rows: dict[str, torch.Tensor], variant: str, sign: int) -> tuple[torch.Tensor, torch.Tensor]:
    action = rows["base_action"].clone()
    action[:, 2] = (action[:, 2] + sign * float(rows["delta"])).clamp(-1.0, 1.0)
    if variant == "current_action":
        history = rows["history"][:, -1:].clone()
        history[:, -1, OBS_DIM:] = action
    elif variant == "history_blind":
        # The blind control is not allowed to extrapolate from a constant action
        # feature; both candidate scores are therefore exactly tied.
        action = rows["base_action"]
        history = _replace_current_action(rows, action)
    elif variant == "history_action":
        history = _replace_current_action(rows, action)
    elif variant == "history_shuffled":
        history = shuffled_history(rows, 240951)
        history[:, -1, OBS_DIM:] = action
    else:
        raise ValueError(f"unknown model variant: {variant}")
    return history, _variant_mask(rows, variant)


def pooled_rows(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    result = concatenate(parts)
    # Keep strata and clusters disjoint between evaluation seeds.
    for index, part in enumerate(parts):
        start = sum(len(previous["step"]) for previous in parts[:index])
        end = start + len(part["step"])
        result["step"][start:end] += index * 10000
        result["env_id"][start:end] += index * NUM_ENVS
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--fit-seed", type=int, default=240950)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    if not 1 <= args.history_len <= 10 or not 100 <= args.steps <= 3000:
        parser.error("history length or training steps outside bounded Probe range")
    if not 32 <= args.batch_size <= 512 or not 100 <= args.bootstraps <= 5000:
        parser.error("batch size or bootstrap count outside bounded Probe range")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "run_id": args.output.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
        "input_sha256": {seed: DATA[seed][1] for seed in (*TRAIN_SEEDS, *TEST_SEEDS)},
        "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
        "history_len": args.history_len, "steps": args.steps,
        "batch_size": args.batch_size, "bootstraps": args.bootstraps,
        "fit_seed": args.fit_seed,
        "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 30,
        "output_budget_mb": 20,
        "stop_rule": "input drift, leakage, non-finite output or sparse randomization",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(args.fit_seed)
        all_rows = {seed: load_rows(seed, args.history_len)
                    for seed in (*TRAIN_SEEDS, *TEST_SEEDS)}
        train = concatenate([all_rows[seed] for seed in TRAIN_SEEDS])
        test_parts = [all_rows[seed] for seed in TEST_SEEDS]
        test = pooled_rows(test_parts)
        target_mean, target_std = fit_targets(train)
        variants = ("history_action", "current_action", "history_blind", "history_shuffled")
        models: dict[str, HistoryValueCm] = {}
        feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        training: dict[str, dict[str, float]] = {}
        for model_index, variant in enumerate(variants):
            history = _variant_history(train, variant, shuffle_seed=240951)
            mask = _variant_mask(train, variant)
            feature_mean, feature_std = fit_stats(history)
            torch.manual_seed(args.fit_seed + 10 + model_index)
            model = HistoryValueCm(history.shape[1])
            model_stats = fit_model(
                model, history, mask, train, feature_mean, feature_std,
                target_mean, target_std, steps=args.steps, batch_size=args.batch_size,
                seed=args.fit_seed + 20 + model_index)
            models[variant] = model.eval()
            feature_stats[variant] = (feature_mean, feature_std)
            training[variant] = model_stats

        prediction_reports: dict[str, dict[str, object]] = {}
        rank_inputs: dict[str, dict[str, list[np.ndarray]]] = {
            variant: {target: [] for target in (*CONTINUOUS_TARGETS, "final_contact")}
            for variant in variants
        }
        observed_inputs = {target: [] for target in (*CONTINUOUS_TARGETS, "final_contact")}
        for seed_index, seed in enumerate(TEST_SEEDS):
            rows = all_rows[seed]
            seed_report: dict[str, object] = {"samples": len(rows["step"])}
            for variant in variants:
                history = _variant_history(rows, variant, shuffle_seed=240970 + seed)
                mask = _variant_mask(rows, variant)
                model = models[variant]
                feature_mean, feature_std = feature_stats[variant]
                prediction = predict(model, history, mask, feature_mean, feature_std,
                                     target_mean, target_std)
                seed_report[variant] = prediction_metrics(prediction, rows, target_std)
                plus_history, plus_mask = candidate_history(rows, variant, 1)
                minus_history, minus_mask = candidate_history(rows, variant, -1)
                plus = predict(model, plus_history, plus_mask, feature_mean, feature_std,
                               target_mean, target_std)
                minus = predict(model, minus_history, minus_mask, feature_mean, feature_std,
                                target_mean, target_std)
                for index, target in enumerate(CONTINUOUS_TARGETS):
                    rank_inputs[variant][target].append(
                        (plus["outcomes"][:, index] - minus["outcomes"][:, index]).numpy())
                rank_inputs[variant]["final_contact"].append(
                    (plus["final_contact"] - minus["final_contact"]).numpy())
            for index, target in enumerate(CONTINUOUS_TARGETS):
                observed_inputs[target].append(rows["outcomes"][:, index].numpy())
            observed_inputs["final_contact"].append(rows["final_contact"].numpy())
            prediction_reports[str(seed)] = seed_report

        pooled_test = test
        ranking: dict[str, object] = {}
        for variant in variants:
            ranking[variant] = {}
            for target in (*CONTINUOUS_TARGETS, "final_contact"):
                if target in CONTINUOUS_TARGETS:
                    target_index = CONTINUOUS_TARGETS.index(target)
                    outcome = pooled_test["outcomes"][:, target_index].numpy()
                else:
                    outcome = pooled_test["final_contact"].numpy()
                score = np.concatenate(rank_inputs[variant][target])
                ranking[variant][target] = rank_report(
                    score, outcome, pooled_test["assignment"].numpy(),
                    pooled_test["step"].numpy(), pooled_test["env_id"].numpy(),
                    bootstraps=args.bootstraps,
                    seed=240980 + list(variants).index(variant))
        ranking_per_seed: dict[str, object] = {}
        for seed_index, seed in enumerate(TEST_SEEDS):
            rows = all_rows[seed]
            ranking_per_seed[str(seed)] = {}
            for variant in variants:
                ranking_per_seed[str(seed)][variant] = {}
                for target in (*CONTINUOUS_TARGETS, "final_contact"):
                    if target in CONTINUOUS_TARGETS:
                        target_index = CONTINUOUS_TARGETS.index(target)
                        outcome = rows["outcomes"][:, target_index].numpy()
                    else:
                        outcome = rows["final_contact"].numpy()
                    ranking_per_seed[str(seed)][variant][target] = rank_report(
                        rank_inputs[variant][target][seed_index], outcome,
                        rows["assignment"].numpy(), rows["step"].numpy(),
                        rows["env_id"].numpy(), bootstraps=args.bootstraps,
                        seed=241000 + seed_index * 10 + list(variants).index(variant))
        history_followup = ranking["history_action"]["followup_dz_mm"]
        history_supported = ranking["history_action"]["supported_dz_mm"]
        history_contact = ranking["history_action"]["contact_fraction"]
        shuffled_followup = ranking["history_shuffled"]["followup_dz_mm"]
        shuffled_supported = ranking["history_shuffled"]["supported_dz_mm"]
        physical_candidates = (history_followup, history_supported)
        physical_pass = any(
            candidate.get("status") == "COMPLETED"
            and candidate["actual_high_minus_low"] >= 5.0
            and candidate["cluster_95ci"][0] > 0.0
            for candidate in physical_candidates)
        control_separation = all(
            candidate.get("status") == "COMPLETED"
            and shuffled.get("status") == "COMPLETED"
            and candidate["actual_high_minus_low"] > shuffled["actual_high_minus_low"]
            for candidate, shuffled in ((history_followup, shuffled_followup),
                                        (history_supported, shuffled_supported)))
        contact_consistent = (
            history_contact.get("status") == "COMPLETED"
            and history_contact["actual_high_minus_low"] >= -0.03)
        gate = {"physical_rank_pass": bool(physical_pass),
                "shuffled_control_separation": bool(control_separation),
                "contact_direction_consistent": bool(contact_consistent),
                "prespecified_continue_gate_passed": bool(
                    physical_pass and control_separation and contact_consistent)}
        observed_ate = {
            target: _weighted_effect(
                pooled_test["outcomes"][:, i].numpy(), pooled_test["assignment"].numpy(),
                pooled_test["step"].numpy())
            for i, target in enumerate(CONTINUOUS_TARGETS)
        }
        observed_ate["final_contact"] = _weighted_effect(
            pooled_test["final_contact"].numpy(), pooled_test["assignment"].numpy(),
            pooled_test["step"].numpy())
        report = {
            "schema": "ref2dex.history_value_cm_probe.v1",
            "run_status": "COMPLETED", "git_commit": manifest["git_commit"],
            "train_samples": len(train["step"]),
            "test_samples": {str(seed): len(all_rows[seed]["step"]) for seed in TEST_SEEDS},
            "continuous_targets": CONTINUOUS_TARGETS,
            "binary_targets": BINARY_TARGETS,
            "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
            "training": training, "prediction": prediction_reports,
            "ranking_pooled_test": ranking,
            "ranking_per_seed": ranking_per_seed,
            "prespecified_gate": gate,
            "observed_ate_pooled": observed_ate,
            "limits": [
                "sequential randomization estimates conditional population effects on reached states",
                "five-step outcomes are not long-horizon policy utility or held-lift",
                "separate heads are retained; no hand-written scalar value target is used",
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for variant, model in models.items():
            torch.save({"schema": "ref2dex.history_value_cm.v1", "variant": variant,
                        "model": model.state_dict(),
                        "feature_mean": feature_stats[variant][0],
                        "feature_std": feature_stats[variant][1],
                        "target_mean": target_mean, "target_std": target_std},
                       args.output / f"{variant}.pt")
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path),
                        checkpoint_sha256={variant: sha256(args.output / f"{variant}.pt")
                                           for variant in variants})
        print(json.dumps({"run_status": "COMPLETED", "ranking": ranking,
                          "prediction": prediction_reports}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}",
                        completed_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
