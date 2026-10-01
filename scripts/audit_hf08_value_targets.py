#!/usr/bin/env python3
"""CPU-only HF08 value-target audit.

This is a read-only diagnostic.  It verifies the native HF08 manifests, reuses
the frozen r7 collections, computes episode-level discounted reward/component
summaries, and evaluates the frozen largest-tier value network on a deterministic
holdout sample.  It does not fit, collect, or alter a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent")
sys.path.insert(0, str(SOURCE))
RUN = SOURCE / "src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7"
MODEL_DIR = RUN / "models"
COMPONENTS = ("base_source", "approach", "held", "progress", "stable")
PHASES = ("pre_contact", "contact", "held", "drop", "other")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def phase_labels(data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Assign a mutually exclusive physical phase to every transition.

    The stored state contains hand/object contact at 49:51 and tracker events
    run/ever/lost/dropped at 51:55.  ``drop`` is an event edge, so persistent
    dropped rows are not counted as repeated drop events.
    """
    n = len(data["state"])
    out = torch.full((n,), 4, dtype=torch.int64)
    state, episode = data["state"], data["episode_id"]
    pair = state[:, 49:51].bool().all(-1)
    run = state[:, 51]
    ever = state[:, 52] > 0.5
    dropped = state[:, 54] > 0.5
    previous = torch.zeros_like(dropped)
    previous[1:] = dropped[:-1] & (episode[1:] == episode[:-1])
    drop_edge = dropped & ~previous
    out[(~pair) & (~ever)] = 0
    out[pair & (run < 1.0) & (~drop_edge)] = 1
    out[(run >= 1.0) & (~drop_edge)] = 2
    out[drop_edge] = 3
    return out


def safe_stats(target: torch.Tensor, pred: torch.Tensor) -> dict:
    err = pred - target
    return {
        "rows": int(len(target)),
        "episodes": None,
        "rmse": float(err.square().mean().sqrt()) if len(target) else None,
        "mae": float(err.abs().mean()) if len(target) else None,
        "bias": float(err.mean()) if len(target) else None,
    }


def add_episode_count(stats: dict, ids: torch.Tensor) -> dict:
    stats["episodes"] = int(torch.unique(ids).numel()) if len(ids) else 0
    return stats


def inspect_online_checkpoint(path: Path) -> dict:
    """Inspect saved e420 parameter/stat containers without replaying them."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    physical = payload.get("physical_value", {})
    value_state = physical.get("value", {})
    model_state = payload.get("model", {})
    critic_state = {k: v for k, v in model_state.items()
                    if "critic_mlp" in k or k.endswith("a2c_network.value.weight")
                    or k.endswith("a2c_network.value.bias")}
    def state_info(state):
        tensors = [v for v in state.values() if isinstance(v, torch.Tensor)]
        return {"tensor_count": len(tensors), "parameter_count": int(sum(v.numel() for v in tensors)),
                "all_finite": bool(all(torch.isfinite(v).all() for v in tensors))}
    rms = payload.get("running_mean_std", {})
    return {
        "path": str(path), "sha256": sha256(path), "epoch": payload.get("epoch"),
        "frame": payload.get("frame"), "arm": physical.get("arm"),
        "physical_value_state": state_info(value_state),
        "ppo_critic_state": state_info(critic_state),
        "running_mean_std": {"keys": sorted(rms.keys()),
                              "count": float(rms["count"]) if isinstance(rms.get("count"), torch.Tensor) else rms.get("count"),
                              "all_finite": bool(all(torch.isfinite(v).all() for v in rms.values() if isinstance(v, torch.Tensor)))},
        "saved_teacher_labels": False,
        "saved_return_targets": False,
        "behavior_integrity_verified": physical.get("behavior_integrity_verified"),
        "actor_supervision_gradient_verified": physical.get("actor_supervision_gradient_verified"),
    }


def aggregate_episodes(dataset, phases):
    data = dataset.data
    rows = []
    by_group = defaultdict(lambda: {"episodes": 0, "rows": 0, "discounted": [0.0] * 5})
    motion_totals = defaultdict(lambda: {"episodes": 0, "rows": 0, "discounted": [0.0] * 5})
    overall = {"episodes": 0, "rows": 0, "discounted": [0.0] * 5}
    episode_success = {}
    episode_drop = {}
    episode_motion = {}
    episode_component = {}
    gamma = float(dataset.data["return"][0] * 0 + 0.99)
    for start, end in zip(dataset.starts.tolist(), dataset.ends.tolist()):
        sl = slice(start, end + 1)
        eid = int(data["episode_id"][start])
        motion = int(data["motion_id"][start])
        comp = data["reward_components"][sl].double()
        length = len(comp)
        weights = torch.pow(torch.tensor(gamma, dtype=torch.float64), torch.arange(length, dtype=torch.float64))
        discounted = (comp * weights[:, None]).sum(0)
        st = data["state"][sl]
        run = st[:, 51]
        success = bool((run >= 1.0).any())
        dropped = st[:, 54] > 0.5
        prev_success = torch.cumsum((run >= 1.0).to(torch.int64), 0) > 0
        # An event after first stable hold is the closest stored proxy for
        # drop-after-success (the tracker does not persist this separate flag).
        drop_edge = dropped.clone()
        if len(drop_edge) > 1:
            drop_edge[1:] &= ~dropped[:-1]
        drop_after = bool((drop_edge & prev_success).any())
        episode_success[eid] = success
        episode_drop[eid] = drop_after
        episode_motion[eid] = motion
        episode_component[eid] = discounted.tolist()
        rows.append({"episode_id": eid, "motion_id": motion, "rows": length,
                     "success": success, "drop_after_success_proxy": drop_after,
                     "discounted": discounted.tolist()})
        motion_totals[motion]["episodes"] += 1
        motion_totals[motion]["rows"] += length
        motion_totals[motion]["discounted"] = [a + float(b) for a, b in zip(motion_totals[motion]["discounted"], discounted)]
        overall["episodes"] += 1
        overall["rows"] += length
        overall["discounted"] = [a + float(b) for a, b in zip(overall["discounted"], discounted)]
        phase = phases[sl]
        for pidx, pname in enumerate(PHASES):
            mask = phase == pidx
            if not bool(mask.any()):
                continue
            part = (comp * (weights[:, None] * mask[:, None])).sum(0)
            g = by_group[(motion, pname)]
            g["episodes"] += 1
            g["rows"] += int(mask.sum())
            g["discounted"] = [a + float(b) for a, b in zip(g["discounted"], part)]
    def finalize(table):
        result = {}
        for key, value in table.items():
            if isinstance(key, tuple):
                k = f"motion_{key[0]}::{key[1]}"
            else:
                k = str(key)
            e = value["episodes"]
            result[k] = {"episodes": e, "rows": value["rows"],
                         "mean_discounted": [v / e for v in value["discounted"]] if e else [None] * 5,
                         "sum_discounted": value["discounted"]}
        return result
    # Aggregate success/drop by episode and motion, never by adjacent rows.
    event_by_motion = {}
    for motion in sorted(set(episode_motion.values())):
        ids = [e for e, m in episode_motion.items() if m == motion]
        event_by_motion[str(motion)] = {
            "episodes": len(ids),
            "stable_success_episodes": int(sum(episode_success[e] for e in ids)),
            "drop_after_success_proxy_episodes": int(sum(episode_drop[e] for e in ids)),
        }
    overall_sum = sum(overall["discounted"])
    overall_summary = {
        "episodes": overall["episodes"], "rows": overall["rows"],
        "mean_discounted": [x / overall["episodes"] for x in overall["discounted"]],
        "sum_discounted": overall["discounted"],
        "signed_share_of_total": [x / overall_sum for x in overall["discounted"]] if overall_sum else [None] * 5,
    }
    return {
        "episode_count": len(rows),
        "episode_rows": rows,
        "motion": finalize(motion_totals),
        "overall": overall_summary,
        "motion_phase": finalize(by_group),
        "events_by_motion": event_by_motion,
        "events": {
            "stable_success_episodes": int(sum(episode_success.values())),
            "drop_after_success_proxy_episodes": int(sum(episode_drop.values())),
        },
        "episode_success": episode_success,
        "episode_drop": episode_drop,
        "episode_motion": episode_motion,
    }


@torch.no_grad()
def evaluate_value(dataset, payload, phases, episode_success, episode_drop, limit=2048):
    from src.task.CmResidual.physical_value_data import HISTORY
    from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork

    device = "cpu"
    features = Features(**{k: v.to(device) for k, v in payload["stats"].items()}, device=device).to(device)
    network = OutcomeNetwork(payload["context_dim"], 0, 1, 9283).to(device)
    network.load_state_dict(payload["value"])
    network.eval()
    generator = torch.Generator().manual_seed(20260930285)
    hold = dataset.hold[torch.randperm(len(dataset.hold), generator=generator)[:limit]]
    fit = dataset.fit
    fit_mean = dataset.data["return"][fit].mean()
    # Fit-only motion/phase means; a missing cell falls back to fit motion then
    # global fit mean and is reported as a fallback rather than silently filled.
    means = {}
    motion_means = {}
    sums = defaultdict(float); counts = defaultdict(int)
    msums = defaultdict(float); mcounts = defaultdict(int)
    for idx in fit.tolist():
        m, p = int(dataset.data["motion_id"][idx]), int(phases[idx])
        y = float(dataset.data["return"][idx])
        sums[(m, p)] += y; counts[(m, p)] += 1
        msums[m] += y; mcounts[m] += 1
    for k, v in sums.items(): means[k] = v / counts[k]
    for k, v in msums.items(): motion_means[k] = v / mcounts[k]
    actual, pred, baseline, motion_phase, selected_phase, selected_success, selected_drop, selected_motion, selected_episode = [], [], [], [], [], [], [], [], []
    for indices in hold.split(128):
        batch = dataset.batch(indices, device)
        history = features.history(batch["history_state"], batch["history_action"], batch["history_mask"])
        context = features.context(batch["context"])
        value = network(history, context).squeeze(-1).cpu()
        actual.append(batch["return"].cpu())
        pred.append(value)
        for i in indices.tolist():
            m, p = int(dataset.data["motion_id"][i]), int(phases[i])
            motion_phase.append(means.get((m, p), motion_means.get(m, float(fit_mean))))
        selected_phase.append(phases[indices].clone())
        selected_motion.append(dataset.data["motion_id"][indices].clone())
        selected_episode.append(dataset.data["episode_id"][indices].clone())
        selected_success.append(torch.tensor([episode_success[int(x)] for x in dataset.data["episode_id"][indices]], dtype=torch.bool))
        selected_drop.append(torch.tensor([episode_drop[int(x)] for x in dataset.data["episode_id"][indices]], dtype=torch.bool))
    actual, pred, motion_phase = torch.cat(actual), torch.cat(pred), torch.tensor(motion_phase)
    selected_phase, selected_motion = torch.cat(selected_phase), torch.cat(selected_motion)
    selected_episode = torch.cat(selected_episode)
    selected_success, selected_drop = torch.cat(selected_success), torch.cat(selected_drop)
    # Keep a row-level table with episode counts so adjacent transitions are
    # never presented as independent successes.
    groups = {}
    for name, mask in {
        "all": torch.ones(len(actual), dtype=torch.bool),
        "success": selected_success,
        "failure": ~selected_success,
        "drop_after_success_proxy": selected_drop,
        "no_drop_after_success_proxy": ~selected_drop,
    }.items():
        groups[name] = add_episode_count(safe_stats(actual[mask], pred[mask]), selected_episode[mask])
        groups[name]["motion_phase_baseline"] = safe_stats(actual[mask], motion_phase[mask])
        groups[name]["constant_fit_baseline"] = safe_stats(actual[mask], torch.full_like(actual[mask], fit_mean))
    phase_groups = {}
    for pidx, pname in enumerate(PHASES):
        mask = selected_phase == pidx
        phase_groups[pname] = add_episode_count(safe_stats(actual[mask], pred[mask]), selected_episode[mask])
        phase_groups[pname]["motion_phase_baseline"] = safe_stats(actual[mask], motion_phase[mask])
        phase_groups[pname]["constant_fit_baseline"] = safe_stats(actual[mask], torch.full_like(actual[mask], fit_mean))
    motion_groups = {}
    for motion in sorted(set(selected_motion.tolist())):
        mask = selected_motion == motion
        motion_groups[str(motion)] = add_episode_count(safe_stats(actual[mask], pred[mask]), selected_episode[mask])
        motion_groups[str(motion)]["motion_phase_baseline"] = safe_stats(actual[mask], motion_phase[mask])
        motion_groups[str(motion)]["constant_fit_baseline"] = safe_stats(actual[mask], torch.full_like(actual[mask], fit_mean))
    fallback = sum(1 for i in hold.tolist() if (int(dataset.data["motion_id"][i]), int(phases[i])) not in means)
    return {
        "rows": int(len(actual)), "episodes": int(torch.unique(selected_episode).numel()),
        "fit_rows": int(len(fit)), "fit_mean_return": float(fit_mean),
        "fit_motion_phase_baseline_fallback_rows": int(fallback),
        "overall": groups, "phase": phase_groups, "motion": motion_groups,
        "sample_seed": 20260930285,
        "selection": "dataset.hold deterministic permutation, first 2048 rows; episode-level counts reported",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    # Import after thread setup and never touch CUDA.
    from src.task.CmResidual.physical_value_data import Episodes

    full_audit = RUN / "full_collection_audit.json"
    model_manifest = MODEL_DIR / "run_manifest.json"
    model_results = MODEL_DIR / "results.json"
    checkpoint = MODEL_DIR / "tier_1000000.pt"
    collections = [RUN / "collect_s283", RUN / "collect_s284"]
    full = json.loads(full_audit.read_text())
    manifest = json.loads(model_manifest.read_text())
    model_result_payload = json.loads(model_results.read_text())
    expected = {x["path"]: x["sha256"] for x in full["inputs"]}
    verified_inputs = {}
    for path_text, digest in expected.items():
        path = Path(path_text)
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(f"INPUT_HASH_DRIFT {path} expected={digest} actual={actual}")
        verified_inputs[str(path)] = actual
    expected_model = model_result_payload["tiers"]["1000000"]["sha256"]
    actual_model = sha256(checkpoint)
    if actual_model != expected_model:
        raise RuntimeError(f"MODEL_HASH_DRIFT expected={expected_model} actual={actual_model}")
    # Native collection result hashes are a second provenance check.
    result_hashes = {str(p / "results.json"): sha256(p / "results.json") for p in collections}
    for path_text, digest in manifest["collection_results_sha256"].items():
        if result_hashes.get(path_text) != digest:
            raise RuntimeError(f"RESULT_HASH_DRIFT {path_text}")
    dataset = Episodes(collections, gamma=.99)
    if not torch.isfinite(dataset.data["return"]).all():
        raise RuntimeError("NONFINITE_RETURN")
    phases = phase_labels(dataset.data)
    aggregates = aggregate_episodes(dataset, phases)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("gamma") != .99 or payload.get("fit_rows") != len(dataset.fit):
        raise RuntimeError("MODEL_CONTRACT_MISMATCH")
    value = evaluate_value(dataset, payload, phases, aggregates["episode_success"], aggregates["episode_drop"])
    # Existing online e420 artifact is only inspected; no replay/recalibration.
    online = {}
    for seed, name in (("s286", "component_audit_online_s286_e420.json"), ("s287", "component_audit_online_s287_e420.json")):
        report = json.loads((MODEL_DIR / name).read_text())
        online[seed] = {
            "artifact_sha256": sha256(MODEL_DIR / name),
            "checkpoint_sha256": report.get("online_dynamics", {}).get("checkpoint_sha256"),
            "epoch": report.get("online_dynamics", {}).get("epoch"),
            "frame": report.get("online_dynamics", {}).get("frame"),
            "initial_model_sha256": report.get("online_dynamics", {}).get("initial_model_sha256"),
            "diagnostics": report.get("diagnostics", {}),
            "scope": report.get("scope"),
        }
        candidates = list((RUN / f"train_cm_value_{seed}").glob("**/GRAB_00000420.pth"))
        if len(candidates) != 1:
            raise RuntimeError(f"ONLINE_CHECKPOINT_PROVENANCE {seed}: {len(candidates)} candidates")
        inspected = inspect_online_checkpoint(candidates[0])
        if inspected["sha256"] != online[seed]["checkpoint_sha256"]:
            raise RuntimeError(f"ONLINE_CHECKPOINT_HASH_DRIFT {seed}")
        online[seed]["parameter_inspection"] = inspected
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        commit = "unavailable"
    elapsed = time.monotonic() - started
    output = {
        "run_status": "COMPLETED",
        "task_id": "T-20261001-hf08-value-target-audit",
        "git_commit": commit,
        "cpu_only": {"device": "cpu", "torch_threads": 2, "cuda_used": False, "gpu": 0},
        "elapsed_seconds": elapsed,
        "output_bytes": None,
        "contract": {"schema": "ref2dex.physical_value.v1", "gamma": .99,
                     "offline_target": "complete episode Monte Carlo return",
                     "episode_split": "episode_id remainder 10 < 2 holdout; no cross-episode rows",
                     "online_target": "PPO GAE(lambda) return: delta=r+gamma*(1-terminate)V(next)-V; recursive gamma*tau with gamma=.99,tau=.95; bootstrap zero only info['terminate']",
                     "reward_components": list(COMPONENTS),
                     "base_component_note": "stored base_source is original_reward; source code does not expose an independent tracking-only column"},
        "input_hashes": {"full_collection_audit": sha256(full_audit), "model_manifest": sha256(model_manifest),
                         "model_results": sha256(model_results), "checkpoint": actual_model,
                         "collection_results": result_hashes, "transition_shards": verified_inputs},
        "dataset": {"rows": int(len(dataset.data["state"])), "fit_rows": int(len(dataset.fit)),
                    "holdout_rows": int(len(dataset.hold)), "episodes": int(len(dataset.starts)),
                    "fit_episodes": int(len(dataset.episode_ids) - int((dataset.episode_ids.remainder(10) < 2).sum())),
                    "holdout_episodes": int((dataset.episode_ids.remainder(10) < 2).sum()),
                    "motions": sorted(set(int(x) for x in dataset.data["motion_id"].tolist()))},
        "discounted_reward_contributions": {k: v for k, v in aggregates.items() if not k.startswith("episode_")},
        "value_holdout": value,
        "online_e420_inspection": online,
        "decision": {
            "classification": "TRAINING_SUFFICIENCY_OR_DISTRIBUTION_UNKNOWN",
            "recommendation": "Do not repair or train V from these diagnostics alone; before any further Cm policy training, collect current-policy complete trajectories with terminal provenance and evaluate V against those returns.",
            "reason_codes": ["SOURCE_MC_NOT_CURRENT_POLICY_TRUTH", "NO_COUNTERFACTUAL_RANKING", "HOLDOUT_PHASE_SAMPLE_DIAGNOSTIC", "NATIVE_ENDPOINT_GATE_UNPROMISING"],
            "policy_utility_claim": False,
            "loss_convergence_claim": False,
        },
        "reproduction": {"command": "python3 scripts/audit_hf08_value_targets.py --output docs/handoffs/HF08_VALUE_TARGET_AUDIT_20261001.json", "threads": 2},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    output["output_bytes"] = args.output.stat().st_size
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"status": output["run_status"], "elapsed_seconds": elapsed,
                      "rows": output["dataset"]["rows"], "episodes": output["dataset"]["episodes"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
