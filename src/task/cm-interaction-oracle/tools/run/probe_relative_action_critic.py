#!/usr/bin/env python3
"""Bounded G0/G1 gate: cross-fitted MC state value -> n-step action labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from relative_action import (crossfit_values, discounted_returns, episode_split,
                             fit_critic, fit_thresholds, label_advantage, nstep_outcome)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def samples(start, end, count):
    return torch.linspace(start, end, min(count, end - start + 1)).round().long().unique()


def assemble(paths, per_episode=128):
    histories, returns, episodes = [], [], []
    packets = {k: [] for k in ["current_rows", "endpoint_rows", "local", "discount", "action",
                              "strata", "episode_index", "source_row", "step", "reward_active"]}
    hashes, offset, gamma = [], 0, None
    for run, path in enumerate(paths):
        hashes.append({"path": str(path), "sha256": digest(path)})
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if shard["physical_timing"] != "pre_env_step":
            raise ValueError("expected pre_env_step")
        if gamma is None:
            gamma = float(shard["gamma"])
        assert gamma == float(shard["gamma"])
        assert torch.isfinite(shard["state"]).all() and torch.isfinite(shard["action"]).all()
        for episode in sorted(shard["episode_id"].unique().tolist()):
            ix = (shard["episode_id"] == episode).nonzero().flatten()
            ix = ix[shard["step"][ix].argsort()]
            n = len(ix)
            assert torch.equal(shard["step"][ix], torch.arange(n)), "incomplete/noncontiguous episode"
            assert int(shard["done"][ix].sum()) == 1 and bool(shard["done"][ix[-1]])
            assert torch.equal(shard["previous_action"][ix[1:]], shard["action"][ix[:-1]])
            assert torch.allclose(shard["next_state"][ix[:-1]], shard["state"][ix[1:]], atol=1e-6, rtol=0)
            motion = int(shard["motion_id"][ix[0]])
            noise = float(shard["noise_std"][ix[0]])
            assert (shard["noise_std"][ix] == noise).all()
            assert (shard["motion_id"][ix] == motion).all()
            noise_bin = min(range(4), key=lambda j: abs([0, .05, .1, .2][j] - noise))
            assert abs([0, .05, .1, .2][noise_bin] - noise) < 1e-6
            # Noise regime is a known episode nuisance, supplied to every arm.
            obs = torch.cat([shard["state"][ix], shard["previous_action"][ix],
                             shard["context"][ix], shard["progress"][ix, None].float(),
                             shard["noise_std"][ix, None]], -1)
            h = obs.unfold(0, 10, 1).transpose(1, 2).contiguous()
            comp = shard["reward_components"][ix]
            reward = comp[:, 2] / 10 + comp[:, 3] / 5 + comp[:, 4]
            assert torch.isfinite(reward).all()
            mc = discounted_returns(reward, gamma)
            t = samples(9, n - 4, per_episode)
            local, endpoint, discount = nstep_outcome(reward, t, 32, gamma)
            qrows = offset + t - 9
            erows = torch.where(endpoint >= 0, offset + endpoint - 9, torch.full_like(endpoint, -1))
            ep_idx = len(episodes)
            episodes.append({"key": (0, run, int(episode)), "stratum": (motion, noise_bin),
                             "rows": n, "reward_nonzero_rows": int((reward.abs() > 1e-8).sum()),
                             "mc_mean": float(mc.mean()), "mc_max": float(mc.max()),
                             "fit_rows": offset + samples(9, n - 1, per_episode) - 9})
            packets["current_rows"].append(qrows)
            packets["endpoint_rows"].append(erows)
            packets["local"].append(local)
            packets["discount"].append(discount)
            packets["action"].append(torch.stack([shard["action"][ix[j:j + 4]] for j in t.tolist()]))
            phase = (shard["context"][ix[t], 3] * 8).long().clamp(0, 7)
            packets["strata"].append(torch.stack([torch.full_like(t, motion), phase, torch.full_like(t, noise_bin)], -1))
            packets["episode_index"].append(torch.full_like(t, ep_idx))
            packets["source_row"].append(ix[t])
            packets["step"].append(t)
            packets["reward_active"].append(torch.tensor([bool((reward[j:j + 32].abs() > 1e-8).any()) for j in t.tolist()]))
            histories.append(h)
            returns.append(mc[9:])
            offset += len(h)
    train_keys, test_keys = episode_split([e["key"] for e in episodes])
    train_ids = [i for i, e in enumerate(episodes) if e["key"] in train_keys]
    test_ids = [i for i, e in enumerate(episodes) if e["key"] in test_keys]
    data = {k: torch.cat(v) for k, v in packets.items()}
    data.update(history=torch.cat(histories), returns=torch.cat(returns), episodes=episodes,
                train_episode_ids=train_ids, test_episode_ids=test_ids, hashes=hashes, gamma=gamma)
    data["train"] = torch.tensor([int(i) in train_ids for i in data["episode_index"]])
    return data


def coverage(data, label, subset):
    out = {"queries": int(subset.sum()), "real_n32_reward_fraction": float(data["reward_active"][subset].float().mean())}
    for name, value in [("positive", 1), ("negative", -1)]:
        mask = subset & (label == value)
        counts = torch.bincount(data["episode_index"][mask], minlength=len(data["episodes"]))
        out[name] = {"rows": int(mask.sum()), "episodes": int((counts > 0).sum()),
                     "largest_episode_share": float(counts.max() / counts.sum()) if counts.sum() else 1.0,
                     "real_n32_reward_fraction": float(data["reward_active"][mask].float().mean()) if mask.any() else 0.0}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--rows-per-episode", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.smoke:
        raise ValueError("research model fitting requires GPU under this protocol")
    out = ROOT / "outputs" / "cm-interaction-oracle" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    emit = lambda obj: print(json.dumps(obj), flush=True)
    manifest = {"run_id": args.run_id, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "script_sha256": digest(Path(__file__)), "module_sha256": digest(TASK / "src" / "relative_action.py"),
                "device": str(device), "args": vars(args), "seeds": [203, 204], "split_id": 20261004,
                "mode": "engineering_smoke" if args.smoke else "Decision_Probe", "reward": "held/10 + lift_progress/5 + stable",
                "horizon": 32, "history": 10, "noise_conditioned": True}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    paths = [ROOT / "tmp" / run / "transitions_000.pt" for run in
             ["fresh_pre_s86e", "fresh_pre_s86f", "fresh_pre_s86g_retry", "fresh_pre_s86h_retry"]]
    data = assemble(paths, args.rows_per_episode)
    emit({"assembled": len(data["history"]), "queries": len(data["local"]), "train_episodes": len(data["train_episode_ids"]),
          "test_episodes": len(data["test_episode_ids"]), "gamma": data["gamma"]})
    manifest["inputs"] = data["hashes"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    seeds = [203, 204]
    runs = [crossfit_values(data, seed, device, args.epochs, emit) for seed in seeds]
    advs = torch.stack([r[0] for r in runs])
    individual = []
    for advantage in advs:
        threshold = fit_thresholds(advantage, data["strata"], data["train"])
        individual.append(label_advantage(advantage, data["strata"], threshold))
    advantage = advs.mean(0)
    thresholds = fit_thresholds(advantage, data["strata"], data["train"])
    label = label_advantage(advantage, data["strata"], thresholds)
    union = (individual[0] != 0) | (individual[1] != 0)
    stability = {}
    for name, subset in [("train", data["train"]), ("test", ~data["train"])]:
        mask = union & subset
        stability[name] = float((individual[0][mask] == individual[1][mask]).float().mean()) if mask.any() else 0.0
    cover = {"train": coverage(data, label, data["train"]), "test": coverage(data, label, ~data["train"])}
    # Model skill is evaluated in MC units on the outer held-out episodes.
    target = data["returns"][data["current_rows"]]
    predicted_v = torch.stack([r[1] for r in runs]).mean(0)
    test = ~data["train"]
    v_mse = float((predicted_v[test] - target[test]).square().mean())
    zero_mse = float(target[test].square().mean())
    gates = {"fit_stability_train": stability["train"] >= .75, "fit_stability_test": stability["test"] >= .75,
             "value_beats_zero_mc": v_mse < zero_mse}
    for cls in ["positive", "negative"]:
        gates[cls + "_train_count"] = cover["train"][cls]["rows"] >= 500
        gates[cls + "_test_count"] = cover["test"][cls]["rows"] >= 100
        gates[cls + "_test_episode_support"] = cover["test"][cls]["episodes"] >= 12
        gates[cls + "_test_concentration"] = cover["test"][cls]["largest_episode_share"] <= .15
    g0 = all(gates.values()) and not args.smoke
    result = {**manifest, "g0_pass": g0, "g0_gates": gates, "stability": stability, "coverage": cover,
              "value_test_mse": v_mse, "zero_value_test_mse": zero_mse,
              "threshold_strata": len(thresholds), "folds": [r[3] for r in runs],
              "episodes": [{k: v for k, v in e.items() if k != "fit_rows"} for e in data["episodes"]],
              "train_episode_ids": data["train_episode_ids"], "test_episode_ids": data["test_episode_ids"],
              "g1_executed": False, "status": "UNCLEAR", "reason": "G0 label reliability gate failed; do not train critic or Cm"}
    packet = {k: v for k, v in data.items() if torch.is_tensor(v) and k not in ["history", "returns", "action"]}
    packet.update(advantages=advs, current_value=torch.stack([r[1] for r in runs]),
                  future_value=torch.stack([r[2] for r in runs]), labels=label, individual_labels=torch.stack(individual),
                  thresholds=thresholds, mc=target)
    torch.save(packet, out / "labels.pt")
    torch.save([r[4] for r in runs], out / "value_models.pt")
    emit({"g0_pass": g0, "gates": gates, "stability": stability, "coverage": cover})
    if g0:
        result["g1_executed"] = True
        critics, scores = {}, {}
        for variant in ["H", "HaK1", "HaK4_realized"]:
            metric, score, weights = fit_critic(data, label, variant, 203, device, args.epochs)
            critics[variant], scores[variant] = metric, score
            torch.save(weights, out / (variant + ".pt"))
            emit(metric)
        torch.save(scores, out / "critic_scores.pt")
        result["critics"] = critics
        h, ha = critics["H"]["metrics"], critics["HaK1"]["metrics"]
        swap = critics["HaK1"]["shuffled_test_metrics"]
        differences = []
        for ep in data["test_episode_ids"]:
            hm, am = critics["H"]["per_episode"][str(ep)], critics["HaK1"]["per_episode"][str(ep)]
            if hm and am:
                differences.append(am["balanced_accuracy"] - hm["balanced_accuracy"])
        largest = max(data["test_episode_ids"], key=lambda i: abs(data["episodes"][i]["mc_mean"]))
        remaining = [critics["HaK1"]["per_episode"][str(i)]["balanced_accuracy"] - critics["H"]["per_episode"][str(i)]["balanced_accuracy"]
                     for i in data["test_episode_ids"] if i != largest and critics["H"]["per_episode"][str(i)]]
        bootstrap = torch.tensor(differences)[torch.randint(len(differences), (2000, len(differences)), generator=torch.Generator().manual_seed(203))].mean(1)
        ci = torch.quantile(bootstrap, torch.tensor([.025, .975])).tolist()
        result["g1_gates"] = {"ba_gain": ha["balanced_accuracy"] - h["balanced_accuracy"] >= .03,
                              "auc_gain": ha["auc"] - h["auc"] >= .03,
                              "shuffled_action_reduces_gain": ha["balanced_accuracy"] - swap["balanced_accuracy"] >= .02,
                              "episode_delta_ci_positive": ci[0] > 0,
                              "excluding_largest_episode_positive": bool(remaining) and sum(remaining) / len(remaining) > 0}
        result["episode_delta_ci95"] = ci
        result["g1_pass"] = all(result["g1_gates"].values())
        result["status"] = "PROMISING" if result["g1_pass"] else "UNPROMISING"
        result["reason"] = "proceed to predicted consequence controls" if result["g1_pass"] else "K1 action-information gate failed in this data/fit; no Cm expansion"
    result["elapsed_seconds"] = time.monotonic() - started
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    (out / "result.json").write_text(json.dumps(result, indent=2))
    emit({"status": result["status"], "reason": result["reason"], "elapsed_seconds": result["elapsed_seconds"]})


if __name__ == "__main__":
    main()
