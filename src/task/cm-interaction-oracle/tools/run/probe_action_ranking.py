#!/usr/bin/env python3
"""Ref2 Decision Probe: frozen advantages, matched H versus current-action ranking."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from action_ranking import (compare_arms, conditional_pairs, fit_ranking,
                            prepare_features, target_reliability)
from probe_relative_action_critic import assemble, digest


def load_frozen(label_directory):
    result = json.loads((label_directory / "result.json").read_text())
    label = torch.load(label_directory / "labels.pt", map_location="cpu", weights_only=False)
    data = assemble([Path(row["path"]) for row in result["inputs"]], 128)
    if data["hashes"] != result["inputs"]:
        raise ValueError("source files changed after advantage fitting")
    for field in ["current_rows", "endpoint_rows", "local", "discount", "strata", "episode_index",
                  "source_row", "step", "reward_active", "train"]:
        if not torch.equal(data[field], label[field]):
            raise ValueError("frozen label/source query join mismatch: " + field)
    if data["train_episode_ids"] != result["train_episode_ids"] or data["test_episode_ids"] != result["test_episode_ids"]:
        raise ValueError("outer episode split changed")
    if not torch.equal(label["local"][None] + label["discount"][None] * label["future_value"] - label["current_value"], label["advantages"]):
        raise ValueError("advantage reconstruction mismatch")
    return data, label["advantages"], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs/cm-interaction-oracle/recap-relative-action-s203-s204")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this protocol uses GPU for model fitting/inference")
    out = ROOT / "outputs/cm-interaction-oracle" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    emit = lambda message: print(json.dumps(message), flush=True)
    seed, epochs = (7, 1) if args.smoke else (207, 16)
    labels = args.labels.resolve()
    manifest = {"experiment_id": "P-20261004-relative-action-ranking", "run_id": args.run_id,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "training_seed": seed, "epochs": epochs, "pair_seed": 207, "permutation_seed": 208,
                "mode": "engineering_smoke" if args.smoke else "Decision_Probe",
                "label_directory": str(labels), "labels_sha256": digest(labels / "labels.pt"),
                "label_result_sha256": digest(labels / "result.json"), "label_manifest_sha256": digest(labels / "manifest.json"),
                "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                [Path(__file__), TASK / "src/action_ranking.py", TASK / "src/relative_action.py",
                                 Path(__file__).parent / "probe_relative_action_critic.py"]},
                "loss": "mean softplus(-(C_i-C_j)*sign(A_i-A_j)); exact target ties skipped",
                "horizon": 32, "action_chunk": {"H": 0, "HaK1": 1, "HaK4_realized": 4}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    data, advantages, source = load_frozen(labels)
    manifest.update(inputs=data["hashes"], gamma=data["gamma"], label_git_commit=source["git_commit"],
                    train_episode_ids=data["train_episode_ids"], test_episode_ids=data["test_episode_ids"],
                    history_dim=int(data["history"].shape[-1]), history_length=10,
                    optimizer="AdamW lr2e-3 weight_decay1e-5 clip2 batch512",
                    queries=int(len(data["local"])))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    train = data["train"]
    train_pairs = conditional_pairs(data["strata"], data["episode_index"], train, 207)
    test_pairs = conditional_pairs(data["strata"], data["episode_index"], ~train, 208)
    reliability = {"train": target_reliability(advantages, data, train, train_pairs),
                   "test": target_reliability(advantages, data, ~train, test_pairs)}
    emit({"frozen_queries": len(advantages[0]), "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
          "target_reliability": reliability})
    torch.save({"train_pairs": train_pairs, "test_pairs": test_pairs, "advantages": advantages,
                "source_row": data["source_row"], "step": data["step"], "episode_index": data["episode_index"],
                "strata": data["strata"], "train": train}, out / "pairs_and_targets.pt")
    history, action, normalization = prepare_features(data, device)
    arms, scores, swapped_scores = {}, {}, {}
    for variant in ["H", "HaK1", "HaK4_realized"]:
        metric, score, swapped, weights = fit_ranking(history, action, data, advantages.mean(0),
                                                     train_pairs, test_pairs, variant, seed, device, epochs)
        arms[variant], scores[variant], swapped_scores[variant] = metric, score, swapped
        torch.save({"state_dict": weights, "normalization": normalization, "variant": variant}, out / (variant + ".pt"))
        emit({"variant": variant, "parameters": metric["parameters"], "final_loss": metric["loss"][-1],
              "test_macro_pair_accuracy": metric["test"]["macro_stratum_pair_accuracy"],
              "test_macro_spearman": metric["test"]["macro_stratum_spearman"],
              "test_permuted_macro_pair_accuracy": metric["test_action_permuted"]["macro_stratum_pair_accuracy"]})
    assert len({arm["parameters"] for arm in arms.values()}) == 1
    assert len({arm["initial_state_sha256"] for arm in arms.values()}) == 1
    assert torch.equal(scores["H"], swapped_scores["H"]), "constant action control must be invariant"
    comparison = compare_arms(arms, scores, advantages, data, test_pairs)
    result = {**manifest, "target_reliability": reliability, "arms": arms, "comparison": comparison,
              "episodes": [{k: v for k, v in ep.items() if k != "fit_rows"} for ep in data["episodes"]],
              "status": "PROMISING" if comparison["pass"] else "UNPROMISING",
              "reason": "current-action ranking gate passes; next require matched consequence and physical candidate probes" if comparison["pass"] else
                        "no robust deployable K1 action-ranking gain under this frozen label/data/fit contract",
              "elapsed_seconds": time.monotonic() - started,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    if args.smoke:
        result["status"], result["reason"] = "engineering_smoke", "one-epoch wiring only; no scientific decision"
    torch.save({"scores": scores, "permuted_scores": swapped_scores}, out / "scores.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    emit({"status": result["status"], "comparison": comparison, "elapsed_seconds": result["elapsed_seconds"]})


if __name__ == "__main__":
    main()
