#!/usr/bin/env python3
"""Read-only statistical audit of a finished relative-action label Probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def correlation(x, y):
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def agreement_detail(data, mask):
    advantage, labels = data["advantages"][:, mask], data["individual_labels"][:, mask]
    union = (labels != 0).any(0)
    both = (labels != 0).all(0)
    return {"rows": int(mask.sum()), "extreme_union": int(union.sum()),
            "extreme_intersection": int(both.sum()),
            "union_agreement": float((labels[0, union] == labels[1, union]).float().mean()) if union.any() else None,
            "both_extreme_sign_agreement": float((labels[0, both] == labels[1, both]).float().mean()) if both.any() else None,
            "opposite_on_union": float((labels[0, union] * labels[1, union] == -1).float().mean()) if union.any() else None,
            "neutral_disagreement_on_union": float(((labels[0, union] == 0) | (labels[1, union] == 0)).float().mean()) if union.any() else None,
            "continuous_advantage_pearson": correlation(advantage[0], advantage[1]),
            "median_absolute_advantage_difference": float((advantage[0] - advantage[1]).abs().median()),
            "advantage_rmse_difference": float((advantage[0] - advantage[1]).square().mean().sqrt())}


def audit(directory):
    torch.set_num_threads(2)
    result = json.loads((directory / "result.json").read_text())
    data = torch.load(directory / "labels.pt", map_location="cpu", weights_only=False)
    recomputed = data["local"][None] + data["discount"][None] * data["future_value"] - data["current_value"]
    assert torch.equal(recomputed, data["advantages"])
    terminal = data["endpoint_rows"] < 0
    assert not data["discount"][terminal].any()
    assert not data["future_value"][:, terminal].any()
    train_ids, test_ids = set(result["train_episode_ids"]), set(result["test_episode_ids"])
    assert not train_ids & test_ids
    for models in result["folds"]:
        excluded_all = []
        for fold in models:
            excluded, fit = set(fold["excluded_episode_ids"]), set(fold["fit_episode_ids"])
            assert fit == train_ids - excluded and not fit & test_ids
            excluded_all.extend(excluded)
        assert sorted(excluded_all) == sorted(train_ids)
    audit = {"run_id": result["run_id"], "git_commit": result["git_commit"],
             "labels_sha256": hashlib.sha256((directory / "labels.pt").read_bytes()).hexdigest(),
             "advantage_reconstruction_max_error": 0., "terminal_queries": int(terminal.sum()),
             "terminal_bootstrap_zero": True, "folds_whole_episode_excluded": True,
             "g0_pass": result["g0_pass"], "g1_executed": result["g1_executed"],
             "interpretation": "Continuous trends correlate; discrete label contract is unstable. No action-information conclusion."}
    gamma = []
    for item in result["inputs"]:
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        shard = torch.load(path, map_location="cpu", weights_only=False)
        gamma.append(float(shard["gamma"]))
    assert len(set(gamma)) == 1
    audit["source_gamma"] = gamma[0]
    for name, split in [("train", data["train"]), ("test", ~data["train"])]:
        largest = max((result["train_episode_ids"] if name == "train" else result["test_episode_ids"]),
                      key=lambda i: abs(result["episodes"][i]["mc_mean"]))
        audit[name] = {"all": agreement_detail(data, split),
                       "real_n32_reward_active": agreement_detail(data, split & data["reward_active"]),
                       "real_n32_reward_zero": agreement_detail(data, split & ~data["reward_active"]),
                       "excluding_largest_mc_episode": agreement_detail(data, split & (data["episode_index"] != largest)),
                       "largest_mc_episode_key": result["episodes"][largest]["key"]}
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    findings = audit(args.directory)
    with args.output.open("x") as stream:
        stream.write(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))
