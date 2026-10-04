#!/usr/bin/env python3
"""Independent CPU raw-score audit; no model fitting or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import scipy
import torch
from scipy.stats import spearmanr

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def independent_metrics(score, target, data, pairs, subset):
    direction = target[pairs[:, 0]] - target[pairs[:, 1]]
    difference = score[pairs[:, 0]] - score[pairs[:, 1]]
    valid = direction != 0
    credit = torch.where(difference == 0, torch.full_like(difference, .5), (direction * difference > 0).float())
    pair_groups, rhos, common_rhos, episodes = [], [], [], {}
    pair_keys = set()
    for key in sorted(set(map(tuple, data["strata"][subset].tolist()))):
        rows = subset & (data["strata"] == torch.tensor(key)).all(-1)
        members = (data["strata"][pairs[:, 0]] == torch.tensor(key)).all(-1) & valid
        if members.any():
            pair_groups.append(float(credit[members].mean()))
            pair_keys.add(key)
        if rows.sum() >= 3 and score[rows].std() > 0 and target[rows].std() > 0:
            rho = float(spearmanr(score[rows].numpy(), target[rows].numpy()).statistic)
            rhos.append(rho)
            if members.any():
                common_rhos.append(rho)
    for ep in data["episode_index"][subset].unique().tolist():
        rows = (data["episode_index"][pairs[:, 0]] == ep) & valid
        if rows.any():
            episodes[str(ep)] = float(credit[rows].mean())
    return {"macro_pair_accuracy": sum(pair_groups) / len(pair_groups),
            "macro_spearman": sum(rhos) / len(rhos),
            "macro_spearman_on_pair_supported_strata": sum(common_rhos) / len(common_rhos),
            "pair_supported_strata": len(pair_keys), "spearman_supported_strata": len(rhos),
            "pair_supported_episodes": len(episodes), "per_episode": episodes}


def audit(directory):
    torch.set_num_threads(2)
    result = json.loads((directory / "result.json").read_text())
    data = torch.load(directory / "pairs_and_targets.pt", map_location="cpu", weights_only=False)
    scores = torch.load(directory / "scores.pt", map_location="cpu", weights_only=False)
    labels = Path(result["label_directory"])
    assert digest(labels / "labels.pt") == result["labels_sha256"]
    assert digest(labels / "result.json") == result["label_result_sha256"]
    assert digest(labels / "manifest.json") == result["label_manifest_sha256"]
    frozen = torch.load(labels / "labels.pt", map_location="cpu", weights_only=False)
    assert torch.equal(data["advantages"], frozen["advantages"])
    for field in ["strata", "episode_index", "source_row", "step", "train"]:
        assert torch.equal(data[field], frozen[field])
    for name, value in result["code_sha256"].items():
        assert digest(ROOT / name) == value
    for item in result["inputs"]:
        assert digest(Path(item["path"])) == item["sha256"]
    target, test = data["advantages"].mean(0), ~data["train"]
    pair_counts = {}
    for name, subset in [("train", data["train"]), ("test", test)]:
        pairs = data[name + "_pairs"]
        assert subset[pairs].all()
        assert torch.equal(data["strata"][pairs[:, 0]], data["strata"][pairs[:, 1]])
        assert (data["episode_index"][pairs[:, 0]] != data["episode_index"][pairs[:, 1]]).all()
        pair_counts[name] = len(pairs)
    norms, model_metrics = [], {}
    for name, score in scores["scores"].items():
        saved = torch.load(directory / (name + ".pt"), map_location="cpu", weights_only=False)
        assert sum(v.numel() for v in saved["state_dict"].values()) == result["arms"][name]["parameters"]
        norms.append(saved["normalization"])
        metric = independent_metrics(score, target, data, data["test_pairs"], test)
        reported = result["arms"][name]["test"]
        assert abs(metric["macro_pair_accuracy"] - reported["macro_stratum_pair_accuracy"]) < 1e-7
        assert abs(metric["macro_spearman"] - reported["macro_stratum_spearman"]) < 1e-7
        permuted = scores["permuted_scores"][name]
        metric["permutation_score_rmse"] = float((score[test] - permuted[test]).square().mean().sqrt())
        metric["score_std"] = float(score[test].std())
        metric["permuted_macro_pair_accuracy"] = independent_metrics(permuted, target, data, data["test_pairs"], test)["macro_pair_accuracy"]
        model_metrics[name] = metric
    for norm in norms[1:]:
        assert all(torch.equal(first, other) for first, other in zip(norms[0], norm))
    assert torch.equal(scores["scores"]["H"], scores["permuted_scores"]["H"])
    h, ha = model_metrics["H"], model_metrics["HaK1"]
    differences = torch.tensor([ha["per_episode"][ep] - value for ep, value in h["per_episode"].items()])
    ci = differences[torch.randint(len(differences), (2000, len(differences)), generator=torch.Generator().manual_seed(207))].mean(1)
    ci = torch.quantile(ci, torch.tensor([.025, .975])).tolist()
    assert ci == result["comparison"]["episode_mean_delta_ci95"]
    stability = {}
    for name, subset in [("train", data["train"]), ("test", test)]:
        rho = float(spearmanr(data["advantages"][0, subset].numpy(), data["advantages"][1, subset].numpy()).statistic)
        assert abs(rho - result["target_reliability"][name]["global_spearman_between_v_fits"]) < 1e-10
        stability[name + "_spearman"] = rho
    return {"run_id": result["run_id"], "git_commit": result["git_commit"], "scipy_version": scipy.__version__,
            "audit_script_sha256": digest(Path(__file__)), "source_code_label_hashes_match": True,
            "frozen_targets_exact": True, "pairs_split_stratum_episode_correct": True,
            "same_normalization": True, "h_permutation_identical": True, "pair_counts": pair_counts,
            "independent_raw_score_metrics": model_metrics, "target_stability": stability,
            "paired_episode_delta_ci95": ci, "paired_episodes": len(differences),
            "gate_pass": result["comparison"]["pass"],
            "interpretation": "No useful current-action ranking increment; action input is wired and alters scores, but conditional rank generalization is weak. Proxy labels do not certify physical candidate quality."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    findings = audit(args.directory)
    with args.output.open("x") as stream:
        stream.write(json.dumps(findings, indent=2))
    print(json.dumps({k: value for k, value in findings.items() if k != "independent_raw_score_metrics"}, indent=2))
