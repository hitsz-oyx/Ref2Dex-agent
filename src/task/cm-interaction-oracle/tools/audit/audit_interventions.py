#!/usr/bin/env python3
"""Audit factual intervention doses, grouped support and saved predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from intervention import physical_targets, window_outcomes
SPEC = importlib.util.spec_from_file_location("probe", ROOT / "src/task/cm-interaction-oracle/tools/run/probe_interventions.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)  # hashes/statistics only; no repeated model inference
    p = torch.load(args.collection / "interventions.pt", map_location="cpu", weights_only=False)
    d = torch.load(args.diagnostic / "diagnostic.pt", map_location="cpu", weights_only=False)
    report = json.loads((args.diagnostic / "result.json").read_text())
    manifest = json.loads((args.diagnostic / "manifest.json").read_text())
    assert sha(args.collection / "interventions.pt") == manifest["dataset_sha256"]
    assert torch.equal(physical_targets(p["before"], p["trajectory"][:, 7]), d["physical_target"])
    y, risk = window_outcomes(p["before"], p["trajectory"], p["rest_height"])
    assert torch.equal(y, d["outcome"]) and torch.equal(risk, d["risk"])
    assert p["valid_steps"].all()
    train, test = d["train_ids"].numpy(), d["test_ids"].numpy()
    clusters = d["clusters"]
    assert not set(clusters[train]) & set(clusters[test])
    assert sorted(np.concatenate((train, test)).tolist()) == list(range(len(p["arm"])))
    for cluster in np.unique(clusters[train]):
        assert len(set(d["fold_ids"][train[clusters[train] == cluster]].tolist())) == 1
    # Factual pre-action timing and identity checks, not model-based reconstruction.
    before = p["before"]
    history = p["history"][:, -1]
    assert torch.allclose(history[:, 36:39], before[:, :3], atol=0, rtol=0)
    assert torch.equal(history[:, 43:49], before[:, 7:13])
    assert torch.equal(history[:, 86:139], before[:, 13:66])
    assert torch.equal(history[:, 73:86], p["hand_root"])
    assert torch.equal(p["actions"][:, 0], p["base_action"]+p["delta"][p["arm"]])
    assert torch.equal(p["actions"][:, 4:], p["base_actions"][:, 4:])
    assert torch.equal(p["pd_targets"][:, 4:], p["pd_base_targets"][:, 4:])
    recomputed = {}
    for name in ("H", "direct", "mediated", "GT"):
        prediction = d["predictions"][name]
        metric = probe.ranking(prediction, y.numpy(), test, d["groups"], p["arm"].numpy())
        assert metric == report["outcomes"][name]["ranking"]
        yn = (torch.as_tensor(prediction)-d["y_mean"])/d["y_scale"]
        mask = d["y_mask"]
        mse = ((yn[test]-d["y_norm"][test]).square()*mask[test]).sum()/mask[test].sum()
        assert abs(float(mse)-report["outcomes"][name]["normalized_mse"]) < 1e-5
        recomputed[name] = dict(test_mse=float(mse),
            train_mse=float(((yn[train]-d["y_norm"][train]).square()*mask[train]).sum()/mask[train].sum()),
            train_ranking=probe.ranking(prediction, y.numpy(), train, d["groups"], p["arm"].numpy()))
    cm = {}
    for name in ("H", "Ha", "Ha_permuted"):
        error = (d["cm_predictions"][name]-d["z_norm"]).square()
        assert abs(float(error[test].mean())-report["cm"][name]["mse"]) < 1e-6
        cm[name] = dict(train_mse=float(error[train].mean()), test_mse=float(error[test].mean()))
    arm_outcomes = []
    arm = p["arm"].numpy()
    for index, name in enumerate(p["arm_names"]):
        rows = np.flatnonzero(arm == index)
        arm_outcomes.append(dict(arm=name, count=len(rows),
            outcome_mean=y[rows].mean(0).tolist(), outcome_std=y[rows].std(0, unbiased=False).tolist(),
            effect_mean=d["physical_target"][rows, :12].mean(0).tolist(),
            interaction_mean=d["physical_target"][rows, 12:].mean(0).tolist(),
            current_lift_mean=float((before[rows, 2]-p["rest_height"][rows]).mean()),
            current_phase_mean=float(p["context"][rows, 3].mean()),
            risk_count=int(risk[rows].sum())))
    # Descriptive cluster bootstrap for prediction error improvements, not validation.
    rng = np.random.default_rng(211)
    test_clusters = np.unique(clusters[test])
    bootstrap = {}
    raw_y = y.numpy()
    for name in ("direct", "mediated", "GT"):
        h_error = ((d["predictions"]["H"]-raw_y)/d["y_scale"].numpy())**2
        n_error = ((d["predictions"][name]-raw_y)/d["y_scale"].numpy())**2
        weights = d["y_mask"].numpy()
        sum_pairs = np.array([[(h_error[test[clusters[test] == c]]*weights[test[clusters[test] == c]]).sum(),
                               (n_error[test[clusters[test] == c]]*weights[test[clusters[test] == c]]).sum()]
                              for c in test_clusters])
        draws = sum_pairs[rng.integers(0, len(test_clusters), (2000, len(test_clusters)))].sum(1)
        gain = 1-draws[:, 1]/draws[:, 0]
        bootstrap[name] = dict(relative_error_gain_CI95=np.quantile(gain, [.025, .975]).tolist(),
                               independent_environment_clusters=len(test_clusters))
    result = dict(status="PASS", data_hash=sha(args.collection / "interventions.pt"),
                  implementation_checks="factual timing, dose, grouped split/folds, fixed target/error/rank exact recomputation",
                  train_arm_counts=np.bincount(arm[train], minlength=7).tolist(),
                  test_arm_counts=np.bincount(arm[test], minlength=7).tolist(),
                  train_drop_risk=int(risk[train].sum()), test_drop_risk=int(risk[test].sum()),
                  task_fits=recomputed, consequence_fits=cm, arm_descriptive_outcomes=arm_outcomes,
                  descriptive_cluster_bootstrap=bootstrap,
                  limits="raw arm means are heterogeneous-state randomized summaries; no same-state counterfactual or certified contact")
    (args.diagnostic / "statistical_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("arm_descriptive_outcomes", "task_fits")}, indent=2))


if __name__ == "__main__":
    main()
