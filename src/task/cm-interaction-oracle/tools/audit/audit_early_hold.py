#!/usr/bin/env python3
"""Recompute factual early-hold labels and saved scores without model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/tools/run"))
from probe_early_hold import dropout_auc, retention_ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)  # CPU: file checks and statistics; no model inference.
    p = torch.load(args.collection / "interventions.pt", map_location="cpu", weights_only=False)
    d = torch.load(args.diagnostic / "diagnostic.pt", map_location="cpu", weights_only=False)
    report = json.loads((args.diagnostic / "result.json").read_text())
    manifest = json.loads((args.diagnostic / "manifest.json").read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(args.collection / "interventions.pt") == manifest["input_sha256"]
    assert all(sha(ROOT / path) == digest for path, digest in manifest["code_sha256"].items())
    # Independent time loop, including contact runs across the step8 boundary.
    pair = p["trajectory"][:, :, 71].numpy() > .5
    lift = p["trajectory"][:, :, 2].numpy()-p["rest_height"].numpy()[:, None]
    failure = lift < .02
    lost = np.zeros(len(pair), dtype=int)
    for step in range(32):
        lost = np.where(pair[:, step], 0, lost+1)
        failure[:, step] |= lost >= 6
    heads = []
    for endpoint in (16, 32):
        region = slice(8, endpoint)
        heads.extend((pair[:, region].mean(1), ((lift[:, region] >= .03) & pair[:, region]).mean(1),
                      failure[:, region].any(1).astype(float)))
    y = np.stack(heads, -1)
    np.testing.assert_allclose(y, d["outcome"].numpy(), atol=3e-8, rtol=0)
    early = failure[:, :8].any(1)
    assert np.array_equal(early, d["details"]["early_failure"].numpy())
    assert p["valid_steps"].all() and (p["pre_hold_steps"] >= 6).all()
    history, before = p["history"][:, -1], p["before"]
    assert torch.equal(history[:, 36:39], before[:, :3])
    assert torch.equal(history[:, 43:49], before[:, 7:13])
    assert torch.equal(history[:, 86:139], before[:, 13:66])
    assert torch.equal(history[:, 73:86], p["hand_root"])
    assert torch.equal(p["actions"][:, 0], p["base_action"]+p["delta"][p["arm"]])
    assert torch.equal(p["actions"][:, 4:], p["base_actions"][:, 4:])
    assert torch.equal(p["pd_targets"][:, 4:], p["pd_base_targets"][:, 4:])
    train, test = d["train_ids"].numpy(), d["test_ids"].numpy()
    clusters, groups, arms = d["clusters"], d["groups"], p["arm"].numpy()
    assert not set(clusters[train]) & set(clusters[test])
    assert sorted(np.concatenate((train, test)).tolist()) == list(range(len(pair)))
    metrics = {}
    for name, pred in d["predictions"].items():
        # Stored float32 targets define the registered tie tolerance exactly.
        target = d["outcome"].numpy()
        assert retention_ranking(pred, target, test, groups, arms) == report["models"][name]["ranking"]
        error = ((pred-target)/d["y_scale"].numpy())**2
        assert abs(error[test].mean()-report["models"][name]["test_normalized_mse"]) < 1e-6
        supported_heads = []
        for channel, tolerance in ((3, .03), (5, .5)):
            rows_out = []
            for group in np.unique(groups[test]):
                rows = test[groups[test] == group]
                i, j = np.triu_indices(len(rows), 1)
                left, right = rows[i], rows[j]
                difference = target[left, channel]-target[right, channel]
                keep = (arms[left] != arms[right]) & (np.abs(difference) > tolerance)
                if keep.sum() >= 10:
                    product = (pred[left[keep], channel]-pred[right[keep], channel])*difference[keep]
                    rows_out.append(dict(stratum=int(group), pairs=int(keep.sum()),
                                         accuracy=float(((product > 0)+.5*(product == 0)).mean())))
            supported_heads.append(rows_out)
        not_early = test[~early[test]]
        metrics[name] = dict(supported_primary_heads=supported_heads,
            supported_primary=float(np.mean([np.mean([r["accuracy"] for r in head]) for head in supported_heads])),
            no_early_failure_normalized_mse=float(error[not_early].mean()),
            no_early_failure_failure32_auc=dropout_auc(pred[not_early, 5], target[not_early, 5]))
    result = dict(status="PASS", input_sha256=manifest["input_sha256"],
        checks="code hashes, independent labels, factual timing/dose, disjoint environments, saved error/rank",
        train_arm_counts=np.bincount(arms[train], minlength=7).tolist(),
        test_arm_counts=np.bincount(arms[test], minlength=7).tolist(),
        train_motion_counts=np.bincount(p["motion_id"].numpy()[train], minlength=3).tolist(),
        test_motion_counts=np.bincount(p["motion_id"].numpy()[test], minlength=3).tolist(),
        early_failures=int(early.sum()), early_only_failures=int((early & ~failure[:, 8:].any(1)).sum()),
        test_no_early_failure_trials=int((~early[test]).sum()),
        test_no_early_failure_late_failures=int((~early[test] & failure[test, 8:].any(1)).sum()),
        descriptive_sensitivity=metrics,
        limits="Posthoc supported-stratum and no-early-failure descriptions do not change registered gates; the latter conditions on post-treatment survival, not a causal cohort. No model retraining.")
    (args.diagnostic / "statistical_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
