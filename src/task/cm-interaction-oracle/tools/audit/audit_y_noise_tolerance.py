#!/usr/bin/env python3
"""Offline ranking/noise audit for the fixed short-Y rolling contract."""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np


def utility(y):
    y = np.asarray(y, dtype=np.float32)
    return y[..., 7] + np.float32(0.25) * y[..., 3] - y[..., 6]


def audit(source, output, sigmas, seeds):
    output = Path(output)
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    plans = sorted(glob.glob(str(Path(source) / "*-plan.json")))
    panels = []
    source_hashes = {}
    for path in plans:
        d = json.load(open(path))
        if d.get("baseline_upper_bound_certificate"):
            continue
        y = np.asarray(d["y"], dtype=np.float32)
        if y.ndim != 3 or y.shape[1:] != (7, 8) or not np.isfinite(y).all():
            raise ValueError(f"invalid full Y panel: {path}")
        panels.append(y)
        source_hashes[Path(path).name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if not panels:
        raise ValueError("no fully observed panels")
    y = np.concatenate(panels, axis=0)
    truth = utility(y)
    rows = []
    for sigma in sigmas:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            pred_y = y + rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
            pred = utility(pred_y)
            pair_correct = []
            regrets = []
            tied = strict = 0
            for gt, pr in zip(truth, pred):
                chosen = int(np.argmax(pr))
                regrets.append(float(np.max(gt) - gt[chosen]))
                for i in range(len(gt)):
                    for j in range(i + 1, len(gt)):
                        delta = float(gt[i] - gt[j])
                        if delta == 0.0:
                            tied += 1
                            continue
                        strict += 1
                        proposed = float(pr[i] - pr[j])
                        pair_correct.append(0.5 if proposed == 0.0 else float(delta * proposed > 0))
            rows.append(dict(sigma=float(sigma), seed=int(seed), panels=len(y),
                             pairwise_accuracy=float(np.mean(pair_correct)),
                             strict_pairs=int(strict), gt_tied_pairs=int(tied),
                             median_top1_regret=float(np.median(regrets)),
                             mean_top1_regret=float(np.mean(regrets))))
    summary = []
    for sigma in sigmas:
        r = [x for x in rows if x["sigma"] == float(sigma)]
        summary.append(dict(sigma=float(sigma), panels=len(y),
                            median_pairwise_accuracy=float(np.median([x["pairwise_accuracy"] for x in r])),
                            min_pairwise_accuracy=float(np.min([x["pairwise_accuracy"] for x in r])),
                            median_top1_regret=float(np.median([x["median_top1_regret"] for x in r])),
                            max_top1_regret=float(np.max([x["median_top1_regret"] for x in r]))))
    result = dict(schema="ref2dex.y_noise_tolerance.v1", source=str(Path(source).resolve()),
                  plans_used=len(source_hashes), panels=int(len(y)), sigmas=list(map(float, sigmas)),
                  seeds=list(map(int, seeds)), summary=summary, runs=rows,
                  source_plan_sha256=source_hashes,
                  status="PROMISING" if len(y) >= 100 and any(
                      x["sigma"] > 0 and x["median_pairwise_accuracy"] >= .70 and
                      x["median_top1_regret"] <= .125 for x in summary) else "UNPROMISING")
    json.dump(result, open(output / "result.json", "w"), indent=2)
    json.dump(dict(command="audit_y_noise_tolerance", source=str(Path(source).resolve()),
                   output=str(output.resolve()), plans_used=len(source_hashes)),
              open(output / "manifest.json", "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = audit(args.source, args.output, [0.0, .02, .05, .10, .20], [100, 101, 102, 103, 104])
    print(json.dumps({k: result[k] for k in ("status", "plans_used", "panels", "summary")}, indent=2))


if __name__ == "__main__":
    main()
