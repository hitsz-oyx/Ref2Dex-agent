"""Audit a baseline-preserving abstaining selector on observed-tau panels.

The selector is deliberately evaluated only on the saved offline panel
predictions.  It does not infer tau, call PointWorld, or execute a plan.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys

sys.path.insert(0, str(TASK / "src"))

from consequence_evaluator.contracts import is_within
from consequence_evaluator.old_utility import panel_metrics


EPSILONS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_with_abstention(scores, epsilon, baseline=0):
    scores = np.asarray(scores, dtype="float32")
    if scores.ndim != 2 or scores.shape[1] != 7 or not np.isfinite(scores).all():
        raise ValueError("seven-candidate finite score panel required")
    best = scores.argmax(axis=1)
    margin = scores.max(axis=1) - scores[:, int(baseline)]
    return np.where(margin <= float(epsilon), int(baseline), best), margin


def selector_metrics(labels, scores, choices):
    labels = np.asarray(labels)
    choices = np.asarray(choices)
    if labels.shape != scores.shape or labels.ndim != 2 or labels.shape[1] != 7:
        raise ValueError("labels and scores must be complete seven-candidate panels")
    if choices.shape != (len(labels),):
        raise ValueError("selector choice shape mismatch")
    regret = labels.max(axis=1) - labels[np.arange(len(labels)), choices]
    informative = np.any(
        np.abs(labels[:, :, None] - labels[:, None, :]) > .02, axis=(1, 2))
    return dict(
        nonbaseline=int(np.sum(choices != 0)),
        baseline_count=int(np.sum(choices == 0)),
        informative_nonbaseline=int(np.sum((choices != 0) & informative)),
        informative_anchors=int(np.sum(informative)),
        mean_regret=float(np.mean(regret)),
        median_regret=float(np.median(regret)),
        informative_mean_regret=(float(np.mean(regret[informative]))
                                 if informative.any() else None),
        top1_agreement=float(np.mean(choices == labels.argmax(axis=1))),
        top1_in_teacher_tie_set=float(np.mean(
            labels[np.arange(len(labels)), choices] == labels.max(axis=1))),
        choices=choices.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True,
                        help="completed history-panel trajectory fit output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / "outputs/consequence-evaluator"):
        raise ValueError("fresh task-owned selector audit output required")
    fit_root = args.fit.resolve()
    manifest_path = fit_root / "manifest.json"
    predictions_path = fit_root / "panel-predictions.npz"
    result_path = fit_root / "result.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETED" or manifest.get("schema") != (
            "ref2dex.history-candidate-utility-fit.v1"):
        raise ValueError("completed history-panel fit required")
    with np.load(predictions_path, allow_pickle=False) as source:
        required = {"target", "T", "T_tau_shuffle"}
        if not required.issubset(source.files):
            raise ValueError("saved tau predictions missing")
        target = np.asarray(source["target"], dtype="float32")
        scores = np.asarray(source["T"], dtype="float32")
        shuffled = np.asarray(source["T_tau_shuffle"], dtype="float32")
    if target.shape != (18, 7) or scores.shape != target.shape or shuffled.shape != target.shape:
        raise ValueError("expected the frozen 18x7 held tau panel")
    if not np.isfinite(target).all() or not np.isfinite(scores).all() or not np.isfinite(shuffled).all():
        raise ValueError("nonfinite saved panel prediction")
    choices = {}
    metrics = {}
    margins = {}
    for epsilon in EPSILONS:
        selected, margin = select_with_abstention(scores, epsilon)
        key = f"eps_{epsilon:g}"
        choices[key] = selected
        margins[key] = margin
        metrics[key] = selector_metrics(target, scores, selected)
    nominal = panel_metrics(target, scores)
    shuffled_metrics = panel_metrics(target, shuffled)
    screen_gate = bool(
        nominal["pairwise_accuracy"] >= .70
        and nominal["pairwise_accuracy"] - shuffled_metrics["pairwise_accuracy"] >= .03
        and metrics["eps_0.01"]["informative_mean_regret"] <= .01
        and metrics["eps_0.03"]["informative_mean_regret"] <= .01
    )
    result = dict(
        status="PROMISING" if screen_gate else "UNCLEAR",
        screen_gate=screen_gate,
        baseline_candidate=0,
        epsilons=list(EPSILONS),
        nominal_tau_metrics=nominal,
        tau_shuffle_metrics=shuffled_metrics,
        selector_metrics=metrics,
        margins={key: value.tolist() for key, value in margins.items()},
        selector_stability=dict(
            eps_001_vs_eps_003=bool(np.array_equal(choices["eps_0.01"], choices["eps_0.03"])),
            eps_003_informative_nonbaseline=metrics["eps_0.03"]["informative_nonbaseline"],
            eps_003_informative_regret=metrics["eps_0.03"]["informative_mean_regret"],
        ),
        decision="prioritize exact-H candidate-bank construction before selector integration",
        claim="observed-tau offline selector safety only; no H-to-tau, PW, native, or control claim",
        provenance=dict(
            fit_manifest=str(manifest_path), fit_manifest_sha256=sha(manifest_path),
            predictions=str(predictions_path), predictions_sha256=sha(predictions_path),
            fit_result=str(result_path), fit_result_sha256=sha(result_path),
            script=str(Path(__file__).resolve()), script_sha256=sha(Path(__file__).resolve()),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        ),
    )
    output.mkdir(parents=True)
    np.savez_compressed(output / "selector.npz", target=target, scores=scores,
                        shuffled_scores=shuffled,
                        **{key: value for key, value in choices.items()})
    write(output / "result.json", result)
    write(output / "manifest.json", dict(schema="ref2dex.tau-selector-abstention-audit.v1",
                                          status="COMPLETED", run_id=output.name,
                                          **result["provenance"]))
    print(json.dumps({key: result[key] for key in (
        "status", "nominal_tau_metrics", "tau_shuffle_metrics", "selector_metrics",
        "selector_stability", "decision")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
