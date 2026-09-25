"""Test randomized-action information for contact-supported object rise."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[4]
TRAIN_SEEDS = (161, 162)
CHECKPOINT_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
MOTION_SHA = "2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(seed: int) -> dict:
    directory = ROOT / f"outputs/CmResidual/agent_multiaxis_h10_s{seed}_n64"
    manifest = json.loads((directory / "run_manifest.json").read_text())
    command = manifest["command"]
    command_seed = int(command[command.index("--seed") + 1])
    if (manifest["run_status"] != "COMPLETED" or command_seed != seed or
            manifest["checkpoint_sha256"] != CHECKPOINT_SHA or
            manifest["motion_manifest_sha256"] != MOTION_SHA or
            manifest["intervention_axes"] != [0, 1, 2] or
            manifest["followup_horizon"] != 10):
        raise ValueError(f"source manifest drift for seed {seed}")
    path = directory / "transitions.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["run_status"] != "COMPLETED" or payload["followup_horizon"] != 10:
        raise ValueError(f"incomplete randomized data seed {seed}")
    records = payload["records"]
    assignment = records["assignment"].numpy().astype(int)
    selected = assignment != 0
    if sorted(np.unique(assignment)) != [-3, -2, -1, 0, 1, 2, 3]:
        raise ValueError("missing treatment cell")
    if not (records["pre_contact"][selected].all() and
            (records["followup_reset"][selected] == 0).all() and
            (records["followup_progress"][selected] ==
             records["progress"][selected] + 10).all()):
        raise ValueError("incomplete selected follow-up")
    def array(name: str) -> np.ndarray:
        return records[name][selected].numpy().astype(np.float32)
    state = np.concatenate([array("q"), array("dof_vel"),
                            array("object_state"), array("base_action")], axis=1)
    delta = (array("executed_action") - array("base_action"))[:, :3]
    expected = np.zeros_like(delta)
    assigned = assignment[selected]
    expected[np.arange(len(delta)), np.abs(assigned) - 1] = np.sign(assigned) * .1
    if not np.allclose(delta, expected, atol=1e-4):
        raise ValueError("executed randomized perturbation differs")
    dz = array("followup_object_state")[:, 2] - array("object_state")[:, 2]
    contact_count = array("followup_contact_count").reshape(-1)
    target = ((dz >= .03) & (contact_count >= 5)).astype(int)
    if not (np.isfinite(state).all() and np.isfinite(delta).all()):
        raise FloatingPointError("nonfinite Cm input")
    return {"state": state, "delta": delta, "label": target,
            "assignment": assigned, "step": array("global_step").reshape(-1).astype(int),
            "source_sha256": sha256(path)}


def combine(items: list[dict]) -> dict:
    return {key: np.concatenate([item[key] for item in items])
            for key in ("state", "delta", "label", "assignment", "step")}


def features(data: dict, mode: str, *, train: bool) -> np.ndarray:
    delta = data["delta"].copy()
    if mode == "blind":
        delta[:] = 0
    elif mode == "shuffled" and train:
        rng = np.random.default_rng(20260925)
        for step in np.unique(data["step"]):
            ids = np.flatnonzero(data["step"] == step)
            delta[ids] = delta[ids][rng.permutation(len(ids))]
    elif mode not in ("aware", "shuffled"):
        raise ValueError(mode)
    return np.concatenate([data["state"], delta], axis=1)


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=150, max_leaf_nodes=15,
        l2_regularization=10, early_stopping=False, random_state=0)


def score_predictions(pred: np.ndarray, label: np.ndarray) -> dict:
    return {"brier": float(brier_score_loss(label, pred)),
            "auroc": float(roc_auc_score(label, pred)),
            "average_precision": float(average_precision_score(label, pred))}


def effect_signs(fitted, test: dict) -> dict:
    state = test["state"]
    report = {}
    for axis, name in enumerate(("x", "y", "z")):
        pos, neg = (test["label"][test["assignment"] == sign * (axis + 1)]
                    for sign in (1, -1))
        observed = float(pos.mean() - neg.mean())
        predictions = []
        for sign in (1, -1):
            delta = np.zeros((len(state), 3), dtype=np.float32)
            delta[:, axis] = sign * .1
            predictions.append(fitted.predict_proba(np.concatenate([state, delta], axis=1))[:, 1])
        predicted = float((predictions[0] - predictions[1]).mean())
        report[name] = {"observed_plus_minus_pp": 100 * observed,
                        "predicted_plus_minus_pp": 100 * predicted,
                        "positive_count": int(pos.sum()), "positive_n": len(pos),
                        "negative_count": int(neg.sum()), "negative_n": len(neg)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-seed", type=int, choices=(163, 237), default=163)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {seed: load(seed) for seed in (*TRAIN_SEEDS, args.test_seed)}
    train = combine([sources[seed] for seed in TRAIN_SEEDS])
    test = sources[args.test_seed]
    arms = {}
    aware_model = None
    for mode in ("aware", "blind", "shuffled"):
        fitted = model().fit(features(train, mode, train=True), train["label"])
        pred = fitted.predict_proba(features(test, mode, train=False))[:, 1]
        arms[mode] = score_predictions(pred, test["label"])
        if mode == "aware":
            aware_model = fitted
    contrasts = effect_signs(aware_model, test)
    matched_signs = sum(abs(item["observed_plus_minus_pp"]) >= 3 and
                        item["observed_plus_minus_pp"] * item["predicted_plus_minus_pp"] > 0
                        for item in contrasts.values())
    gate = (arms["aware"]["brier"] <= .95 * arms["blind"]["brier"] and
            arms["aware"]["brier"] <= .95 * arms["shuffled"]["brier"] and
            arms["aware"]["auroc"] >= arms["blind"]["auroc"] + .02 and
            matched_signs >= 2)
    report = {"experiment_id": "P-20260925-cm-supported-rise-event",
              "test_seed": args.test_seed, "train_seeds": TRAIN_SEEDS,
              "source_sha256": {str(seed): sources[seed]["source_sha256"] for seed in sources},
              "train_rows": len(train["label"]), "test_rows": len(test["label"]),
              "train_positive": int(train["label"].sum()),
              "test_positive": int(test["label"].sum()),
              "arms": arms, "contrasts": contrasts,
              "matched_signs": int(matched_signs), "gate_pass": bool(gate)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": bool(gate), "arms": arms,
                      "matched_signs": int(matched_signs)}, sort_keys=True))


if __name__ == "__main__":
    main()
