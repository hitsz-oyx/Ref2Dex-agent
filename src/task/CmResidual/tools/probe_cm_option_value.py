"""Compare action-aware and action-blind option outcome predictors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "outputs/CmResidual/agent_cm_option_value_dataset_20260925"
EXPERTS = ("source_e260", "mixed12_e300", "train5_e320", "balanced_e360", "duck_e340")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(seed: int) -> dict:
    values = {}
    aligned = None
    for expert in EXPERTS:
        directory = DATA / f"s{seed}" / expert
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if manifest["run_status"] != "COMPLETED" or manifest["summary"]["num_episodes"] != 64:
            raise ValueError(f"incomplete seed{seed} {expert}")
        path = directory / "initial_features.pt"
        features = torch.load(path, map_location="cpu", weights_only=False)
        keys = ("observation", "q", "object_state", "action", "motion_id",
                "start_frame", "lift_success")
        if any(key not in features for key in keys):
            raise ValueError(f"missing features seed{seed} {expert}")
        current = (features["motion_id"].reshape(-1).numpy().astype(int),
                   features["start_frame"].reshape(-1).numpy().astype(int))
        if aligned is None:
            aligned = current
        elif not all(np.array_equal(a, b) for a, b in zip(aligned, current)):
            raise ValueError(f"state alignment differs seed{seed} {expert}")
        arrays = {key: features[key].numpy() for key in keys}
        if (arrays["observation"].shape != (64, 1442) or arrays["q"].shape != (64, 18) or
                arrays["object_state"].shape != (64, 13) or arrays["action"].shape != (64, 18)):
            raise ValueError(f"feature shape differs seed{seed} {expert}")
        if any(not np.isfinite(arr).all() for arr in arrays.values()):
            raise ValueError(f"nonfinite feature seed{seed} {expert}")
        arrays["sha256"] = digest(path)
        values[expert] = arrays
    parent = json.loads((DATA / "run_manifest.json").read_text())
    if parent["run_status"] != "COMPLETED":
        raise ValueError("collection manifest incomplete")
    return values


def rows(values: dict) -> dict:
    first = values[EXPERTS[0]]
    observation = np.concatenate([values[name]["observation"] for name in EXPERTS])
    q = np.concatenate([values[name]["q"] for name in EXPERTS])
    obj = np.concatenate([values[name]["object_state"] for name in EXPERTS])
    action = np.concatenate([values[name]["action"] for name in EXPERTS])
    labels = np.concatenate([values[name]["lift_success"] for name in EXPERTS]).astype(int)
    identity = np.repeat(np.eye(len(EXPERTS), dtype=np.float32), 64, axis=0)
    return {"observation": observation, "q": q, "object": obj,
            "action": action, "identity": identity, "labels": labels,
            "motion_id": first["motion_id"], "start_frame": first["start_frame"]}


def option_scores(train: dict, test: dict, mode: str, observation_train: np.ndarray,
                  observation_test: np.ndarray) -> np.ndarray:
    features_train = [observation_train, train["q"], train["object"], train["identity"]]
    features_test = [observation_test, test["q"], test["object"], test["identity"]]
    if mode != "blind":
        action_train = train["action"].copy()
        action_test = test["action"].copy()
        if mode == "shuffled":
            for action, seed in ((action_train, 20260925223), (action_test, 20260925224)):
                rng = np.random.RandomState(seed)
                for index in range(len(EXPERTS)):
                    section = slice(index * 64, (index + 1) * 64)
                    action[section] = action[section][rng.permutation(64)]
        features_train.append(action_train)
        features_test.append(action_test)
    x_train = np.concatenate(features_train, axis=1)
    x_test = np.concatenate(features_test, axis=1)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=1, max_iter=1000, random_state=0)
    model.fit(scaler.transform(x_train), train["labels"])
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_raw, test_raw = load(223), load(224)
    train, test = rows(train_raw), rows(test_raw)
    scaler = StandardScaler().fit(train["observation"])
    pca = PCA(n_components=32, svd_solver="randomized", random_state=0)
    observation_train = pca.fit_transform(scaler.transform(train["observation"]))
    observation_test = pca.transform(scaler.transform(test["observation"]))
    models = {}
    for mode in ("aware", "blind", "shuffled"):
        scores = option_scores(train, test, mode, observation_train, observation_test)
        matrix = scores.reshape(len(EXPERTS), 64)
        choice = matrix.argmax(axis=0)
        outcomes = test["labels"].reshape(len(EXPERTS), 64)
        selected = outcomes[choice, np.arange(64)]
        models[mode] = {"brier": float(brier_score_loss(test["labels"], scores)),
                        "route_held_lift": int(selected.sum()),
                        "choice_counts": {name: int((choice == i).sum())
                                          for i, name in enumerate(EXPERTS)},
                        "choice_expert_index": choice.tolist()}
    fixed_path = DATA / "s224/fixed_a/results.json"
    fixed = json.loads(fixed_path.read_text())
    if fixed["summary"]["num_episodes"] != 64:
        raise ValueError("incomplete fixed route")
    fixed_rows = {int(row["env_id"]): row for row in fixed["per_episode"]}
    if len(fixed_rows) != 64 or any(
            fixed_rows[i]["motion_id"] != int(test["motion_id"][i]) or
            fixed_rows[i]["start_frame"] != int(test["start_frame"][i])
            for i in range(64)):
        raise ValueError("fixed-route states differ from option states")
    fixed_lift = sum(int(row["lift_success"]) for row in fixed["per_episode"])
    aware, blind, shuffled = (models[key] for key in ("aware", "blind", "shuffled"))
    gate = (aware["brier"] <= .9 * blind["brier"] and
            aware["brier"] <= .95 * shuffled["brier"] and
            aware["route_held_lift"] >= blind["route_held_lift"] + 5 and
            aware["route_held_lift"] >= fixed_lift + 5)
    report = {"experiment_id": "P-20260925-cm-option-value",
              "train_seed": 223, "test_seed": 224, "experts": EXPERTS,
              "feature_sha256": {str(seed): {name: value[name]["sha256"] for name in EXPERTS}
                                 for seed, value in ((223, train_raw), (224, test_raw))},
              "train_rows": len(train["labels"]), "test_rows": len(test["labels"]),
              "train_successes": int(train["labels"].sum()),
              "test_successes": int(test["labels"].sum()),
              "fixed_route_held_lift": fixed_lift,
              "fixed_route_result_sha256": digest(fixed_path),
              "models": models, "gate_pass": bool(gate)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": gate, "fixed": fixed_lift,
                      "aware": aware["route_held_lift"],
                      "brier": {name: item["brier"] for name, item in models.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
