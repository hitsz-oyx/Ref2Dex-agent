"""Test whether initial policy observations recover a frozen expert route."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[4]
TRAIN_SEEDS = (214, 215)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_seed(seed: int, route: dict, feature_route_hash: str):
    directory = ROOT / f"outputs/CmResidual/agent_observation_route_s{seed}"
    manifest = json.loads((directory / "run_manifest.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    features_path = directory / "initial_features.pt"
    features = torch.load(features_path, map_location="cpu", weights_only=False)
    if manifest["run_status"] != "COMPLETED" or manifest["route_config_sha256"] != feature_route_hash:
        raise ValueError(f"incomplete or drifted seed{seed}")
    if result["summary"]["num_episodes"] != 64 or features["observation"].shape != (64, 1442):
        raise ValueError(f"incomplete feature export seed{seed}")
    log = (ROOT / f"outputs/CmResidual/observation_route_s{seed}.log").read_text()
    lines = [line.split("REF2DEX_OBJECT_ROUTE ", 1)[1]
             for line in log.splitlines() if line.startswith("REF2DEX_OBJECT_ROUTE ")]
    if len(lines) != 1:
        raise ValueError(f"missing motion mapping seed{seed}")
    objects = json.loads(lines[0])["motion_objects"]
    if len(objects) != 12:
        raise ValueError(f"wrong motion count seed{seed}")
    motion_id = features["motion_id"].reshape(-1).numpy().astype(int)
    labels = np.asarray([route[objects[index]] for index in motion_id])
    observation = features["observation"].numpy()
    if not np.isfinite(observation).all():
        raise ValueError(f"nonfinite observation seed{seed}")
    return observation, labels, np.asarray([objects[index] for index in motion_id]), digest(features_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, default=216)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--route-config", type=Path,
                        default=ROOT / "src/task/CmResidual/configs/multitrajectory_object_router_probe.json")
    parser.add_argument("--train-feature-config", type=Path)
    parser.add_argument("--strong-objects", nargs="+",
                        default=("airplane", "duck", "mug", "toothpaste"))
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.model_output is not None and args.model_output.exists():
        raise FileExistsError(args.model_output)
    config_path = args.route_config.resolve()
    config = json.loads(config_path.read_text())
    if args.test_seed in TRAIN_SEEDS:
        raise ValueError("test seed must be held out")
    seeds = (*TRAIN_SEEDS, args.test_seed)
    train_hash = digest(args.train_feature_config.resolve()) if args.train_feature_config else digest(config_path)
    data = {seed: load_seed(seed, config["object_route"],
                            train_hash if seed in TRAIN_SEEDS else digest(config_path))
            for seed in seeds}
    x_train = np.concatenate([data[seed][0] for seed in TRAIN_SEEDS])
    y_train = np.concatenate([data[seed][1] for seed in TRAIN_SEEDS])
    x_test, y_test, object_test, _ = data[args.test_seed]
    class_weight = None if args.class_weight == "none" else args.class_weight
    model = make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale",
                                                class_weight=class_weight))
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    names = sorted(config["experts"])
    per_object = {}
    for obj in sorted(set(object_test)):
        ids = object_test == obj
        per_object[obj] = {
            "correct": int((predicted[ids] == y_test[ids]).sum()),
            "count": int(ids.sum()),
            "predicted_experts": {name: int((predicted[ids] == name).sum()) for name in names},
        }
    strong = args.strong_objects
    report = {
        "experiment_id": args.experiment_id or ("P-20260925-balanced-observation-router"
                          if class_weight else "P-20260925-observation-route-identifiability"),
        "train_seeds": list(TRAIN_SEEDS), "test_seed": args.test_seed,
        "route_config_sha256": digest(config_path),
        "train_feature_config_sha256": train_hash,
        "feature_sha256": {str(seed): data[seed][3] for seed in seeds},
        "classifier": ("StandardScaler + SVC(C=1, kernel=rbf, gamma=scale, "
                       f"class_weight={args.class_weight})"),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "correct": int((predicted == y_test).sum()), "count": len(y_test),
        "expert_names": names,
        "confusion_true_rows_pred_columns": confusion_matrix(y_test, predicted, labels=names).tolist(),
        "per_object": per_object,
        "gate_pass": bool(accuracy_score(y_test, predicted) >= .90 and
                          all(per_object[obj]["correct"] == per_object[obj]["count"] for obj in strong)),
    }
    if args.model_output is not None:
        if not report["gate_pass"]:
            raise ValueError("do not export a router that failed its heldout gate")
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output)
        report["model_path"] = str(args.model_output.resolve())
        report["model_sha256"] = digest(args.model_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"correct": report["correct"], "count": report["count"],
                      "gate_pass": report["gate_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
