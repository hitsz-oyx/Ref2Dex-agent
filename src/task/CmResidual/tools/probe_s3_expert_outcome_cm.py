"""Test whether candidate initial actions improve long-horizon grasp outcome ranking."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
SOURCES = {
    "source": ("agent_v135_s3_cmoff_s70_e180", 180,
               "863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c"),
    "back240": ("agent_v139_s3_backtrack_s70_e260", 240,
                "a88924a590966c964b029e067985fea41465f230d459454b8048899cbe8670c2"),
    "back260": ("agent_v139_s3_backtrack_s70_e260", 260,
                "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"),
}
EXPERTS = tuple(SOURCES)
TRAIN_SEEDS = (301, 302, 303)
TEST_SEEDS = (304, 305)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auc(pred: torch.Tensor, label: torch.Tensor) -> float:
    pos = pred[label]
    neg = pred[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).float() +
                  0.5 * (pos[:, None] == neg[None, :]).float()).mean())


def load_data() -> tuple[dict[int, dict[str, dict]], dict]:
    data = defaultdict(dict)
    provenance = {}
    for expert, (run_name, epoch, expected_sha) in SOURCES.items():
        run = ROOT / "outputs/Dexplore" / run_name
        for seed in TRAIN_SEEDS + TEST_SEEDS:
            eval_dir = run / f"eval_s{seed}_e{epoch}_full"
            manifest = json.loads((eval_dir / "run_manifest.json").read_text())
            if (manifest["run_status"] != "COMPLETED" or
                    manifest["checkpoint_sha256"] != expected_sha or
                    manifest["seed"] != seed or
                    not manifest["early_termination_disabled"]):
                raise ValueError(f"evaluation provenance mismatch: {eval_dir}")
            features = eval_dir / "initial_features.pt"
            x = torch.load(features, map_location="cpu", weights_only=False)
            if x["schema"] != "ref2dex.initial_grasp_candidate.v1":
                raise ValueError(f"wrong feature schema: {features}")
            for name in ("observation", "q", "object_state", "action",
                         "start_frame", "motion_id", "lift_success",
                         "max_contact_lift_m", "contact_fraction"):
                if x[name].shape[0] != 64 or not torch.isfinite(x[name].float()).all():
                    raise ValueError(f"incomplete or nonfinite {name}: {features}")
            data[seed][expert] = x
            provenance[f"{seed}:{expert}"] = {
                "feature_path": str(features), "feature_sha256": sha256(features),
                "checkpoint_sha256": expected_sha}
    alignment = {}
    for seed, candidates in data.items():
        base = candidates[EXPERTS[0]]
        maximum = 0.0
        for expert in EXPERTS[1:]:
            other = candidates[expert]
            for key in ("start_frame", "motion_id"):
                if not torch.equal(base[key], other[key]):
                    raise ValueError(f"candidate {key} not aligned: seed{seed} {expert}")
            for key in ("q", "object_state", "observation"):
                diff = float((base[key] - other[key]).abs().max())
                maximum = max(maximum, diff)
                if diff > 1e-6:
                    raise ValueError(f"candidate initial {key} drift {diff}: seed{seed} {expert}")
        alignment[seed] = maximum
    return data, {"sources": provenance, "max_initial_state_difference": alignment}


def features(data: dict, seeds: tuple[int, ...], action: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, successes, lifts = [], [], []
    for seed in seeds:
        for expert_id, expert in enumerate(EXPERTS):
            item = data[seed][expert]
            onehot = F.one_hot(torch.full((64,), expert_id), len(EXPERTS)).float()
            parts = [item["q"].float(), item["object_state"].float(), onehot]
            if action:
                parts.append(item["action"].float())
            xs.append(torch.cat(parts, dim=-1))
            successes.append(item["lift_success"].float())
            lifts.append(item["max_contact_lift_m"].float().clamp(0, 0.3) / 0.3)
    return torch.cat(xs), torch.cat(successes), torch.cat(lifts)


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit(x: torch.Tensor, success: torch.Tensor, lift: torch.Tensor,
        *, seed: int) -> tuple[Head, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed)
    mean = x.mean(0)
    std = x.std(0).clamp_min(0.05)
    normalized = ((x - mean) / std).clamp(-10, 10)
    model = Head(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(500):
        model.train()
        prediction = model(normalized)
        loss = F.binary_cross_entropy_with_logits(prediction[:, 0], success)
        loss = loss + 0.2 * F.smooth_l1_loss(prediction[:, 1], lift)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite outcome-model loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std, float(loss.detach())


def evaluate(model: Head, mean: torch.Tensor, std: torch.Tensor,
             x: torch.Tensor, label: torch.Tensor, data: dict) -> dict:
    with torch.no_grad():
        score = model(((x - mean) / std).clamp(-10, 10))[:, 0].sigmoid()
    n_seed = len(TEST_SEEDS)
    selection = score.reshape(n_seed, len(EXPERTS), 64).argmax(1)
    chosen_success = []
    by_seed = {}
    for index, seed in enumerate(TEST_SEEDS):
        outcome = torch.stack([data[seed][expert]["lift_success"] for expert in EXPERTS])
        success = outcome.gather(0, selection[index][None]).reshape(-1)
        chosen_success.append(success)
        by_seed[str(seed)] = int(success.sum())
    return {"auc": auc(score, label.bool()), "selected_successes":
            int(torch.cat(chosen_success).sum()), "selected_by_seed": by_seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    data, provenance = load_data()
    x_train_action, y_train, lift_train = features(data, TRAIN_SEEDS, True)
    x_test_action, y_test, _ = features(data, TEST_SEEDS, True)
    x_train_blind, _, _ = features(data, TRAIN_SEEDS, False)
    x_test_blind, _, _ = features(data, TEST_SEEDS, False)
    shuffled = x_train_action.clone()
    generator = torch.Generator().manual_seed(20260924)
    action_start = x_train_blind.shape[1]
    for expert_id in range(len(EXPERTS)):
        # Rows are seed-major, so shuffle only within the same expert ID.
        indices = torch.cat([torch.arange(seed_idx * len(EXPERTS) * 64 +
                                          expert_id * 64,
                                          seed_idx * len(EXPERTS) * 64 +
                                          (expert_id + 1) * 64)
                             for seed_idx in range(len(TRAIN_SEEDS))])
        shuffled[indices, action_start:] = x_train_action[
            indices[torch.randperm(len(indices), generator=generator)], action_start:]
    arms = {}
    output.mkdir(parents=True)
    for name, x_train, x_test in (
            ("action_aware", x_train_action, x_test_action),
            ("action_blind", x_train_blind, x_test_blind),
            ("action_shuffled", shuffled, x_test_action)):
        model, mean, std, loss = fit(x_train, y_train, lift_train, seed=20260924)
        arms[name] = {"train_loss": loss,
                      **evaluate(model, mean, std, x_test, y_test, data)}
        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "experts": EXPERTS, "input_dim": x_train.shape[1]},
                   output / f"{name}.pt")
    expert_success = {expert: {str(seed): int(data[seed][expert]["lift_success"].sum())
                               for seed in TEST_SEEDS} for expert in EXPERTS}
    best_fixed = max(sum(values.values()) for values in expert_success.values())
    report = {"experiment_id": "P-20260924-s3-expert-outcome-cm",
              "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
              "experts": EXPERTS, "provenance": provenance,
              "expert_success": expert_success,
              "best_fixed_expert_successes_per_128": best_fixed,
              "arms": arms,
              "gate": (arms["action_aware"]["auc"] >=
                       max(arms["action_blind"]["auc"],
                           arms["action_shuffled"]["auc"]) + 0.05 and
                       arms["action_aware"]["selected_successes"] >=
                       best_fixed + 7)}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"best_fixed": best_fixed, "arms": arms, "gate": report["gate"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    random.seed(20260924)
    torch.set_num_threads(4)
    main()
