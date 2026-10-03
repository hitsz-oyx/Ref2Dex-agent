#!/usr/bin/env python3
"""Held-only Probe for Cm prediction-error memory at the next decision.

Cm remains a frozen one-step physical consequence model.  The new interface is a
causal timing contract: after a real transition, the next decision receives the
observed-minus-predicted physical consequence and ensemble disagreement from the
previous action.  The script fits only a small value residual on the fit split;
it never uses the current transition's future fields as its feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task.CmResidual.physical_value_contract import advance_events  # noqa: E402
from src.task.CmResidual.physical_value_data import Episodes  # noqa: E402
from src.task.CmResidual.physical_value_models import (  # noqa: E402
    Features, OutcomeNetwork, dynamics_output,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("physical value model schema mismatch")
    stats = {key: value.to(device) for key, value in payload["stats"].items()}
    features = Features(**stats, device=device).to(device).eval()
    direct_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    direct_q.load_state_dict(payload["direct_q"], strict=True)
    direct_q.eval()
    dynamics = []
    for index, state in enumerate(payload["dynamics"]):
        model = OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + index).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        dynamics.append(model)
    return payload, features, direct_q, dynamics


class ResidualHead(nn.Module):
    """Predict return residual from direct-Q representation and prior Cm error."""

    def __init__(self, hidden_dim: int, input_dim: int, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + input_dim, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1),
            )

    def forward(self, hidden, extra):
        return self.net(torch.cat((hidden, extra), dim=-1)).squeeze(-1)


@torch.no_grad()
def consequence_rows(dataset: Episodes, indices: torch.Tensor, payload, features,
                     dynamics, device: torch.device, batch_size: int):
    """Compute current direct-Q and prior-transition Cm error features.

    The feature at row t is generated solely from row t-1's observed next_state,
    reward/done fields and frozen Cm forecast.  Row t's own next_state is never
    included in its feature.
    """
    state_std = features.state_std
    reward_scale = payload["reward_scale"].to(device).clamp_min(1)
    rows, q_hidden, q_values, returns, episodes = [], [], [], [], []
    for start in range(0, len(indices), batch_size):
        current_indices = indices[start:start + batch_size]
        batch = dataset.batch(current_indices, device)
        history = features.history(batch["history_state"], batch["history_action"],
                                   batch["history_mask"])
        context = features.context(batch["context"])
        hidden = direct_q_hidden = None
        # direct_q is passed by the caller through a closure attribute to keep the
        # function signature focused on the physical feature calculation.
        direct_q = consequence_rows.direct_q
        hidden = direct_q.encode(history)
        q = direct_q.from_hidden(hidden, context, batch["action"]).squeeze(-1)

        physical = []
        current = batch["state"]
        for model in dynamics:
            predicted, contact, reward, terminal = dynamics_output(
                model, features, batch["history_state"], batch["history_action"],
                batch["history_mask"], batch["context"], batch["action"])
            predicted = predicted.clone()
            predicted[:, 49:51] = contact.sigmoid()
            predicted[:, 51:55], _, _ = advance_events(
                current[:, 51:55], predicted[:, 38],
                predicted[:, 49:51].bool().all(-1), batch["context"][:, 6])
            actual_physical = torch.cat((
                (batch["next_state"][:, 36:39] - current[:, 36:39]) /
                state_std[36:39].clamp_min(.01),
                batch["next_state"][:, 43:49] / state_std[43:49].clamp_min(.01),
                batch["next_state"][:, 49:51], batch["next_state"][:, 51:55],
                batch["reward"][:, None] / reward_scale,
                batch["done"].float()[:, None],
            ), dim=-1)
            predicted_physical = torch.cat((
                (predicted[:, 36:39] - current[:, 36:39]) /
                state_std[36:39].clamp_min(.01),
                predicted[:, 43:49] / state_std[43:49].clamp_min(.01),
                predicted[:, 49:51], predicted[:, 51:55],
                reward[:, None] / reward_scale, terminal.sigmoid()[:, None],
            ), dim=-1)
            physical.append(torch.cat((actual_physical - predicted_physical,
                                       predicted_physical), dim=-1))
        physical = torch.stack(physical)
        mean = physical.mean(0)
        disagreement = physical.std(0, unbiased=False)
        # Keep both the signed model error and its ensemble uncertainty.  The
        # actual fields are included only because this row is the consequence of
        # the previous action in the memory-aligned construction below.
        rows.append(torch.cat((mean[:, :17], disagreement[:, :17]), dim=-1).cpu())
        q_hidden.append(hidden.cpu())
        q_values.append(q.cpu())
        returns.append(batch["return"].cpu())
        episodes.append(batch["episode_id"].cpu())
    return dict(index=indices.clone(), raw_feature=torch.cat(rows), hidden=torch.cat(q_hidden),
                direct=torch.cat(q_values), target=torch.cat(returns), episode=torch.cat(episodes))


def align_rows(dataset: Episodes, selected: torch.Tensor, computed: dict):
    """Select rows and align the previous transition's physical feature exactly."""
    positions = {int(value): index for index, value in enumerate(computed["index"].tolist())}
    selected_pos = torch.tensor([positions[int(value)] for value in selected.tolist()], dtype=torch.long)
    hidden = computed["hidden"][selected_pos]
    direct = computed["direct"][selected_pos]
    target = computed["target"][selected_pos]
    episode = computed["episode"][selected_pos]
    feature = torch.zeros(len(selected), computed["raw_feature"].shape[-1] + 1)
    mask = torch.zeros(len(selected))
    for row, value in enumerate(selected.tolist()):
        previous = value - 1
        if previous >= 0 and dataset.data["episode_id"][previous] == dataset.data["episode_id"][value]:
            if previous not in positions:
                raise RuntimeError("predecessor row was not included in consequence feature batch")
            feature[row, :-1] = computed["raw_feature"][positions[previous]]
            mask[row] = 1.0
    feature[:, -1] = mask
    return dict(hidden=hidden, direct=direct, target=target, episode=episode,
                feature=feature, mask=mask)


def metrics(prediction: torch.Tensor, target: torch.Tensor, episode: torch.Tensor):
    x, y = prediction.numpy(), target.numpy()
    pred_episode, target_episode = [], []
    for identifier in torch.unique(episode):
        selected = episode == identifier
        pred_episode.append(float(prediction[selected].mean()))
        target_episode.append(float(target[selected].mean()))
    return {
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "mae": float(np.mean(np.abs(x - y))),
        "row_spearman": float(spearmanr(x, y).statistic),
        "episode_spearman": float(spearmanr(pred_episode, target_episode).statistic),
    }


@torch.no_grad()
def predict(head, rows, indices, device, shuffled=False, permutation=None,
            batch_size=1024):
    values = []
    feature = rows["feature"]
    if shuffled:
        if permutation is None:
            raise ValueError("shuffle requires a permutation")
        feature = feature[permutation]
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        values.append(head(rows["hidden"][selected].to(device),
                           feature[selected].to(device)).cpu())
    return torch.cat(values)


def fit_head(rows, fit_indices, target, device, updates, batch_size, seed):
    head = ResidualHead(rows["hidden"].shape[-1], rows["feature"].shape[-1], seed).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    scale = target[fit_indices].std().clamp_min(1)
    head.train()
    for _ in range(updates):
        positions = torch.randint(len(fit_indices), (batch_size,), generator=generator)
        selected = fit_indices[positions]
        prediction = head(rows["hidden"][selected].to(device),
                          rows["feature"][selected].to(device))
        loss = ((prediction - target[selected].to(device)) / scale.to(device)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite consequence-memory loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5)
        optimizer.step()
    head.eval()
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-fit-rows", type=int, default=120000)
    parser.add_argument("--max-held-rows", type=int, default=0)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--uncertainty-gate", type=float, default=.5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = Episodes(args.collections)
    payload, features, direct_q, dynamics = load_models(args.checkpoint.resolve(), device)
    consequence_rows.direct_q = direct_q
    fit_source = dataset.fit
    if args.max_fit_rows and len(fit_source) > args.max_fit_rows:
        generator = torch.Generator(device="cpu").manual_seed(2026100308)
        fit_source = fit_source[torch.randperm(len(fit_source), generator=generator)[:args.max_fit_rows]]
    # Keep each split in its original episode/time order for exact memory shift.
    fit_source = fit_source.sort().values
    hold_source = dataset.hold.sort().values
    if args.max_held_rows and len(hold_source) > args.max_held_rows:
        generator = torch.Generator(device="cpu").manual_seed(2026100313)
        hold_source = hold_source[torch.randperm(len(hold_source), generator=generator)[:args.max_held_rows]].sort().values
    def with_predecessors(selected):
        previous = selected - 1
        valid = (previous >= 0)
        safe = previous.clamp_min(0)
        valid &= dataset.data["episode_id"][safe] == dataset.data["episode_id"][selected]
        return torch.unique(torch.cat((selected, safe[valid]))).sort().values
    fit_computed = consequence_rows(dataset, with_predecessors(fit_source), payload,
                                    features, dynamics, device, args.batch_size)
    hold_computed = consequence_rows(dataset, with_predecessors(hold_source), payload,
                                     features, dynamics, device, args.batch_size)
    fit_rows = align_rows(dataset, fit_source, fit_computed)
    hold_rows = align_rows(dataset, hold_source, hold_computed)
    fit_indices = torch.arange(len(fit_source), dtype=torch.long)
    hold_indices = torch.arange(len(hold_source), dtype=torch.long)
    baseline_fit = metrics(fit_rows["direct"], fit_rows["target"], fit_rows["episode"])
    baseline_hold = metrics(hold_rows["direct"], hold_rows["target"], hold_rows["episode"])
    real = fit_head(fit_rows, fit_indices, fit_rows["target"] - fit_rows["direct"],
                    device, args.updates, args.batch_size, 2026100309)
    # The head predicts a residual, not an absolute task-success label.
    real_pred = hold_rows["direct"] + predict(real, hold_rows, hold_indices, device)
    real_metrics = metrics(real_pred, hold_rows["target"], hold_rows["episode"])
    permutation = torch.randperm(len(fit_rows["feature"]), generator=torch.Generator(
        device="cpu").manual_seed(2026100310))
    shuffled = fit_head(
        dict(hidden=fit_rows["hidden"], feature=fit_rows["feature"][permutation]),
        fit_indices, fit_rows["target"] - fit_rows["direct"], device,
        args.updates, args.batch_size, 2026100311)
    shuffled_perm = torch.randperm(len(hold_rows["feature"]), generator=torch.Generator(
        device="cpu").manual_seed(2026100312))
    shuffled_pred = hold_rows["direct"] + predict(
        shuffled, hold_rows, hold_indices, device, shuffled=True, permutation=shuffled_perm)
    shuffled_metrics = metrics(shuffled_pred, hold_rows["target"], hold_rows["episode"])
    real_rmse_delta = real_metrics["rmse"] - baseline_hold["rmse"]
    shuffled_rmse_delta = shuffled_metrics["rmse"] - baseline_hold["rmse"]
    valid_fraction = float(hold_rows["mask"].mean())
    gates = {
        "real_rmse_improvement_at_least_0_5": real_rmse_delta <= -.5,
        "episode_spearman_loss_within_0_01": real_metrics["episode_spearman"] >= baseline_hold["episode_spearman"] - .01,
        "valid_memory_fraction_at_least_0_80": valid_fraction >= .80,
        "real_beats_shuffled_by_0_2_rmse": real_rmse_delta <= shuffled_rmse_delta - .2,
    }
    result = {
        "schema": "ref2dex.cm_consequence_memory_probe.v1",
        "run_status": "COMPLETED",
        "experiment_id": "P-20261003-cm-consequence-memory",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha(args.checkpoint.resolve()),
        "rows": {"fit_used": len(fit_rows["target"]), "held": len(hold_rows["target"])},
        "episodes": {"fit": int(torch.unique(fit_rows["episode"]).numel()),
                     "held": int(torch.unique(hold_rows["episode"]).numel())},
        "contract": {
            "memory": "previous observed-minus-Cm one-step consequence plus ensemble disagreement",
            "current_transition_future_fields_used": False,
            "success_predictor": False,
            "actor_changed": False,
            "uncertainty_gate": args.uncertainty_gate,
        },
        "valid_memory_fraction": valid_fraction,
        "held_models": {
            "frozen_direct_q": baseline_hold,
            "real_consequence_memory_residual": real_metrics,
            "shuffled_consequence_memory_residual": shuffled_metrics,
        },
        "delta_vs_direct": {
            "real": {key: real_metrics[key] - baseline_hold[key] for key in baseline_hold},
            "shuffled": {key: shuffled_metrics[key] - baseline_hold[key] for key in baseline_hold},
        },
        "gates": gates,
        "probe_label": "PROMISING" if all(gates.values()) else "UNPROMISING",
        "decision": "Run one matched policy Probe with previous-consequence memory" if all(gates.values()) else "Close consequence-memory observation contract; no native or policy follow-up",
    }
    torch.save({"schema": "ref2dex.cm_consequence_memory_head.v1",
                "state_dict": real.state_dict(),
                "source_checkpoint_sha256": result["checkpoint_sha256"],
                "feature_dim": int(hold_rows["feature"].shape[-1]),
                "hidden_dim": int(hold_rows["hidden"].shape[-1])},
               args.output.parent / "real_residual_head.pt")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
