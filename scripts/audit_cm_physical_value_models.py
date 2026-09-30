#!/usr/bin/env python3
"""Development-set diagnostics; never selects a model or changes a policy gate."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.task.CmResidual.physical_value_data import Episodes, sha
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, dynamics_output


@torch.no_grad()
def audit(dataset, payload, device, limit):
    features = Features(**{k: v.to(device) for k, v in payload["stats"].items()}, device=device).to(device)
    networks = {}
    for name, action_dim, seed in (("value", 0, 9283), ("direct_q", 18, 9284)):
        network = OutcomeNetwork(payload["context_dim"], action_dim, 1, seed).to(device)
        network.load_state_dict(payload[name])
        networks[name] = network.eval()
    dynamics = []
    for i, state in enumerate(payload["dynamics"]):
        network = OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + i).to(device)
        network.load_state_dict(state)
        dynamics.append(network.eval())
    generator = torch.Generator().manual_seed(20260930285)
    hold = dataset.hold[torch.randperm(len(dataset.hold), generator=generator)[:limit]]
    fit = dataset.subset(payload["tier"])
    assert len(fit) == payload["fit_rows"]
    fit_mean = dataset.data["return"][fit].mean()
    actuals, estimates, physical = [], {name: [] for name in networks}, []
    baselines = {name: [] for name in ("copy_state", "constant_velocity")}
    for indices in hold.split(128):
        b = dataset.batch(indices, device)
        history = features.history(b["history_state"], b["history_action"], b["history_mask"])
        context = features.context(b["context"])
        actuals.append(b["return"].cpu())
        for name in baselines:
            predicted = b["state"].clone()
            if name == "constant_velocity":
                predicted[:, :18] += predicted[:, 18:36] / 30
                predicted[:, 36:39] += predicted[:, 43:46] / 30
            target = b["next_state"]
            dot = (predicted[:, 39:43] * target[:, 39:43]).sum(-1).abs().clamp(0, 1)
            baselines[name].append(dict(
                joint_position_normalized_mse=((predicted[:, :18]-target[:, :18])/features.state_std[:18]).square().mean(-1).cpu(),
                joint_velocity_normalized_mse=((predicted[:, 18:36]-target[:, 18:36])/features.state_std[18:36]).square().mean(-1).cpu(),
                object_position_squared_meters=(predicted[:, 36:39]-target[:, 36:39]).square().mean(-1).cpu(),
                object_rotation_squared_radians=(2*dot.acos()).square().cpu(),
                object_velocity_normalized_mse=((predicted[:, 43:49]-target[:, 43:49])/features.state_std[43:49]).square().mean(-1).cpu(),
                contact_brier=(predicted[:, 49:51]-target[:, 49:51]).square().mean(-1).cpu()))
        for name, network in networks.items():
            pred = network(history, context, b["action"] if network.action_dim else None).squeeze(-1)
            estimates[name].append(pred.cpu())
        members = []
        for model in dynamics:
            predicted, contact, reward, terminal = dynamics_output(model, features, b["history_state"],
                b["history_action"], b["history_mask"], b["context"], b["action"])
            target = b["next_state"]
            # Component metrics avoid mixing quaternion sign with Euclidean
            # error, or letting 36 joint fields obscure object movement.
            dot = (predicted[:, 39:43] * target[:, 39:43]).sum(-1).abs().clamp(0, 1)
            members.append(dict(
                joint_position_normalized_mse=((predicted[:, :18] - target[:, :18]) / features.state_std[:18]).square().mean(-1),
                joint_velocity_normalized_mse=((predicted[:, 18:36] - target[:, 18:36]) / features.state_std[18:36]).square().mean(-1),
                object_position_squared_meters=(predicted[:, 36:39] - target[:, 36:39]).square().mean(-1),
                object_rotation_squared_radians=(2 * dot.acos()).square(),
                object_velocity_normalized_mse=((predicted[:, 43:49] - target[:, 43:49]) / features.state_std[43:49]).square().mean(-1),
                contact_brier=(contact.sigmoid() - target[:, 49:51]).square().mean(-1),
                reward_squared_error=(reward - b["reward"]).square(),
                termination_brier=(terminal.sigmoid() - b["done"].float()).square()))
        physical.append({key: torch.stack([member[key] for member in members]).mean(0).cpu()
                         for key in members[0]})
    target = torch.cat(actuals)
    baseline_mse = (target - fit_mean).square().mean()
    values = {}
    for name in networks:
        pred = torch.cat(estimates[name])
        mse = (pred - target).square().mean()
        values[name] = dict(return_rmse=float(mse.sqrt()), return_mae=float((pred-target).abs().mean()),
                            normalized_return_mse=float(mse / payload["return_scale"].square()),
                            improvement_over_fit_mean=float(1 - mse / baseline_mse.clamp_min(1e-9)))
    def reduce_components(rows):
        components = {key: float(torch.cat([row[key] for row in rows]).mean()) for key in rows[0]}
        components["object_position_rmse_meters"] = components.pop("object_position_squared_meters") ** .5
        components["object_rotation_rmse_radians"] = components.pop("object_rotation_squared_radians") ** .5
        if "reward_squared_error" in components:
            components["reward_rmse"] = components.pop("reward_squared_error") ** .5
        return components
    components = reduce_components(physical)
    return dict(rows=len(hold), fit_mean_return=float(fit_mean), constant_return_rmse=float(baseline_mse.sqrt()),
                value_fit=values, physical_member_mean=components,
                physical_baselines={name: reduce_components(rows) for name, rows in baselines.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collections", nargs="+", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--rows", type=int, default=2048)
    a = p.parse_args()
    torch.set_num_threads(2)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if payload["schema"] != "ref2dex.physical_value_models.v1":
        raise ValueError("model schema")
    dataset = Episodes(a.collections, gamma=payload["gamma"])
    report = dict(run_status="COMPLETED", checkpoint_sha256=sha(a.checkpoint), command=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        tier=payload["tier"], fit_rows=payload["fit_rows"],
        inputs=dataset.inputs,
        scope="development holdout under noisy source collection policy; diagnostic only, no counterfactual action-value or policy utility claim",
        diagnostics=audit(dataset, payload, a.device, a.rows))
    with a.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report["diagnostics"]), flush=True)


if __name__ == "__main__":
    main()
