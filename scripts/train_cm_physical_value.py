#!/usr/bin/env python3
"""Fixed nested-data physical models and value/Q pretraining for HF08."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.task.CmResidual.physical_value_contract import advance_events
from src.task.CmResidual.physical_value_data import Episodes, sha
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, dynamics_output, dynamics_loss
MANIFEST_PATH = None
MANIFEST = None


def roll_loss(model, features, dataset, indices, device, reward_scale, blind=False):
    first = dataset.batch(indices, device)
    hs, ha, hm = first["history_state"], first["history_action"], first["history_mask"]
    total = 0
    for offset in range(5):
        target, valid = dataset.future(indices, offset, device)
        actual_action = target["action"]
        action = torch.zeros_like(actual_action) if blind else actual_action
        output, contact, reward, terminal = dynamics_output(model, features, hs, ha, hm,
                                                            target["context"], action)
        rows = dynamics_loss(output, contact, reward, terminal, target["next_state"],
                             target["reward"], target["done"], features, reward_scale, reduction="none")
        total = total + (rows * valid).sum() / valid.sum().clamp_min(1) / 5
        output = output.clone()
        output[:, 49:51] = contact.sigmoid()
        events, _, _ = advance_events(hs[:, -1, 51:55], output[:, 38],
                                      (output[:, 49:51] > .5).all(-1), target["context"][:, 6])
        output[:, 51:55] = events
        hs = torch.cat((hs[:, 1:], output[:, None]), 1)
        ha = torch.cat((ha[:, 1:], actual_action[:, None]), 1)
        hm = torch.cat((hm[:, 1:], torch.ones_like(hm[:, :1])), 1)
    return total


@torch.no_grad()
def diagnostics(models, value, features, dataset, hold, device, reward_scale, blind=False):
    totals = dict(rows=0, normalized_physical_mse=0., contact_brier=0., reward_mse=0., value_weighted_mse=0.)
    for indices in hold[:2048].split(128):
        b = dataset.batch(indices, device)
        predictions, probabilities, rewards, predicted_values = [], [], [], []
        actual_history = torch.cat((b["history_state"][:, 1:], b["next_state"][:, None]), 1)
        next_actions = torch.cat((b["history_action"][:, 1:], b["action"][:, None]), 1)
        next_mask = torch.cat((b["history_mask"][:, 1:], torch.ones_like(b["history_mask"][:, :1])), 1)
        actual_value = value(features.history(actual_history, next_actions, next_mask), features.context(b["next_context"])).squeeze(-1)
        for model in models:
            out, contact, r, term = dynamics_output(model, features, b["history_state"], b["history_action"],
                                                   b["history_mask"], b["context"], torch.zeros_like(b["action"]) if blind else b["action"])
            out[:, 49:51] = contact.sigmoid()
            out[:, 51:55], _, _ = advance_events(b["state"][:, 51:55], out[:, 38],
                                                 (out[:, 49:51] > .5).all(-1), b["context"][:, 6])
            ph = torch.cat((b["history_state"][:, 1:], out[:, None]), 1)
            predictions.append(out)
            probabilities.append(contact.sigmoid())
            rewards.append(r)
            predicted_values.append(value(features.history(ph, next_actions, next_mask), features.context(b["next_context"])).squeeze(-1))
        prediction = torch.stack(predictions).mean(0)
        n = len(indices)
        totals["rows"] += n
        totals["normalized_physical_mse"] += float(((prediction[:, :49] - b["next_state"][:, :49]) / features.state_std[:49]).square().mean()) * n
        totals["contact_brier"] += float((torch.stack(probabilities).mean(0) - b["next_state"][:, 49:51]).square().mean()) * n
        totals["reward_mse"] += float((torch.stack(rewards).mean(0) - b["reward"]).square().mean()) * n
        totals["value_weighted_mse"] += float((torch.stack(predicted_values).mean(0) - actual_value).square().mean()) * n
    return {k: v / totals["rows"] if k != "rows" else v for k, v in totals.items()}


def main():
    global MANIFEST_PATH, MANIFEST
    p = argparse.ArgumentParser()
    p.add_argument("--collections", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--wall-seconds", type=int, default=7100)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    MANIFEST_PATH = a.output / "run_manifest.json"
    MANIFEST = dict(run_status="STARTED", command=sys.argv,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    collection_results_sha256={str(p / "results.json"): sha(p / "results.json") for p in a.collections},
                    created_at=datetime.now(timezone.utc).isoformat(), wall_seconds=a.wall_seconds)
    MANIFEST_PATH.write_text(json.dumps(MANIFEST, indent=2) + "\n")
    started = time.monotonic()
    torch.set_num_threads(2)
    dataset = Episodes(a.collections)
    if not len(dataset.fit) or not len(dataset.hold):
        raise ValueError("episode split missing")
    tiers = [128] if a.smoke else [100000, 500000, 1000000]
    report = dict(run_status="RUNNING", rows=len(dataset.data["state"]), fit_rows=len(dataset.fit),
                  holdout_rows=len(dataset.hold), complete_episodes=len(dataset.starts), excluded_rows=dataset.excluded_rows,
                  inputs=dataset.inputs, updates=a.updates, smoke=a.smoke, tiers={})
    sampler = torch.Generator(device="cpu").manual_seed(20260930285)
    dataset.hold = dataset.hold[torch.randperm(len(dataset.hold), generator=sampler)]
    for size in tiers:
        fit = dataset.subset(size)
        state = dataset.data["state"][fit]
        ctx = dataset.data["context"][fit]
        stats = dict(state_mean=state.mean(0), state_std=state.std(0).clamp_min(.01),
                     context_mean=ctx.mean(0), context_std=ctx.std(0).clamp_min(.01))
        del state, ctx
        features = Features(**{k: v.to(a.device) for k, v in stats.items()}, device=a.device).to(a.device)
        reward_scale = dataset.data["reward"][fit].std().to(a.device).clamp_min(1)
        return_scale = dataset.data["return"][fit].std().to(a.device).clamp_min(1)
        cdim = len(stats["context_mean"])
        value = OutcomeNetwork(cdim, 0, 1, 9283).to(a.device)
        direct_q = OutcomeNetwork(cdim, 18, 1, 9284).to(a.device)
        for name, network in (("value", value), ("direct_q", direct_q)):
            optimizer = torch.optim.Adam(network.parameters(), lr=.001)
            best, best_state = float("inf"), None
            for update in range(a.updates):
                if time.monotonic() - started > a.wall_seconds:
                    raise TimeoutError("pretraining wall limit")
                indices = fit[torch.randint(len(fit), (128,), generator=sampler)]
                b = dataset.batch(indices, a.device)
                pred = network(features.history(b["history_state"], b["history_action"], b["history_mask"]),
                               features.context(b["context"]), b["action"] if network.action_dim else None).squeeze(-1)
                loss = ((pred - b["return"]) / return_scale).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()
                if (update + 1) % max(1, a.updates // 4) == 0:
                    with torch.no_grad():
                        h = dataset.batch(dataset.hold[:512], a.device)
                        prediction = network(features.history(h["history_state"], h["history_action"], h["history_mask"]),
                                             features.context(h["context"]), h["action"] if network.action_dim else None).squeeze(-1)
                        metric = float(((prediction - h["return"]) / return_scale).square().mean())
                    if metric < best:
                        best, best_state = metric, copy.deepcopy(network.state_dict())
            network.load_state_dict(best_state)
        models = []
        metrics = []
        blind_model = None
        for member in range(4 if size == tiers[-1] else 3):
            blind = member == 3
            model = OutcomeNetwork(cdim, 18, 53, 9300 + member).to(a.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            best, best_state = float("inf"), None
            for update in range(a.updates):
                if time.monotonic() - started > a.wall_seconds:
                    raise TimeoutError("pretraining wall limit")
                indices = fit[torch.randint(len(fit), (128,), generator=sampler)]
                loss = roll_loss(model, features, dataset, indices, a.device, reward_scale, blind=blind)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite model loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                if (update + 1) % max(1, a.updates // 4) == 0:
                    with torch.no_grad():
                        metric = float(roll_loss(model, features, dataset, dataset.hold[:128], a.device, reward_scale, blind=blind))
                    if metric < best:
                        best, best_state = metric, copy.deepcopy(model.state_dict())
                    print(json.dumps(dict(tier=size, member=member, update=update + 1, holdout_loss=metric)), flush=True)
            model.load_state_dict(best_state)
            if blind:
                blind_model = model
            else:
                models.append(model)
            metrics.append(best)
        checkpoint = a.output / ("tier_%d.pt" % size)
        payload = dict(schema="ref2dex.physical_value_models.v1", stats=stats, context_dim=cdim,
                       dynamics=[{k: v.cpu() for k, v in m.state_dict().items()} for m in models],
                       value={k: v.cpu() for k, v in value.state_dict().items()},
                       direct_q={k: v.cpu() for k, v in direct_q.state_dict().items()},
                       reward_scale=reward_scale.cpu(), return_scale=return_scale.cpu(),
                       gamma=.99, fit_rows=len(fit), tier=size)
        torch.save(payload, checkpoint)
        report["tiers"][str(size)] = dict(checkpoint=str(checkpoint), sha256=sha(checkpoint), fit_rows=len(fit),
                                            member_loss=metrics, diagnostics=diagnostics(models, value, features, dataset,
                                                                                      dataset.hold, a.device, reward_scale))
        if blind_model is not None:
            report["tiers"][str(size)]["action_blind_diagnostics"] = diagnostics(
                [blind_model], value, features, dataset, dataset.hold, a.device, reward_scale, blind=True)
            torch.save(blind_model.state_dict(), a.output / "action_blind.pt")
        (a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report.update(run_status="COMPLETED", elapsed_seconds=time.monotonic() - started,
                  selected_tier=tiers[-1], selected_checkpoint=report["tiers"][str(tiers[-1])]["checkpoint"])
    (a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
        MANIFEST["run_status"] = "COMPLETED"
    except BaseException as error:
        if MANIFEST is not None:
            MANIFEST.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        if MANIFEST_PATH is not None:
            MANIFEST["completed_at"] = datetime.now(timezone.utc).isoformat()
            MANIFEST_PATH.write_text(json.dumps(MANIFEST, indent=2) + "\n")
