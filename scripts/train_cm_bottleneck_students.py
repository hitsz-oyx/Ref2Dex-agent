#!/usr/bin/env python3
"""Fixed matched BC fit; holdout never selects an update or configuration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.task.CmResidual.cm_bottleneck_student import BottleneckStudent, PhysicalFeatures


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    count = payload["observation"].shape[0]
    if count % 64:
        raise ValueError("trajectory rows must be complete 64-environment steps")
    done = payload["done"].reshape(-1, 64).bool()
    if not done.any(dim=0).all():
        raise ValueError("missing first completed episode")
    previous = done.long().cumsum(dim=0) - done.long()
    mask = (previous == 0).reshape(-1)
    obs = payload["observation"][mask].float()
    action = payload["source_action"][mask].float()
    if obs.shape[1] != 1442 or action.shape[1] != 18:
        raise ValueError("observation/action dimension mismatch")
    if not torch.isfinite(obs).all() or not torch.isfinite(action).all():
        raise FloatingPointError("nonfinite teacher data")
    return obs, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    fit_obs, fit_action = data(args.fit)
    hold_obs, hold_action = data(args.holdout)
    physical = torch.load(args.physical, map_location="cpu", weights_only=False)
    result = dict(run_status="STARTED", seed=278, steps=2000, batch_size=512,
                  fit_rows=len(fit_obs), holdout_rows=len(hold_obs),
                  fit_sha256=sha(args.fit), holdout_sha256=sha(args.holdout),
                  physical_sha256=sha(args.physical), arms={})
    obs_mean = fit_obs.mean(dim=0)
    obs_std = fit_obs.std(dim=0).clamp_min(1e-3)
    for arm in ("on", "off", "random", "action"):
        started = time.monotonic()
        feature_model = PhysicalFeatures(physical, arm)
        with torch.no_grad():
            extra = feature_model(fit_obs, fit_action)
            hold_extra = feature_model(hold_obs, hold_action)
        mean = torch.cat([obs_mean, extra.mean(dim=0)])
        std = torch.cat([obs_std, extra.std(dim=0).clamp_min(1e-3)])
        x = ((torch.cat([fit_obs, extra], dim=1) - mean) / std).clamp(-10, 10).to(device)
        y = fit_action.to(device)
        torch.manual_seed(278)
        torch.cuda.manual_seed_all(278)
        model = BottleneckStudent().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(278)
        initial_sha = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        for step in range(2000):
            ids = torch.randint(len(x), (512,), generator=generator).to(device)
            prediction = model(x[ids])
            loss = (prediction - y[ids]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite BC loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            hx = ((torch.cat([hold_obs, hold_extra], dim=1) - mean) / std).clamp(-10, 10)
            prediction = torch.cat([model(batch.to(device)).cpu() for batch in hx.split(1024)])
            mse = float((prediction - hold_action).square().mean())
        path = args.output / f"{arm}.pt"
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                        arm=arm, mean=mean, std=std, physical_checkpoint=physical,
                        fit_sha256=result["fit_sha256"], seed=278), path)
        result["arms"][arm] = dict(checkpoint=str(path.resolve()), sha256=sha(path),
                                   initial_state_sha256=initial_sha,
                                   holdout_action_mse=mse, last_loss=float(loss),
                                   wall_seconds=time.monotonic() - started)
        print(json.dumps(dict(arm=arm, **result["arms"][arm])), flush=True)
        del model, optimizer, x, y
    if len({value["initial_state_sha256"] for value in result["arms"].values()}) != 1:
        raise ValueError("matched student initializations differ")
    result["run_status"] = "COMPLETED"
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
