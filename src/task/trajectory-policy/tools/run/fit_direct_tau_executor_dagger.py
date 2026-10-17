"""Warm-start the temporal direct executor with learner-state teacher queries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from fit_direct_tau_executor_bc_r3 import as_env, build, encode, evaluate, gpu_state, sha, write

ACTIVE = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15])
RUNTIME = Path("/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl660-root", type=Path, required=True)
    p.add_argument("--unseen-trajectory", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--seconds", type=float, default=300.)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=429)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def select_sequences(root):
    paths = sorted(root.glob("batches/*/sequences/*/trajectory.npz"))
    selected = []
    for path in paths:
        sequence = path.parent.name
        if sequence == "s1_airplane_lift" or "lift" not in sequence:
            continue
        with np.load(path, allow_pickle=False) as packet:
            p = as_env({k: packet[k] for k in packet.files})
        z = p["obj"][:, 0, 2, 3]
        if float(z.max() - z[0]) < .03 or float(z[-1] - z[0]) < -.05:
            continue
        selected.append((sequence, path))
    if len(selected) < 20:
        raise ValueError("fewer than twenty multi-sequence lift teachers passed selection")
    selected = sorted(selected)[:40]
    n_train = int(len(selected) * .70)
    n_val = int(len(selected) * .15)
    return dict(train=selected[:n_train], validation=selected[n_train:n_train + n_val],
                test=selected[n_train + n_val:])


def load_original(root, horizon):
    splits = select_sequences(root)
    datasets = {}
    for split, rows in splits.items():
        chunks = []
        for _, path in rows:
            with np.load(path, allow_pickle=False) as packet:
                chunks.append(build({k: packet[k] for k in packet.files}, [0], horizon))
        datasets[split] = {key: np.concatenate([c[key] for c in chunks], axis=0)
                           for key in chunks[0]}
    return splits, datasets


def query_data(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    required = {"state", "tau", "previous", "label", "pd_target", "q", "env", "tick"}
    if not required.issubset(data):
        raise ValueError("query packet missing fields: " + repr(sorted(required - set(data))))
    if data["state"].ndim != 3 or data["tau"].ndim != 4 or data["label"].ndim != 3:
        raise ValueError("query packet must retain [time, env, ...] axes")
    controls, envs = data["state"].shape[:2]
    flat = {}
    for key in ("state", "tau", "previous", "label", "pd_target", "q", "env", "tick"):
        value = data[key]
        if value.shape[:2] != (controls, envs):
            raise ValueError("query field axis mismatch: " + key)
        flat[key] = value.reshape((controls * envs,) + value.shape[2:])
    return data, flat


def metrics(policy, data, mean, scale, device, torch):
    x = encode(data, mean, scale)
    with torch.inference_mode():
        pred = policy(*(torch.from_numpy(v).to(device) for v in x)).cpu().numpy()
    err = pred - data["label"]
    return dict(active_l1=float(np.abs(err).mean()), active_rmse=float(np.sqrt(np.mean(err ** 2))),
                max_abs=float(np.abs(err).max()))


def main():
    args = parse()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh trajectory-policy output required")
    before = gpu_state(args.gpu)
    if before["utilization"] > 15 or before["used_mib"] > 1024:
        raise RuntimeError("GPU is occupied: " + repr(before))
    if Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise RuntimeError("DAgger fit requires the pinned DExplore runtime")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(ROOT / "tmp/trajectory-policy/direct-tau-dagger-fit"),
                      OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    import torch
    from trajectory_policy.tau_executor import TemporalTauExecutorPolicy
    if torch.__version__ != "2.0.1+cu118" or not torch.cuda.is_available():
        raise RuntimeError("pinned DExplore CUDA runtime required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0")

    payload = torch.load(args.base_checkpoint, map_location="cpu")
    if payload.get("schema") != TemporalTauExecutorPolicy.schema:
        raise ValueError("base checkpoint schema mismatch")
    if int(payload["horizon"]) != args.horizon or int(payload["width"]) != args.width:
        raise ValueError("base checkpoint architecture mismatch")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if mean.shape != (363,) or scale.shape != (363,) or (scale <= 0).any():
        raise ValueError("base standardizer mismatch")

    splits, original = load_original(args.rl660_root, args.horizon)
    query_raw, query = query_data(args.query)
    query_envs = query_raw["state"].shape[1]
    if query_envs != 8:
        raise ValueError("expected eight direct DAgger query environments")
    train_mask = query["env"] < 5
    val_mask = query["env"] >= 5
    query_train = {k: v[train_mask] for k, v in query.items()}
    query_val = {k: v[val_mask] for k, v in query.items()}

    original_train_x = [torch.from_numpy(v).to(device) for v in encode(original["train"], mean, scale)]
    query_train_x = [torch.from_numpy(v).to(device) for v in encode(query_train, mean, scale)]
    original_val_x = [torch.from_numpy(v).to(device) for v in encode(original["validation"], mean, scale)]
    query_val_x = [torch.from_numpy(v).to(device) for v in encode(query_val, mean, scale)]
    original_y = torch.from_numpy(original["train"]["label"]).to(device)
    query_y = torch.from_numpy(query_train["label"]).to(device)
    policy = TemporalTauExecutorPolicy(horizon=args.horizon, width=args.width).to(device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True)
    started = time.monotonic(); best = None
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds:
            break
        n_orig = min(args.batch_size // 2, len(original_y)); n_query = min(args.batch_size - n_orig, len(query_y))
        oi = torch.randint(len(original_y), (n_orig,), generator=generator, device=device)
        qi = torch.randint(len(query_y), (n_query,), generator=generator, device=device)
        pred_orig = policy(*(x[oi] for x in original_train_x))
        pred_query = policy(*(x[qi] for x in query_train_x))
        loss = .5 * torch.nn.functional.smooth_l1_loss(pred_orig, original_y[oi]) + \
               .5 * torch.nn.functional.smooth_l1_loss(pred_query, query_y[qi])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.); optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.inference_mode():
                query_l1 = (policy(*query_val_x) - torch.from_numpy(query_val["label"]).to(device)).abs().mean().item()
                orig_l1 = (policy(*original_val_x) - torch.from_numpy(original["validation"]["label"]).to(device)).abs().mean().item()
            score = query_l1 + .25 * orig_l1
            row = dict(step=step, train_loss=float(loss.item()), query_val_active_l1=float(query_l1),
                       original_val_active_l1=float(orig_l1), selection_score=float(score),
                       elapsed_s=time.monotonic() - started, gpu=gpu_state(args.gpu))
            print(json.dumps(row), flush=True)
            with (args.output / "monitor.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
            if best is None or score < best["selection_score"]:
                best = dict(selection_score=float(score), query_val_active_l1=float(query_l1),
                            original_val_active_l1=float(orig_l1), step=step,
                            state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
    if best is None:
        raise RuntimeError("no DAgger fit checkpoint selected")
    policy.load_state_dict(best["state"])
    with np.load(args.unseen_trajectory, allow_pickle=False) as packet:
        unseen_packet = {k: packet[k] for k in packet.files}
    unseen = build(unseen_packet, list(range(as_env(unseen_packet)["action"].shape[1])), args.horizon)
    checkpoint = dict(schema=TemporalTauExecutorPolicy.schema, horizon=args.horizon, width=args.width,
                      state_dict=policy.state_dict(), mean=mean, scale=scale, active_action=ACTIVE.tolist(),
                      inference_contract=payload["inference_contract"], tau_contract=payload["tau_contract"],
                      dagger_contract="warm-start r3 executor; equal original/query batches; query labels are teacher native action and current-q PD targets",
                      base_checkpoint_sha256=sha(args.base_checkpoint), query_sha256=sha(args.query))
    torch.save(checkpoint, args.output / "best.pt")
    result = dict(status="PROMISING" if best["query_val_active_l1"] < .20 else "UNPROMISING",
        offline_only=True, base_checkpoint=str(args.base_checkpoint.resolve()),
        query_train=metrics(policy, query_train, mean, scale, device, torch),
        query_validation=metrics(policy, query_val, mean, scale, device, torch),
        original_validation=metrics(policy, original["validation"], mean, scale, device, torch),
        original_test=metrics(policy, original["test"], mean, scale, device, torch),
        unseen=metrics(policy, unseen, mean, scale, device, torch), best={k: v for k, v in best.items() if k != "state"},
        query_split=dict(train_envs=list(range(5)), validation_envs=list(range(5, 8)), query_controls=int(query_raw["state"].shape[0])))
    write(args.output / "result.json", result)
    files = [args.base_checkpoint, args.query, args.unseen_trajectory, Path(__file__),
             TASK / "src/trajectory_policy/tau_executor.py", TASK / "src/trajectory_policy/retargeter_data.py"]
    input_hashes = {str(Path(p).resolve()): sha(p) for p in files}
    write(args.output / "manifest.json", dict(schema="ref2dex.direct-tau-dagger-fit.v1", status="COMPLETED",
        task="trajectory-policy", experiment_id="P-20261018-direct-tau-dagger", run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, seed=args.seed, horizon=args.horizon, width=args.width,
        input_sha256=input_hashes, gpu_before=before, gpu_after=gpu_state(args.gpu), elapsed_s=time.monotonic() - started,
        base_checkpoint_sha256=sha(args.base_checkpoint), query_sha256=sha(args.query), checkpoint_sha256=sha(args.output / "best.pt"),
        query_contract=checkpoint["dagger_contract"]))


if __name__ == "__main__":
    main()
