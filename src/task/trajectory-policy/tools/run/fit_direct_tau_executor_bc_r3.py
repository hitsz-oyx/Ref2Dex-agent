"""Fit the repaired direct executor on multiple RL660 teacher sequences.

The reference plan is encoded from reference hand/object frames only.  The
student's measured state is a separate feedback input.  No task goal or
oracle-C branch is used in this probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_state(gpu):
    row = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
    util, used, total = [int(x.strip()) for x in row.strip().split(",")]
    processes = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True).strip()
    return dict(utilization=util, used_mib=used, total_mib=total, processes=processes)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl660-root", type=Path, required=True)
    p.add_argument("--unseen-trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--idle-context-pid", type=int, action="append", default=[])
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--seconds", type=float, default=600.)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=427)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def as_env(packet):
    """Normalize an RL660 one-environment packet to [T,E,...]."""
    result = dict(packet)
    for key in ("q", "dq", "hand", "obj", "velocity", "action", "applied"):
        value = result[key]
        if key in ("q", "dq", "action", "applied") and value.ndim == 2:
            value = value[:, None, :]
        elif key == "hand" and value.ndim == 3:
            value = value[:, None, :, :]
        elif key == "obj" and value.ndim == 3:
            value = value[:, None, :, :]
        elif key == "velocity" and value.ndim == 2:
            value = value[:, None, :]
        result[key] = value
    return result


def build(packet, env_ids, horizon):
    from trajectory_policy.retargeter_data import ACTIVE, features, reference_tau
    packet = as_env(packet)
    action, applied = packet["action"], packet["applied"]
    if not np.array_equal(action, applied):
        raise ValueError("teacher labels must equal applied native commands")
    controls, envs = action.shape[:2]
    ticks = np.arange(controls - horizon, dtype=np.int64)
    rows = []
    for env in env_ids:
        future = packet["hand"][ticks[:, None] + 1 + np.arange(horizon)[None, :], env]
        state, _ = features(packet["q"][ticks, env], packet["dq"][ticks, env],
                            packet["hand"][ticks, env], packet["obj"][ticks, env],
                            packet["velocity"][ticks, env], future)
        tau = reference_tau(packet["hand"][ticks, env], packet["obj"][ticks, env], future)
        previous = np.zeros((len(ticks), len(ACTIVE)), dtype=np.float32)
        if len(ticks) > 1:
            previous[1:] = applied[ticks[:-1], env][:, list(ACTIVE)]
        rows.append((state, tau, previous,
                     applied[ticks, env][:, list(ACTIVE)].astype(np.float32),
                     np.full(len(ticks), env, dtype=np.int64), ticks.copy()))
    state, tau, previous, label, env, tick = [np.concatenate(values, axis=0)
        for values in zip(*rows)]
    return dict(state=state, tau=tau, previous=previous, label=label, env=env, tick=tick)


def encode(data, mean, scale):
    state_dim = data["state"].shape[1]
    tau_dim = data["tau"].shape[1] * data["tau"].shape[2]
    value = np.concatenate((data["state"], data["tau"].reshape(len(data["tau"]), -1),
                            data["previous"]), axis=1)
    a = 0
    state = (value[:, a:a + state_dim] - mean[a:a + state_dim]) / scale[a:a + state_dim]; a += state_dim
    tau = (value[:, a:a + tau_dim] - mean[a:a + tau_dim]) / scale[a:a + tau_dim]; a += tau_dim
    previous = (value[:, a:] - mean[a:]) / scale[a:]
    return state.astype(np.float32), tau.reshape(len(tau), -1, 33).astype(np.float32), previous.astype(np.float32)


def evaluate(policy, data, mean, scale, device, torch):
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
    if not args.rl660_root.exists() or not args.unseen_trajectory.exists():
        raise FileNotFoundError("RL660 root and unseen trajectory are required")
    before = gpu_state(args.gpu)
    if before["utilization"] > 15 or before["used_mib"] > 1024:
        raise RuntimeError("GPU is occupied: " + repr(before))
    scratch = ROOT / "tmp/trajectory-policy/direct-tau-bc-r3"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch / "torch-extensions"), OMP_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1")
    import torch
    from trajectory_policy.tau_executor import TemporalTauExecutorPolicy
    if torch.__version__ != "2.0.1+cu118" or not torch.cuda.is_available():
        raise RuntimeError("pinned DExplore CUDA runtime required")
    torch.set_num_threads(2); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); device = torch.device("cuda:0")

    paths = sorted(args.rl660_root.glob("batches/*/sequences/*/trajectory.npz"))
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
        raise ValueError("fewer than twenty multi-sequence lift teachers passed the bounded selection")
    selected = sorted(selected)[:40]
    n_train = int(len(selected) * .70); n_val = int(len(selected) * .15)
    splits = dict(train=selected[:n_train], validation=selected[n_train:n_train + n_val],
                  test=selected[n_train + n_val:])
    datasets = {}
    input_hashes = {str(path.resolve()): sha(path) for _, path in selected}
    for split, rows in splits.items():
        chunks = []
        for sequence, path in rows:
            with np.load(path, allow_pickle=False) as packet:
                chunks.append(build({k: packet[k] for k in packet.files}, [0], args.horizon))
        datasets[split] = {key: np.concatenate([c[key] for c in chunks], axis=0)
                           for key in chunks[0]}
    train = datasets["train"]
    value = np.concatenate((train["state"], train["tau"].reshape(len(train["tau"]), -1), train["previous"]), axis=1)
    mean = value.mean(axis=0).astype(np.float32); scale = value.std(axis=0).astype(np.float32); scale[scale < 1e-4] = 1.
    train_x = [torch.from_numpy(v).to(device) for v in encode(train, mean, scale)]
    val_x = [torch.from_numpy(v).to(device) for v in encode(datasets["validation"], mean, scale)]
    train_y = torch.from_numpy(train["label"]).to(device); val_y = torch.from_numpy(datasets["validation"]["label"]).to(device)
    policy = TemporalTauExecutorPolicy(horizon=args.horizon, width=args.width).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True); started = time.monotonic(); best = None
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds: break
        ids = torch.randint(len(train_y), (min(args.batch_size, len(train_y)),), generator=generator, device=device)
        pred = policy(*(x[ids] for x in train_x)); loss = torch.nn.functional.smooth_l1_loss(pred, train_y[ids])
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.); optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.inference_mode(): val_l1 = (policy(*val_x) - val_y).abs().mean().item()
            row = dict(step=step, train_loss=float(loss.item()), val_active_l1=float(val_l1),
                       elapsed_s=time.monotonic()-started, gpu=gpu_state(args.gpu))
            print(json.dumps(row), flush=True)
            with (args.output / "monitor.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
            if best is None or val_l1 < best["val_active_l1"]:
                best = dict(val_active_l1=float(val_l1), step=step,
                            state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
    policy.load_state_dict(best["state"])
    with np.load(args.unseen_trajectory, allow_pickle=False) as packet:
        unseen_packet = {k: packet[k] for k in packet.files}
    unseen = build(unseen_packet, list(range(as_env(unseen_packet)["action"].shape[1])), args.horizon)
    checkpoint = dict(schema=TemporalTauExecutorPolicy.schema, horizon=args.horizon, width=args.width,
                      state_dict=policy.state_dict(), mean=mean, scale=scale,
                      active_action=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15],
                      inference_contract="live measured state87 + reference-frame tau plan8 + previous applied active action -> direct active native action12",
                      tau_contract="tau_k = R_ref_t (h_ref[t+k] - h_ref[t]); reference hand/object only, independent of student state")
    torch.save(checkpoint, args.output / "best.pt")
    result = dict(status="PROMISING" if best["val_active_l1"] < .20 else "UNPROMISING", offline_only=True,
        validation=evaluate(policy, datasets["validation"], mean, scale, device, torch),
        test=evaluate(policy, datasets["test"], mean, scale, device, torch),
        unseen=evaluate(policy, unseen, mean, scale, device, torch), best=dict(val_active_l1=best["val_active_l1"], step=best["step"]),
        checkpoint_sha256=sha(args.output / "best.pt"), sequences=[s for s, _ in selected], sequence_split={k:[s for s,_ in v] for k,v in splits.items()},
        source_limitation="RL660 sequence-level split; s1_airplane_lift is held out")
    write(args.output / "result.json", result)
    input_hashes[str(args.unseen_trajectory.resolve())] = sha(args.unseen_trajectory)
    input_hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    input_hashes[str((TASK / "src/trajectory_policy/tau_executor.py").resolve())] = sha(TASK / "src/trajectory_policy/tau_executor.py")
    input_hashes[str((TASK / "src/trajectory_policy/retargeter_data.py").resolve())] = sha(TASK / "src/trajectory_policy/retargeter_data.py")
    write(args.output / "manifest.json", dict(schema="ref2dex.direct-tau-bc-r3.v1", status="COMPLETED",
        task="trajectory-policy", experiment_id="P-20261018-direct-tau-executor-r3", run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), physical_gpu=args.gpu,
        seed=args.seed, horizon=args.horizon, width=args.width, input_sha256=input_hashes,
        gpu_before=before, gpu_after=gpu_state(args.gpu), elapsed_s=time.monotonic()-started,
        checkpoint_sha256=sha(args.output / "best.pt"), training_sequences=len(splits["train"]),
        validation_sequences=len(splits["validation"]), test_sequences=len(splits["test"])))


if __name__ == "__main__": main()
