"""Warm-start the direct tau executor with live-state PD-space DAgger loss."""

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
from fit_direct_tau_executor_bc_r3 import as_env, build, encode, gpu_state, sha, write

ACTIVE = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15])
RUNTIME = Path("/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl660-root", type=Path, required=True)
    p.add_argument("--unseen-trajectory", type=Path, required=True)
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--rollin-trajectory", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--seconds", type=float, default=300.)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=432)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def select_sequences(root):
    selected = []
    for path in sorted(root.glob("batches/*/sequences/*/trajectory.npz")):
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
    n_train, n_val = int(len(selected) * .70), int(len(selected) * .15)
    return dict(train=selected[:n_train], validation=selected[n_train:n_train + n_val],
                test=selected[n_train + n_val:])


def load_original(root, horizon):
    splits = select_sequences(root)
    datasets = {}
    for split, rows in splits.items():
        chunks = []
        for _, path in rows:
            with np.load(path, allow_pickle=False) as packet:
                packet = {k: packet[k] for k in packet.files}
                chunk = build(packet, [0], horizon)
                normalized = as_env(packet)
                ticks = np.arange(normalized["action"].shape[0] - horizon, dtype=np.int64)
                chunk["q"] = normalized["q"][ticks, 0].astype(np.float32)
                chunk["pd_target"] = normalized["pd_targets"][ticks].astype(np.float32)
                chunks.append(chunk)
        datasets[split] = {key: np.concatenate([c[key] for c in chunks], axis=0)
                           for key in chunks[0]}
    return splits, datasets


def query_data(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    required = {"state", "tau", "previous", "label", "pd_target", "q", "env", "tick",
                "native_offset", "native_scale"}
    if not required.issubset(data):
        raise ValueError("query packet missing fields: " + repr(sorted(required - set(data))))
    controls, envs = data["state"].shape[:2]
    if data["state"].ndim != 3 or data["tau"].ndim != 4 or data["label"].ndim != 3:
        raise ValueError("query packet must retain [time, env, ...] axes")
    flat = {}
    for key in ("state", "tau", "previous", "label", "pd_target", "q", "env", "tick"):
        value = data[key]
        if value.shape[:2] != (controls, envs):
            raise ValueError("query field axis mismatch: " + key)
        flat[key] = value.reshape((controls * envs,) + value.shape[2:])
    return data, flat


def active_pd_tensor(action, q, offset, scale, torch):
    full = action.new_zeros((len(action), 18))
    full[:, ACTIVE.tolist()] = action
    from trajectory_policy.pd_retargeter import pd_from_action
    return pd_from_action(full, q, offset, scale)[:, ACTIVE.tolist()]


def pd_group_loss(pred, target, pd_scale, torch):
    error = (pred - target) / pd_scale
    wrist = torch.nn.functional.smooth_l1_loss(error[:, :6], torch.zeros_like(error[:, :6]))
    finger = torch.nn.functional.smooth_l1_loss(error[:, 6:], torch.zeros_like(error[:, 6:]))
    return .5 * wrist + .5 * finger


def action_metrics(policy, data, mean, scale, device, torch):
    x = encode(data, mean, scale)
    with torch.inference_mode():
        pred = policy(*(torch.from_numpy(v).to(device) for v in x)).cpu().numpy()
    err = pred - data["label"]
    return dict(active_l1=float(np.abs(err).mean()), active_rmse=float(np.sqrt(np.mean(err ** 2))),
                max_abs=float(np.abs(err).max()))


def pd_metrics(policy, data, mean, scale, offset, native_scale, pd_norm, device, torch):
    x = encode(data, mean, scale)
    q = torch.from_numpy(data["q"]).to(device)
    target = torch.from_numpy(data["pd_target"][:, ACTIVE]).to(device)
    with torch.inference_mode():
        action = policy(*(torch.from_numpy(v).to(device) for v in x))
        pred = active_pd_tensor(action, q, offset, native_scale, torch)
    raw = (pred - target).cpu().numpy()
    norm = raw / pd_norm.detach().cpu().numpy()
    return dict(pd_l1=float(np.abs(raw).mean()), pd_rmse=float(np.sqrt(np.mean(raw ** 2))),
                pd_normalized_huber=float(pd_group_loss(pred, target, pd_norm, torch).item()),
                wrist_pd_rmse=float(np.sqrt(np.mean(raw[:, :6] ** 2))),
                finger_pd_rmse=float(np.sqrt(np.mean(raw[:, 6:] ** 2))),
                max_abs=float(np.abs(raw).max()))


def divergence_stats(query_raw, query, rollin_path, unseen_path, base_policy, policy,
                     mean, scale, offset, native_scale, pd_norm, device, torch):
    with np.load(rollin_path, allow_pickle=False) as f:
        roll = {k: f[k].copy() for k in ("q", "hand", "obj")}
    with np.load(unseen_path, allow_pickle=False) as f:
        source = {k: f[k].copy() for k in ("q", "hand", "obj")}
    controls, envs = query_raw["state"].shape[:2]
    env = query["env"].astype(np.int64)
    tick = query["tick"].astype(np.int64)
    direct = 8 + env
    student_q = roll["q"][tick, direct]
    student_hand = roll["hand"][tick, direct]
    student_obj = roll["obj"][tick, direct]
    ref_q = source["q"][tick, 24 + env]
    ref_hand = source["hand"][tick, 24 + env]
    ref_obj = source["obj"][tick, 24 + env]
    q_err = np.sqrt(np.mean((student_q - ref_q) ** 2, axis=-1))
    hand_err = np.sqrt(np.mean((student_hand - ref_hand) ** 2, axis=(-1, -2)))
    obj_err = np.linalg.norm(student_obj[:, :3, 3] - ref_obj[:, :3, 3], axis=-1)
    group = np.where((hand_err <= .01) & (obj_err <= .01), "near",
                     np.where((hand_err <= .05) & (obj_err <= .05), "moderate", "catastrophic"))
    x = encode(query, mean, scale)
    q_t = torch.from_numpy(query["q"]).to(device)
    target = torch.from_numpy(query["pd_target"][:, ACTIVE]).to(device)
    with torch.inference_mode():
        base_action = base_policy(*(torch.from_numpy(v).to(device) for v in x))
        new_action = policy(*(torch.from_numpy(v).to(device) for v in x))
        base_pd = active_pd_tensor(base_action, q_t, offset, native_scale, torch)
        new_pd = active_pd_tensor(new_action, q_t, offset, native_scale, torch)
    rows = {}
    for name in ("near", "moderate", "catastrophic"):
        mask = group == name
        if not mask.any():
            rows[name] = dict(count=0)
            continue
        base_raw = (base_pd[mask] - target[mask]).cpu().numpy()
        new_raw = (new_pd[mask] - target[mask]).cpu().numpy()
        rows[name] = dict(count=int(mask.sum()), fraction=float(mask.mean()),
            hand_error_m_mean=float(hand_err[mask].mean()), object_error_m_mean=float(obj_err[mask].mean()),
            q_error_rms_mean=float(q_err[mask].mean()),
            base_pd_rmse=float(np.sqrt(np.mean(base_raw ** 2))),
            dagger_pd_rmse=float(np.sqrt(np.mean(new_raw ** 2))),
            base_wrist_pd_rmse=float(np.sqrt(np.mean(base_raw[:, :6] ** 2))),
            dagger_wrist_pd_rmse=float(np.sqrt(np.mean(new_raw[:, :6] ** 2))),
            base_finger_pd_rmse=float(np.sqrt(np.mean(base_raw[:, 6:] ** 2))),
            dagger_finger_pd_rmse=float(np.sqrt(np.mean(new_raw[:, 6:] ** 2))))
    return dict(thresholds=dict(near_hand_m=0.01, near_object_m=0.01,
                                moderate_hand_m=0.05, moderate_object_m=0.05), groups=rows)


def main():
    args = parse(); args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents:
        raise ValueError("fresh task output required")
    before = gpu_state(args.gpu)
    if before["utilization"] > 15 or before["used_mib"] > 1024:
        raise RuntimeError("GPU is occupied: " + repr(before))
    if Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise RuntimeError("pinned DExplore runtime required")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(ROOT / "tmp/trajectory-policy/direct-tau-dagger-pd-fit"),
                      OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    import torch
    from trajectory_policy.tau_executor import TemporalTauExecutorPolicy
    if torch.__version__ != "2.0.1+cu118" or not torch.cuda.is_available():
        raise RuntimeError("pinned DExplore CUDA runtime required")
    torch.set_num_threads(2); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0")
    payload = torch.load(args.base_checkpoint, map_location="cpu")
    if payload.get("schema") != TemporalTauExecutorPolicy.schema:
        raise ValueError("base checkpoint schema mismatch")
    if int(payload["horizon"]) != args.horizon or int(payload["width"]) != args.width:
        raise ValueError("base checkpoint architecture mismatch")
    mean, scale = np.asarray(payload["mean"], np.float32), np.asarray(payload["scale"], np.float32)
    splits, original = load_original(args.rl660_root, args.horizon)
    query_raw, query = query_data(args.query)
    if query_raw["state"].shape[1] != 8:
        raise ValueError("expected eight direct query environments")
    train_mask, val_mask = query["env"] < 5, query["env"] >= 5
    query_train = {k: v[train_mask] for k, v in query.items()}
    query_val = {k: v[val_mask] for k, v in query.items()}
    offset = torch.from_numpy(query_raw["native_offset"]).to(device)
    native_scale = torch.from_numpy(query_raw["native_scale"]).to(device)
    train_pd = np.concatenate((original["train"]["pd_target"][:, ACTIVE], query_train["pd_target"][:, ACTIVE]), axis=0)
    pd_norm_np = np.maximum(train_pd.std(axis=0), .05).astype(np.float32)
    pd_norm = torch.from_numpy(pd_norm_np).to(device)
    original_train_x = [torch.from_numpy(v).to(device) for v in encode(original["train"], mean, scale)]
    query_train_x = [torch.from_numpy(v).to(device) for v in encode(query_train, mean, scale)]
    original_val_x = [torch.from_numpy(v).to(device) for v in encode(original["validation"], mean, scale)]
    query_val_x = [torch.from_numpy(v).to(device) for v in encode(query_val, mean, scale)]
    original_y = torch.from_numpy(original["train"]["label"]).to(device)
    query_y = torch.from_numpy(query_train["label"]).to(device)
    original_q = torch.from_numpy(original["train"]["q"]).to(device)
    query_q = torch.from_numpy(query_train["q"]).to(device)
    original_pd = torch.from_numpy(original["train"]["pd_target"][:, ACTIVE]).to(device)
    query_pd = torch.from_numpy(query_train["pd_target"][:, ACTIVE]).to(device)
    policy = TemporalTauExecutorPolicy(args.horizon, args.width).to(device); policy.load_state_dict(payload["state_dict"], strict=True)
    base_policy = TemporalTauExecutorPolicy(args.horizon, args.width).to(device); base_policy.load_state_dict(payload["state_dict"], strict=True); base_policy.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True); started = time.monotonic(); best = None
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds: break
        n_orig, n_query = min(args.batch_size // 2, len(original_y)), min(args.batch_size // 2, len(query_y))
        oi = torch.randint(len(original_y), (n_orig,), generator=generator, device=device)
        qi = torch.randint(len(query_y), (n_query,), generator=generator, device=device)
        pred_o = policy(*(x[oi] for x in original_train_x)); pred_q = policy(*(x[qi] for x in query_train_x))
        pd_o = active_pd_tensor(pred_o, original_q[oi], offset, native_scale, torch)
        pd_q = active_pd_tensor(pred_q, query_q[qi], offset, native_scale, torch)
        loss_pd = .5 * pd_group_loss(pd_o, original_pd[oi], pd_norm, torch) + .5 * pd_group_loss(pd_q, query_pd[qi], pd_norm, torch)
        loss_action = .5 * torch.nn.functional.smooth_l1_loss(pred_o, original_y[oi]) + .5 * torch.nn.functional.smooth_l1_loss(pred_q, query_y[qi])
        loss = loss_pd + .1 * loss_action
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.); optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            qpd = pd_metrics(policy, query_val, mean, scale, offset, native_scale, pd_norm, device, torch)
            opd = pd_metrics(policy, original["validation"], mean, scale, offset, native_scale, pd_norm, device, torch)
            score = qpd["pd_normalized_huber"] + .25 * opd["pd_normalized_huber"]
            row = dict(step=step, loss=float(loss.item()), pd_loss=float(loss_pd.item()), action_loss=float(loss_action.item()),
                       query_val=qpd, original_val=opd, selection_score=float(score), elapsed_s=time.monotonic() - started, gpu=gpu_state(args.gpu))
            print(json.dumps(row), flush=True)
            with (args.output / "monitor.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
            if best is None or score < best["selection_score"]:
                best = dict(selection_score=float(score), query_val=qpd, original_val=opd, step=step,
                            state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
    if best is None: raise RuntimeError("no PD-Dagger checkpoint selected")
    policy.load_state_dict(best["state"]); policy.eval()
    with np.load(args.unseen_trajectory, allow_pickle=False) as packet:
        unseen_packet = {k: packet[k] for k in packet.files}
    unseen = build(unseen_packet, list(range(as_env(unseen_packet)["action"].shape[1])), args.horizon)
    unseen_env = as_env(unseen_packet)
    unseen_ticks = np.arange(unseen_env["action"].shape[0] - args.horizon, dtype=np.int64)
    unseen["q"] = np.concatenate([unseen_env["q"][unseen_ticks, env]
                                   for env in range(unseen_env["q"].shape[1])], axis=0).astype(np.float32)
    unseen["pd_target"] = np.concatenate([unseen_env["pd_targets"][unseen_ticks, env]
                                           for env in range(unseen_env["pd_targets"].shape[1])], axis=0).astype(np.float32)
    checkpoint = dict(schema=TemporalTauExecutorPolicy.schema, horizon=args.horizon, width=args.width,
                      state_dict=policy.state_dict(), mean=mean, scale=scale, active_action=ACTIVE.tolist(),
                      inference_contract=payload["inference_contract"], tau_contract=payload["tau_contract"],
                      dagger_contract="warm-start r3; primary normalized wrist/finger PD Huber plus 0.1 action Huber; equal original/query batches",
                      pd_norm=pd_norm_np, base_checkpoint_sha256=sha(args.base_checkpoint), query_sha256=sha(args.query))
    torch.save(checkpoint, args.output / "best.pt")
    query_all_metrics = dict(action=action_metrics(policy, query, mean, scale, device, torch),
                             pd=pd_metrics(policy, query, mean, scale, offset, native_scale, pd_norm, device, torch))
    result = dict(status="PROMISING" if best["query_val"]["pd_normalized_huber"] < .20 else "UNPROMISING",
        offline_only=True, loss_contract=checkpoint["dagger_contract"], pd_norm=pd_norm_np.tolist(),
        base_query_validation_pd=pd_metrics(base_policy, query_val, mean, scale, offset, native_scale, pd_norm, device, torch),
        query_train=dict(action=action_metrics(policy, query_train, mean, scale, device, torch), pd=pd_metrics(policy, query_train, mean, scale, offset, native_scale, pd_norm, device, torch)),
        query_validation=query_all_metrics,
        original_validation=dict(action=action_metrics(policy, original["validation"], mean, scale, device, torch), pd=pd_metrics(policy, original["validation"], mean, scale, offset, native_scale, pd_norm, device, torch)),
        original_test=dict(action=action_metrics(policy, original["test"], mean, scale, device, torch), pd=pd_metrics(policy, original["test"], mean, scale, offset, native_scale, pd_norm, device, torch)),
        unseen=dict(action=action_metrics(policy, unseen, mean, scale, device, torch), pd=pd_metrics(policy, unseen, mean, scale, offset, native_scale, pd_norm, device, torch)),
        best={k: v for k, v in best.items() if k != "state"}, query_split=dict(train_envs=list(range(5)), validation_envs=list(range(5, 8)), query_controls=int(query_raw["state"].shape[0])))
    result["divergence"] = divergence_stats(query_raw, query, args.rollin_trajectory, args.unseen_trajectory,
                                              base_policy, policy, mean, scale, offset, native_scale, pd_norm, device, torch)
    write(args.output / "result.json", result)
    files = [args.base_checkpoint, args.query, args.rollin_trajectory, args.unseen_trajectory, Path(__file__),
             TASK / "src/trajectory_policy/tau_executor.py", TASK / "src/trajectory_policy/retargeter_data.py"]
    write(args.output / "manifest.json", dict(schema="ref2dex.direct-tau-dagger-pd-fit.v1", status="COMPLETED",
        task="trajectory-policy", experiment_id="P-20261018-direct-tau-dagger-pd", run_id=args.output.name,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), physical_gpu=args.gpu,
        seed=args.seed, horizon=args.horizon, width=args.width, input_sha256={str(Path(p).resolve()): sha(p) for p in files},
        gpu_before=before, gpu_after=gpu_state(args.gpu), elapsed_s=time.monotonic() - started,
        base_checkpoint_sha256=sha(args.base_checkpoint), query_sha256=sha(args.query), rollin_sha256=sha(args.rollin_trajectory),
        checkpoint_sha256=sha(args.output / "best.pt"), pd_norm=pd_norm_np.tolist(), query_contract=checkpoint["dagger_contract"]))
    write(args.output / "divergence.json", result["divergence"])


if __name__ == "__main__": main()
