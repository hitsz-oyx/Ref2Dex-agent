"""Fit an absolute-plan direct executor with physical control-space PD loss."""

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
from fit_direct_tau_executor_bc_r3 import as_env, build, gpu_state, sha, write

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
    p.add_argument("--seed", type=int, default=433)
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
    selected = sorted(selected)[:40]
    if len(selected) < 20:
        raise ValueError("fewer than twenty lift sequences passed selection")
    n_train, n_val = int(len(selected) * .70), int(len(selected) * .15)
    return dict(train=selected[:n_train], validation=selected[n_train:n_train + n_val], test=selected[n_train + n_val:])


def anchor_from(hand, obj):
    rotation, center = obj[..., :3, :3], obj[..., :3, 3]
    return np.einsum("...pj,...jk->...pk", hand - center[..., None, :], rotation).reshape(*hand.shape[:-2], 33).astype(np.float32)


def load_original(root, horizon):
    splits = select_sequences(root); datasets = {}
    for split, rows in splits.items():
        chunks = []
        for _, path in rows:
            with np.load(path, allow_pickle=False) as packet:
                packet = {k: packet[k] for k in packet.files}; p = as_env(packet)
                chunk = build(packet, [0], horizon)
                ticks = np.arange(p["action"].shape[0] - horizon, dtype=np.int64)
                chunk["q"] = p["q"][ticks, 0].astype(np.float32)
                chunk["pd_target"] = p["pd_targets"][ticks].astype(np.float32)
                chunk["anchor"] = anchor_from(p["hand"][ticks, 0], p["obj"][ticks, 0])
                chunks.append(chunk)
        datasets[split] = {key: np.concatenate([c[key] for c in chunks], axis=0) for key in chunks[0]}
    return splits, datasets


def load_query(path, source_path, rollin_path):
    with np.load(path, allow_pickle=False) as f: data = {k: f[k].copy() for k in f.files}
    with np.load(source_path, allow_pickle=False) as f: source = {k: f[k].copy() for k in f.files}
    with np.load(rollin_path, allow_pickle=False) as f: rollin = {k: f[k].copy() for k in f.files}
    controls, envs = data["state"].shape[:2]
    flat = {}
    for key in ("state", "tau", "previous", "label", "pd_target", "q", "env", "tick"):
        value = data[key]; flat[key] = value.reshape((controls * envs,) + value.shape[2:])
    env, tick = flat["env"].astype(np.int64), flat["tick"].astype(np.int64)
    ref_hand = source["hand"][tick, 24 + env]
    ref_obj = source["obj"][tick, 24 + env]
    flat["anchor"] = anchor_from(ref_hand, ref_obj)
    if "intervention" in data:
        flat["intervention"] = data["intervention"].reshape(-1).astype(np.bool_)
    flat["query_clipped"] = rollin["query_clipped"].reshape(-1).astype(np.bool_)
    return data, flat


def encode(data, mean, scale):
    value = np.concatenate((data["state"], data["tau"].reshape(len(data["tau"]), -1), data["anchor"], data["previous"]), axis=1)
    state_end, tau_end = 87, 87 + 8 * 33
    a = 0; state = (value[:, a:a + state_end] - mean[a:a + state_end]) / scale[a:a + state_end]; a += state_end
    tau = (value[:, a:a + 264] - mean[a:a + 264]) / scale[a:a + 264]; a += 264
    anchor = (value[:, a:a + 33] - mean[a:a + 33]) / scale[a:a + 33]; a += 33
    previous = (value[:, a:] - mean[a:] ) / scale[a:]
    return [state.astype(np.float32), tau.reshape(len(tau), 8, 33).astype(np.float32), anchor.astype(np.float32), previous.astype(np.float32)]


def control_target(pd, q):
    value = pd[:, ACTIVE].copy()
    value[:, :6] -= q[:, :6]
    return value.astype(np.float32)


def active_pd_tensor(action, q, offset, native_scale, torch):
    from trajectory_policy.pd_retargeter import pd_from_action
    full = action.new_zeros((len(action), 18)); full[:, ACTIVE.tolist()] = action
    return pd_from_action(full, q, offset, native_scale)[:, ACTIVE.tolist()]


def control_target_tensor(pd, q, torch):
    value = pd.clone(); value[:, :6] -= q[:, :6]; return value


def group_loss(pred, target, norm, torch):
    err = (pred - target) / norm
    return .5 * torch.nn.functional.smooth_l1_loss(err[:, :6], torch.zeros_like(err[:, :6])) + .5 * torch.nn.functional.smooth_l1_loss(err[:, 6:], torch.zeros_like(err[:, 6:]))


def metrics(policy, data, mean, scale, offset, native_scale, norm, device, torch):
    x = encode(data, mean, scale); q = torch.from_numpy(data["q"]).to(device)
    target = torch.from_numpy(control_target(data["pd_target"], data["q"])).to(device)
    with torch.inference_mode():
        action = policy(*(torch.from_numpy(v).to(device) for v in x)); pred = active_pd_tensor(action, q, offset, native_scale, torch); pred_control = control_target_tensor(pred, q, torch)
    raw = (pred_control - target).cpu().numpy(); diff = action.cpu().numpy() - data["label"]
    return dict(action_l1=float(np.abs(diff).mean()), control_l1=float(np.abs(raw).mean()), control_rmse=float(np.sqrt(np.mean(raw ** 2))),
                xyz_rmse_mm=float(np.sqrt(np.mean(raw[:, :3] ** 2)) * 1000), rotation_rmse_rad=float(np.sqrt(np.mean(raw[:, 3:6] ** 2))),
                finger_rmse_rad=float(np.sqrt(np.mean(raw[:, 6:] ** 2))), normalized_huber=float(group_loss(pred_control, target, norm, torch).item()),
                max_abs=float(np.abs(raw).max()))


def warm_start(old_payload, model, torch):
    from trajectory_policy.tau_executor import TemporalTauExecutorPolicy
    old = TemporalTauExecutorPolicy(old_payload["horizon"], old_payload["width"]); old.load_state_dict(old_payload["state_dict"], strict=True)
    with torch.no_grad():
        model.tau_input.load_state_dict(old.tau_input.state_dict()); model.position.copy_(old.position)
        model.tau_encoder.load_state_dict(old.tau_encoder.state_dict()); model.head.load_state_dict(old.head.state_dict())
        model.state_encoder[0].weight[:, :87].copy_(old.state_encoder[0].weight[:, :87])
        model.state_encoder[0].weight[:, 87 + 33:].copy_(old.state_encoder[0].weight[:, 87:])
        model.state_encoder[0].bias.copy_(old.state_encoder[0].bias)


def main():
    args = parse(); args.output = args.output.resolve()
    if args.output.exists() or ROOT / "outputs/trajectory-policy" not in args.output.parents: raise ValueError("fresh task output required")
    before = gpu_state(args.gpu)
    if before["utilization"] > 15 or before["used_mib"] > 1024: raise RuntimeError("GPU occupied: " + repr(before))
    if Path(sys.prefix).resolve() != RUNTIME.resolve(): raise RuntimeError("pinned runtime required")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(ROOT / "tmp/trajectory-policy/absolute-plan-fit"), OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    import torch
    from trajectory_policy.tau_executor import AbsolutePlanTauExecutorPolicy
    if torch.__version__ != "2.0.1+cu118" or not torch.cuda.is_available(): raise RuntimeError("pinned CUDA runtime required")
    torch.set_num_threads(2); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed); device = torch.device("cuda:0")
    old_payload = torch.load(args.base_checkpoint, map_location="cpu")
    splits, original = load_original(args.rl660_root, args.horizon)
    query_raw, query = load_query(args.query, args.unseen_trajectory, args.rollin_trajectory)
    train_mask, val_mask = query["env"] < 5, query["env"] >= 5
    query_train, query_val = ({k: v[m] for k, v in query.items()} for m in (train_mask, val_mask))
    value = np.concatenate((original["train"]["state"], original["train"]["tau"].reshape(-1, 264), original["train"]["anchor"], original["train"]["previous"]), axis=1)
    mean, scale = value.mean(0).astype(np.float32), value.std(0).astype(np.float32); scale[scale < 1e-4] = 1.
    train_control = np.concatenate((control_target(original["train"]["pd_target"], original["train"]["q"]), control_target(query_train["pd_target"], query_train["q"])), axis=0)
    norm_np = np.maximum(train_control.std(0), .05).astype(np.float32); norm = torch.from_numpy(norm_np).to(device)
    offset = torch.from_numpy(query_raw["native_offset"]).to(device); native_scale = torch.from_numpy(query_raw["native_scale"]).to(device)
    original_x = [torch.from_numpy(v).to(device) for v in encode(original["train"], mean, scale)]; query_x = [torch.from_numpy(v).to(device) for v in encode(query_train, mean, scale)]
    original_y = torch.from_numpy(original["train"]["label"]).to(device); query_y = torch.from_numpy(query_train["label"]).to(device)
    original_q = torch.from_numpy(original["train"]["q"]).to(device); query_q = torch.from_numpy(query_train["q"]).to(device)
    original_pd = torch.from_numpy(control_target(original["train"]["pd_target"], original["train"]["q"])).to(device); query_pd = torch.from_numpy(control_target(query_train["pd_target"], query_train["q"])).to(device)
    # Keep a frozen warm-start copy for an apples-to-apples offline baseline.
    # The previous script accidentally reported the trained policy twice under
    # ``base_query_validation_pd``; this probe keeps the baseline immutable.
    base_policy = AbsolutePlanTauExecutorPolicy(args.horizon, args.width).to(device)
    warm_start(old_payload, base_policy, torch); base_policy.eval()
    policy = AbsolutePlanTauExecutorPolicy(args.horizon, args.width).to(device); warm_start(old_payload, policy, torch)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4); generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True); started = time.monotonic(); best = None
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds: break
        n = min(args.batch_size // 2, len(original_y)); m = min(args.batch_size // 2, len(query_y)); oi = torch.randint(len(original_y), (n,), generator=generator, device=device); qi = torch.randint(len(query_y), (m,), generator=generator, device=device)
        po = policy(*(x[oi] for x in original_x)); pq = policy(*(x[qi] for x in query_x)); pdo = control_target_tensor(active_pd_tensor(po, original_q[oi], offset, native_scale, torch), original_q[oi], torch); pdq = control_target_tensor(active_pd_tensor(pq, query_q[qi], offset, native_scale, torch), query_q[qi], torch)
        lpd = .5 * group_loss(pdo, original_pd[oi], norm, torch) + .5 * group_loss(pdq, query_pd[qi], norm, torch); la = .5 * torch.nn.functional.smooth_l1_loss(po, original_y[oi]) + .5 * torch.nn.functional.smooth_l1_loss(pq, query_y[qi]); loss = lpd + .1 * la
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.); optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            qval = metrics(policy, query_val, mean, scale, offset, native_scale, norm, device, torch); oval = metrics(policy, original["validation"], mean, scale, offset, native_scale, norm, device, torch); score = qval["normalized_huber"] + .25 * oval["normalized_huber"]; row = dict(step=step, loss=float(loss.item()), pd_loss=float(lpd.item()), action_loss=float(la.item()), query_val=qval, original_val=oval, selection_score=float(score), elapsed_s=time.monotonic() - started, gpu=gpu_state(args.gpu)); print(json.dumps(row), flush=True); (args.output / "monitor.jsonl").open("a").write(json.dumps(row) + "\n")
            if best is None or score < best["selection_score"]: best = dict(selection_score=float(score), query_val=qval, original_val=oval, step=step, state={k: v.detach().cpu().clone() for k, v in policy.state_dict().items()})
    policy.load_state_dict(best["state"]); policy.eval()
    with np.load(args.unseen_trajectory, allow_pickle=False) as f: unseen_packet = {k: f[k] for k in f.files}
    unseen = build(unseen_packet, list(range(as_env(unseen_packet)["action"].shape[1])), args.horizon); p = as_env(unseen_packet); ticks = np.arange(p["action"].shape[0] - args.horizon); unseen["q"] = np.concatenate([p["q"][ticks, e] for e in range(p["q"].shape[1])]); unseen["pd_target"] = np.concatenate([p["pd_targets"][ticks, e] for e in range(p["pd_targets"].shape[1])]); unseen["anchor"] = np.concatenate([anchor_from(p["hand"][ticks, e], p["obj"][ticks, e]) for e in range(p["q"].shape[1])])
    checkpoint = dict(schema=AbsolutePlanTauExecutorPolicy.schema, horizon=args.horizon, width=args.width, state_dict=policy.state_dict(), mean=mean, scale=scale, pd_norm=norm_np, active_action=ACTIVE.tolist(), inference_contract="live state87 + absolute reference hand anchor33 + ordered reference tau8 + previous active action12 -> direct active native action12", tau_contract="reference-frame tau plus A_ref_t=R_ref_t(h_ref_t-o_ref_t)", base_checkpoint_sha256=sha(args.base_checkpoint), query_sha256=sha(args.query))
    torch.save(checkpoint, args.output / "best.pt")
    result = dict(status="PROMISING" if best["query_val"]["normalized_huber"] < .2 else "UNPROMISING", offline_only=True, loss_contract="control-space target: wrist qdes-q, finger qdes; normalized wrist/finger Huber plus 0.1 action Huber", control_norm=norm_np.tolist(), base_query_validation_pd=metrics(base_policy, query_val, mean, scale, offset, native_scale, norm, device, torch), query_validation=metrics(policy, query_val, mean, scale, offset, native_scale, norm, device, torch), original_validation=metrics(policy, original["validation"], mean, scale, offset, native_scale, norm, device, torch), unseen=metrics(policy, unseen, mean, scale, offset, native_scale, norm, device, torch), query_clipped_fraction=float(query["query_clipped"].mean()), query_clipped_by_env=[float(query["query_clipped"][query["env"] == e].mean()) for e in range(8)], intervention_fraction=float(query["intervention"].mean()) if "intervention" in query else 0.0, intervention_by_env=[float(query["intervention"][query["env"] == e].mean()) for e in range(8)] if "intervention" in query else [], best={k: v for k, v in best.items() if k != "state"})
    write(args.output / "result.json", result); files = [args.base_checkpoint, args.query, args.rollin_trajectory, args.unseen_trajectory, Path(__file__), TASK / "src/trajectory_policy/tau_executor.py", TASK / "src/trajectory_policy/retargeter_data.py"]; write(args.output / "manifest.json", dict(schema="ref2dex.direct-tau-executor.absolute-plan-pd-fit.v1", status="COMPLETED", task="trajectory-policy", experiment_id="P-20261018-absolute-plan-executor", run_id=args.output.name, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), physical_gpu=args.gpu, seed=args.seed, input_sha256={str(Path(p).resolve()): sha(p) for p in files}, checkpoint_sha256=sha(args.output / "best.pt"), gpu_before=before, gpu_after=gpu_state(args.gpu), elapsed_s=time.monotonic() - started, query_clipped_fraction=result["query_clipped_fraction"], control_norm=norm_np.tolist()))


if __name__ == "__main__": main()
