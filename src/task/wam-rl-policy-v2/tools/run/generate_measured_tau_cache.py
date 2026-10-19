"""Generate the new teacher-goal WAM tau cache for the measured corpus.

The cache is keyed by the audited same-execution prefix and uses the v4
checkpoint only as a frozen tau predictor.  It never reads the old reference
goal or old tau cache.  Records can be sharded by ``index % world``; each
worker writes disjoint row ranges in the preallocated split arrays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def prepare_layout(out, manifest):
    """Create flat tau/valid arrays and immutable record row ranges."""
    records = []
    offsets = {"train": 0, "dev": 0}
    for split in ("train", "dev"):
        for row in manifest["records"]:
            actual = "dev" if row["split"] == "val" else row["split"]
            n = int(row["training_prefix_commands"])
            if actual != split or row["engineering"] != "PASS" or n <= 0:
                continue
            records.append(dict(sequence=row["sequence"], object=row["object"], split=split,
                                source=row["source"], sidecar=row["sidecar"], commands=n,
                                lo=offsets[split], hi=offsets[split] + n))
            offsets[split] += n
    for split in ("train", "dev"):
        np.lib.format.open_memmap(out / f"{split}-tau.npy", mode="w+", dtype=np.float32,
                                 shape=(offsets[split], 264))[:] = 0
        np.lib.format.open_memmap(out / f"{split}-valid.npy", mode="w+", dtype=np.bool_,
                                 shape=(offsets[split],))[:] = False
    write(out / "layout.json", dict(schema="ref2dex.measured-scheduled-tau-layout.v1",
                                    status="READY", records=records,
                                    splits={s: dict(commands=offsets[s]) for s in ("train", "dev")}))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wam", type=Path, default=ROOT / "outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt")
    p.add_argument("--geometry-cache", type=Path, default=ROOT / "outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache")
    p.add_argument("--gpu", type=int, default=7)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--max-records", type=int)
    p.add_argument("--seconds", type=int, default=21600)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not 1 <= a.world <= 8 or not 0 <= a.rank < a.world or not 60 <= a.seconds <= 21600:
        raise ValueError("bounded shard configuration required")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(ROOT / "tmp" / out.name),
                      OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", REF2DEX_DEEP_OPT="all")
    (ROOT / "tmp" / out.name).mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(TASK / "src"))
    import torch
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v2.inspire_contract import NATIVE_TO_WAM
    from wam_rl_policy_v2.goal_scheduler import OrderedGoalScheduler, rigid_goal_delta
    from wam_rl_policy_v2.task_spec import TaskSpec
    from wam_rl_policy_v2.wam_v4_runtime import V4Runtime, collate_v4, v4_online_samples
    from wam_rl_policy_v2.inspire_shadow import tau_world

    audit = json.loads(a.audit_manifest.resolve().read_text())
    if audit.get("status") != "COMPLETED" or not audit.get("training_allowed") or not audit.get("quality_uses_training_reference") is False:
        raise ValueError("completed measured-rollout admission required")
    layout_path = out / "layout.json"
    if not layout_path.exists():
        if a.rank != 0:
            raise ValueError("rank0 must initialize tau layout")
        records = prepare_layout(out, audit)
    else:
        records = json.loads(layout_path.read_text())["records"]
    meta = json.loads(layout_path.read_text())
    if meta.get("status") != "READY":
        raise ValueError("invalid tau layout")
    # The old cache is used only as a read-only geometry binding.  New goals,
    # observations and row identities all come from the measured corpus.
    old_meta = json.loads((a.geometry_cache / "manifest.json").read_text())
    env_objects = old_meta["records"]
    object_index = {r["object_code"].split("/", 1)[-1]: i for i, r in enumerate(env_objects)}
    dense = a.geometry_cache / "surface-manifest.json"
    runtime = V4Runtime(a.wam.resolve(), env_objects, a.geometry_cache.resolve(),
                        dense_manifest=dense, allow_geometry_replacement=True, device="cuda:0")
    tau_arrays = {s: np.load(out / f"{s}-tau.npy", mmap_mode="r+") for s in ("train", "dev")}
    valid_arrays = {s: np.load(out / f"{s}-valid.npy", mmap_mode="r+") for s in ("train", "dev")}
    assigned = [r for i, r in enumerate(records) if i % a.world == a.rank]
    if a.max_records is not None:
        assigned = assigned[:a.max_records]
    started = time.monotonic(); done = 0; valid_count = 0
    for record in assigned:
        if time.monotonic() - started > a.seconds or (out / "STOP").exists():
            raise TimeoutError("owned tau cache stop condition")
        source = Path(record["source"]); sidecar = Path(a.audit_manifest).parent / record["sidecar"]
        spec = TaskSpec.from_packet(json.loads((sidecar / "task.json").read_text()))
        with np.load(source / "trajectory.npz") as z:
            poses = z["obj"][:record["commands"]].copy()
            hands = z["hand"][:record["commands"]].copy()[:, NATIVE_TO_WAM]
        geo_index = object_index[record["object"]]
        geometry = runtime.geometry[geo_index]
        scheduler = OrderedGoalScheduler(spec.object_path, spec.source_frames / spec.source_fps,
                                          geometry[2], interaction_points=geometry[3], goal_type=spec.goal_type)
        taus = tau_arrays[record["split"]][record["lo"]:record["hi"]]
        valids = valid_arrays[record["split"]][record["lo"]:record["hi"]]
        samples = []; rotations = []; local_indices = []
        for t, (pose, hand) in enumerate(zip(poses, hands)):
            decision = scheduler.update(pose, t / spec.source_fps, hand_points=hand)
            if t < 3:
                continue
            history = poses[t - 3:t + 1]
            hand_history = hands[t - 3:t + 1]
            objects = np.c_[history[:, :3, 3], Rotation.from_matrix(history[:, :3, :3]).as_quat()][None]
            goal = rigid_goal_delta(pose, decision["goal_pose"], geometry[2])[None]
            sample = v4_online_samples(objects, hand_history[None], [geometry], goal_delta=goal)[0]
            if not runtime.workspace_valid([sample])[0]:
                continue
            samples.append(sample); rotations.append(pose[:3, :3].copy()); local_indices.append(t)
        for start in range(0, len(samples), 64):
            batch = collate_v4(samples[start:start + 64])
            batch = {k: v.to(runtime.device, non_blocking=True) for k, v in batch.items()}
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                trajectory = runtime.predict_hand(batch)["trajectory"].float().cpu().numpy()
            tau = tau_world(trajectory, np.asarray(rotations[start:start + 64]))
            for j, t in enumerate(local_indices[start:start + 64]):
                taus[t] = tau[j].reshape(-1); valids[t] = True; valid_count += 1
        taus.flush(); valids.flush(); done += 1
        report = dict(status="RUNNING", rank=a.rank, world=a.world, records_done=done,
                      records_total=len(assigned), valid_rows=valid_count,
                      elapsed_s=time.monotonic() - started,
                      gpu=subprocess.check_output(["nvidia-smi", "-i", str(a.gpu),
                          "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"], text=True).strip())
        write(out / f"progress-{a.rank}.json", report)
        print(json.dumps(report), flush=True)
    write(out / f"completion-{a.rank}.json", dict(status="COMPLETED", rank=a.rank, world=a.world,
        records_done=done, records_total=len(assigned), valid_rows=valid_count,
        elapsed_s=time.monotonic() - started, wam=str(a.wam.resolve()),
        wam_sha256=hashlib.sha256(a.wam.read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
