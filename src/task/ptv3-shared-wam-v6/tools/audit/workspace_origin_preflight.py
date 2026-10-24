"""No-optimizer check of voxel origin [-4,-4,-4]. Zero parameter updates."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / "src/task/ptv3-shared-wam-v6"
sys.path[:0] = [str(TASK / "src"), str(TASK / "tools/run"), str(ROOT / "src/task/cm-pointflow-effect-pretrain/src")]
from diagnostic_components import DiagnosticWindows
from remote_staged_data import read

DIAG = ROOT / "tmp/rh/diagnosis"
OUT = DIAG / "workspace-repair-preflight"
OLD = np.array([-2, -2, -2], np.float32)
NEW = np.array([-4, -4, -4], np.float32)
VOXEL = np.float32(0.005)


def grid_bad(hand, valid, surface, origin):
    grid = np.floor((hand - origin) / VOXEL)
    bad = (valid & ((grid < 0).any(-1) | (grid > 65535).any(-1) | ~np.isfinite(hand).all(-1))).any(1)
    if surface is not None:
        g = np.floor((surface - origin) / VOXEL)
        if ((g < 0) | (g > 65535) | ~np.isfinite(surface)).any():
            bad[:] = True
    return bad


def group_check(arg):
    row, slot, items, geometry, origin = arg
    folder = Path(row["path"])
    ticks = np.array([tick for _, tick in items], np.int64)
    joints = np.load(folder / "joints.npy", mmap_mode="r")
    joint_valid = np.load(folder / "joint_valid.npy", mmap_mode="r")
    surface_valid = np.load(folder / "surface_valid.npy", mmap_mode="r")
    offsets = np.load(folder / "surface_offsets.npy", mmap_mode="r")
    poses = np.load(folder / "object_poses.npy", mmap_mode="r")
    anchor = np.linalg.inv(poses[ticks, slot]).astype(np.float32)
    current = joints[ticks] @ np.swapaxes(anchor[:, None, :3, :3], -1, -2) + anchor[:, None, None, :3, 3]
    dense = offsets[ticks].astype(np.float32) + joints[ticks, :, None, 0, :]
    dense = dense @ np.swapaxes(anchor[:, None, :3, :3], -1, -2) + anchor[:, None, None, :3, 3]
    hv = joint_valid[ticks].copy()
    dv = surface_valid[ticks].copy()
    for lag in (1, 2, 3):
        hv &= joint_valid[ticks - lag]
        dv &= surface_valid[ticks - lag]
    hv = hv.reshape(len(ticks), -1)
    dv = np.broadcast_to(dv[:, :, None], (len(ticks), 2, 778)).reshape(len(ticks), -1)
    hand = np.concatenate([current.reshape(len(ticks), -1, 3), dense.reshape(len(ticks), -1, 3)], 1)
    valid = np.concatenate([hv, dv], 1)
    surface = np.load(geometry, allow_pickle=False)["surface"]
    bad = grid_bad(hand, valid, surface, origin)
    return [int(items[i][0]) for i in np.flatnonzero(bad)]


def known_window(ds, index, origin):
    import torch
    from oakink_wm.pointworld import fixed_workspace_grid

    sample = ds[index]
    xyz = np.concatenate(
        [sample["xyz"], sample["start_hand"].reshape(-1, 3), sample["dense_hand"].reshape(-1, 3)]
    ).astype(np.float32)
    valid = np.concatenate(
        [sample["point_valid"], sample["start_hand_valid"].ravel(), sample["dense_hand_valid"].ravel()]
    )
    finite = bool(np.isfinite(xyz[valid]).all()) if valid.any() else False
    try:
        fixed_workspace_grid(
            torch.from_numpy(xyz[None]),
            torch.from_numpy(valid[None]),
            origin.tolist(),
            float(VOXEL),
        )
        rejected = False
        error = None
    except ValueError as exc:
        rejected = True
        error = str(exc)
    return dict(index=int(index), finite=finite, rejected=rejected, error=error)


def draw_audit(ds):
    from concurrent.futures import ThreadPoolExecutor

    recipe = read(DIAG / "C2/recipe-v2.json")
    config = read(recipe["config"])
    assert list(config["voxel_origin_m"]) == [-2, -2, -2]
    assert float(config["voxel_m"]) == 0.005
    first = {}
    for step in range(1, 501):
        for rank in range(4):
            for index in ds.draw(96, 170, step, rank):
                first.setdefault(int(index), (step, rank))
    groups = defaultdict(list)
    for index, _when in first.items():
        sid, slot, tick = map(int, ds.index[index])
        groups[(sid, slot)].append((index, tick))
    tasks = []
    for (sid, slot), items in groups.items():
        row = ds.rows[sid]
        name = row["objects"][slot]
        key = "s%d:%s" % (row["source_id"], name) if row["source_id"] < 3 else name
        geometry = Path(recipe["geometry_manifest"]).parent / ds.geometry[key]["path"]
        tasks.append((row, slot, items, geometry, NEW))
    bad = []
    with ThreadPoolExecutor(8) as pool:
        for values in pool.map(group_check, tasks):
            bad.extend(values)
    return dict(
        prescribed_draws=500 * 4 * 96,
        unique_windows=len(first),
        sequence_slot_groups=len(groups),
        bad_unique_windows=len(bad),
        bad_windows_sample=sorted(bad)[:8],
    )


def panel_audit(recipe):
    import torch
    from oakink_wm.pointworld import fixed_workspace_grid
    from ptv3_shared_wam_v6.model import SharedWAMV6
    from ptv3_shared_wam_v6.precision import configure_strict_fp32
    from ptv3_shared_wam_v6.goal_policy import assert_unknown_inputs
    from remote_staged_data import sha

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = read(recipe["config"])
    stats = read(recipe["stats"])
    model = SharedWAMV6(stats, config).cuda().eval()
    saved = torch.load(recipe["initialization"], map_location="cpu", weights_only=False)
    assert saved["step"] == 3000
    assert sha(recipe["initialization"]) == recipe["initialization_sha256"]
    model.load_state_dict(saved["model"], strict=True)
    del saved
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    meta = read(recipe["panel_manifest"])
    base = Path(recipe["panel_manifest"]).parent
    reference = recipe["local3000_panel"]

    def forward_all(origin):
        model.geometry.voxel_origin_m.copy_(torch.tensor(origin, device="cuda", dtype=torch.float32))
        configure_strict_fp32(model, dict(model.config, fp32_attention_backend="math"))
        trajectories = []
        scenes = []
        batches = []
        groups = []
        with torch.inference_mode():
            for row in meta["files"]:
                path = base / row["path"]
                assert sha(path) == row["sha256"]
                with np.load(path, allow_pickle=False) as blob:
                    batch = {key: torch.from_numpy(blob[key]).cuda() for key in blob.files}
                assert_unknown_inputs(batch)
                xyz = torch.cat(
                    (
                        batch["xyz"],
                        batch["start_hand"].flatten(1, 2),
                        batch["dense_hand"].flatten(1, 2),
                    ),
                    1,
                )
                valid = torch.cat(
                    (
                        batch["point_valid"].bool(),
                        batch["start_hand_valid"].bool().flatten(1),
                        batch["dense_hand_valid"].bool().flatten(1),
                    ),
                    1,
                )
                try:
                    grid = fixed_workspace_grid(xyz, valid, model.geometry.voxel_origin_m, model.geometry.voxel_size_m)
                except ValueError as exc:
                    raise RuntimeError("origin %s panel file %s rejected: %s" % (origin, row["path"], exc)) from exc
                counts = []
                ids = torch.arange(xyz.shape[0], device=xyz.device)[:, None].expand_as(valid)
                keys = torch.cat((ids[valid][:, None], grid[valid]), -1)
                for item in range(xyz.shape[0]):
                    chosen = keys[:, 0] == item
                    counts.append(int(torch.unique(keys[chosen][:, 1:], dim=0).shape[0]) if chosen.any() else 0)
                pred = model(batch)
                trajectories.append(pred["trajectory"].float().cpu())
                scenes.append(pred["scene"].float().cpu())
                batches.append({key: batch[key].cpu() for key in ("trajectory_valid", "start_hand_valid", "scene_valid", "object_valid", "interaction_valid", "contact_target", "trajectory_target", "scene_target")})
                groups.append(counts)
        return trajectories, scenes, batches, groups

    old_t, old_s, batches, old_groups = forward_all(OLD.tolist())
    new_t, new_s, _, new_groups = forward_all(NEW.tolist())
    point_deltas = []
    group_mismatch = 0
    sums = []
    counts = []
    for left_t, right_t, left_s, right_s, batch, left_g, right_g in zip(old_t, new_t, old_s, new_s, batches, old_groups, new_groups):
        if left_g != right_g:
            group_mismatch += sum(int(a != b) for a, b in zip(left_g, right_g))
        hv = batch["trajectory_valid"] & batch["start_hand_valid"][:, None]
        sv = batch["scene_valid"] & batch["object_valid"][:, :, None, None]
        cf = (batch["interaction_valid"] & (batch["contact_target"] > 0.25)).any(dim=3).any(dim=1)
        first = torch.arange(24)[None] < 8
        hand_delta = (right_t - left_t).norm(dim=-1) * 1000
        scene_delta = (right_s - left_s).norm(dim=-1) * 1000
        for item in range(left_t.shape[0]):
            values = []
            if hv[item].any():
                values.append(float(hand_delta[item][hv[item]].max()))
            if sv[item].any():
                values.append(float(scene_delta[item][sv[item]].max()))
            point_deltas.append(max(values) if values else 0.0)
        masks = [hv & first[:, :, None, None], hv & cf[:, :, None, None], sv & first[:, None, :, None]]
        errors = [hand_delta] * 2 + [scene_delta]
        # Means below compare the new-origin prediction with the target, matching the fixed panel.
        new_hand = (right_t - batch["trajectory_target"]).norm(dim=-1) * 1000
        new_scene = (right_s - batch["scene_target"]).norm(dim=-1) * 1000
        scored = [new_hand, new_hand, new_scene]
        sums.append(torch.stack([torch.where(mask, err, 0.0).flatten(1).sum(1, dtype=torch.float64) for err, mask in zip(scored, masks)], 1).numpy())
        counts.append(torch.stack([mask.flatten(1).sum(1) for mask in masks], 1).numpy())
        del errors
    sums = np.concatenate(sums)
    counts = np.concatenate(counts)
    means = sums.sum(0) / counts.sum(0)
    with np.load(reference, allow_pickle=False) as ref:
        base = ref["sums"].sum(0) / counts.sum(0)
    delta = means - base
    return dict(
        rows=int(len(point_deltas)),
        max_prediction_delta_mm=float(np.max(point_deltas)),
        group_count_mismatched_rows=int(group_mismatch),
        new_origin_means_mm=means.tolist(),
        delta_vs_local3000_mm=delta.tolist(),
        abs_delta_vs_local3000_mm=np.abs(delta).tolist(),
    )


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    card = TASK / "docs/experiments/probes/P-20261024-workspace-origin-preflight.md"
    if not card.exists():
        raise SystemExit("interpretation card missing")
    recipe = read(DIAG / "C2/recipe-v2.json")
    ds = DiagnosticWindows(
        recipe["manifest"],
        "train",
        recipe["geometry_manifest"],
        recipe["manifest_sha256"],
        geometry_gt_object=False,
        probabilities=recipe["source_probabilities"],
    )
    fault = read(DIAG / "workspace-fault-reproduction.json")["bad_windows"][0]
    known = int(fault["index"])
    print("known-window", known, flush=True)
    old_known = known_window(ds, known, OLD)
    new_known = known_window(ds, known, NEW)
    print("known", json.dumps(dict(old=old_known, new=new_known)), flush=True)
    draws = draw_audit(ds)
    print("draws", json.dumps(draws), flush=True)
    boundary_fail = new_known["rejected"] or not new_known["finite"] or draws["bad_unique_windows"]
    panel = None
    panel_error = None
    if not boundary_fail:
        try:
            panel = panel_audit(recipe)
            print("panel", json.dumps(panel), flush=True)
        except Exception as exc:
            panel_error = str(exc)
            print("panel-error", panel_error, flush=True)
    identity_fail = False
    if panel is not None:
        identity_fail = (
            panel["max_prediction_delta_mm"] > 0.01
            or max(panel["abs_delta_vs_local3000_mm"]) > 0.001
            or panel["group_count_mismatched_rows"] > 0
        )
    if boundary_fail:
        verdict = "BOUNDARY_INSUFFICIENT"
    elif panel is None and panel_error and "origin [-4, -4, -4]" in panel_error and "outside fixed" in panel_error:
        verdict = "NOT_TRANSPARENT"
    elif panel is None:
        verdict = "PANEL_NOT_MEASURED"
    elif identity_fail:
        verdict = "NOT_TRANSPARENT"
    else:
        verdict = "PASS_TRANSPARENT"
    result = dict(
        status=verdict,
        updates=0,
        optimizer_steps=0,
        old_origin_m=OLD.tolist(),
        new_origin_m=NEW.tolist(),
        voxel_m=float(VOXEL),
        known_window=fault,
        known_old=old_known,
        known_new=new_known,
        draws=draws,
        panel=panel,
        panel_error=panel_error,
        elapsed_s=time.monotonic() - started,
        training_authorized=verdict == "PASS_TRANSPARENT",
    )
    target = OUT / "result.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VERDICT", verdict, flush=True)


if __name__ == "__main__":
    main()
