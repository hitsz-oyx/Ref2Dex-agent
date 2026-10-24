"""Zero-update sample-path gate for the old-workspace C2 flow-only arm."""
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / "src/task/ptv3-shared-wam-v6"
sys.path[:0] = [str(TASK / "src"), str(TASK / "tools/run"), str(TASK / "tools/audit"), str(ROOT / "src/task/cm-pointflow-effect-pretrain/src")]
from diagnostic_components import DiagnosticWindows
from remote_staged_data import read
from workspace_origin_preflight import group_check

DIAG = ROOT / "tmp/rh/diagnosis"
OUT = DIAG / "C2-flow-identity"
ALLOWED = {"arm", "stage", "flow_only", "recipe_path", "sampling", "git_commit"}
ORIGIN = [-2.0, -2.0, -2.0]


def open_ds(recipe):
    return DiagnosticWindows(
        recipe["manifest"], "train", recipe["geometry_manifest"], recipe["manifest_sha256"],
        probabilities=recipe["source_probabilities"], curated=recipe.get("curated_manifest"),
        validity_gate=recipe.get("validity_gate"))


def replay(recipe):
    ds = open_ds(recipe)
    rows = []
    for step in range(1, 501):
        for rank in range(4):
            for index in ds.draw(96, 170, step, rank):
                sid, slot, tick = map(int, ds.index[int(index)])
                row = ds.rows[sid]
                rows.append((int(index), int(row["source_id"]), str(row["key"]), int(slot), int(tick)))
    return ds, rows


def digest(rows):
    blob = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def grid_failures(ds, rows, recipe):
    groups = defaultdict(list)
    seen = {}
    for index, _source, _key, slot, tick in rows:
        seen.setdefault(index, (slot, tick))
    for index, (slot, tick) in seen.items():
        sid = int(ds.index[index][0])
        groups[(sid, slot)].append((index, tick))
    tasks = []
    for (sid, slot), items in groups.items():
        row = ds.rows[sid]
        name = row["objects"][slot]
        key = "s%d:%s" % (row["source_id"], name) if row["source_id"] < 3 else name
        geometry = Path(recipe["geometry_manifest"]).parent / ds.geometry[key]["path"]
        tasks.append((row, slot, items, geometry, np.array(ORIGIN, np.float32)))
    bad = []
    with ThreadPoolExecutor(8) as pool:
        for values in pool.map(group_check, tasks):
            bad.extend(values)
    return sorted(set(bad))


def main():
    started = time.monotonic()
    card = TASK / "docs/experiments/probes/P-20261024-c2-flow-old-workspace.md"
    if not card.exists():
        raise SystemExit("interpretation card missing")
    OUT.mkdir(parents=True, exist_ok=True)
    c2 = read(DIAG / "C2/recipe-v2.json")
    flow = read(DIAG / "C2-flow/recipe-v2.json")
    changed = sorted(k for k in set(c2) | set(flow) if c2.get(k) != flow.get(k))
    recipe_ok = set(changed) <= ALLOWED and flow["flow_only"] is True and flow["arm"] == "C2-flow"
    config = read(c2["config"])
    origin_ok = list(config["voxel_origin_m"]) == ORIGIN and float(config["voxel_m"]) == 0.005
    print("recipe", changed, recipe_ok, origin_ok, flush=True)
    _ds, left = replay(c2)
    print("c2-draws", len(left), flush=True)
    ds, right = replay(flow)
    same_path = left == right and len(left) == 192000
    print("path", same_path, digest(left), flush=True)
    bad = grid_failures(ds, left, c2) if same_path else []
    print("grid-bad", len(bad), bad[:8], flush=True)
    panel = None
    forward = None
    trainable = None
    if recipe_ok and origin_ok and same_path and not bad:
        panel, forward, trainable = gpu_checks(flow)
        print("gpu", json.dumps(dict(panel=panel, forward=forward, trainable=trainable)), flush=True)
    path_ok = same_path and not bad
    panel_ok = panel is not None and panel["max_abs_delta_mm"] <= 0.001
    freeze_ok = forward is not None and forward["equal"] and trainable["outside_flow_future"] == 0 and trainable["flow_future"] > 0
    passed = bool(recipe_ok and origin_ok and path_ok and panel_ok and freeze_ok)
    result = dict(
        status="PASS_IDENTITY" if passed else "IDENTITY_FAIL",
        updates=0, optimizer_steps=0, origin_m=ORIGIN, voxel_m=0.005,
        recipe_changes=changed, recipe_ok=recipe_ok, origin_ok=origin_ok,
        prescribed_draws=len(left), path_equal=same_path, path_sha256=digest(left),
        unique_windows=len({row[0] for row in left}), bad_windows=bad[:8], bad_window_count=len(bad),
        panel=panel, forward=forward, trainable=trainable,
        elapsed_s=time.monotonic() - started, training_authorized=passed)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VERDICT", result["status"], flush=True)
    if not passed:
        raise SystemExit(2)


def gpu_checks(recipe):
    import torch
    from ptv3_shared_wam_v6.model import SharedWAMV6
    from ptv3_shared_wam_v6.precision import configure_strict_fp32
    from ptv3_shared_wam_v6.goal_policy import assert_unknown_inputs
    from remote_staged_data import sha
    from train_diagnosis import panel

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(170)
    config = read(recipe["config"])
    model = SharedWAMV6(read(recipe["stats"]), config).cuda()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("flow_future."))
    saved = torch.load(recipe["initialization"], map_location="cpu", weights_only=False)
    assert saved["step"] == 3000 and sha(recipe["initialization"]) == recipe["initialization_sha256"]
    model.load_state_dict(saved["model"], strict=True)
    del saved
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
    trainable = dict(
        flow_future=sum(name.startswith("flow_future.") for name in names),
        outside_flow_future=sum(not name.startswith("flow_future.") for name in names),
        frozen=len(frozen), total=len(names) + len(frozen))
    other = SharedWAMV6(read(recipe["stats"]), config).cuda()
    other.load_state_dict(model.state_dict(), strict=True)
    meta = read(recipe["panel_manifest"])
    path = Path(recipe["panel_manifest"]).parent / meta["files"][0]["path"]
    with np.load(path, allow_pickle=False) as blob:
        batch = {key: torch.from_numpy(blob[key]).cuda() for key in blob.files}
    assert_unknown_inputs(batch)
    configure_strict_fp32(model, dict(model.config, fp32_attention_backend="math"))
    configure_strict_fp32(other, dict(other.config, fp32_attention_backend="math"))
    model.eval(); other.eval()
    with torch.inference_mode():
        left, right = model(batch), other(batch)
    forward = dict(
        equal=bool(torch.equal(left["trajectory"], right["trajectory"]) and torch.equal(left["scene"], right["scene"])),
        file=meta["files"][0]["path"])
    del other, left, right, batch
    scored = panel(model, recipe["panel_manifest"], OUT / "panel-replay", 0, recipe["local3000_panel"])
    reference = read(DIAG / "C2/panel-step000000/summary.json")["means_mm"]
    delta = [float(a - b) for a, b in zip(scored["means_mm"], reference)]
    panel_result = dict(
        means_mm=scored["means_mm"], c2_init_means_mm=reference, delta_vs_c2_init_mm=delta,
        max_abs_delta_mm=float(max(abs(v) for v in delta)))
    return panel_result, forward, trainable


if __name__ == "__main__":
    main()
