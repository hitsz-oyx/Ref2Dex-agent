"""Bind dense geometry and recheck thinned windows without altering v3 data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
for name in ("3d-data-preparation", "ptv3-shared-wam-v4"):
    sys.path.insert(0, str(TASK.parent / name / "src"))
from ptv3_shared_wam_v4 import paths
from pose_mesh_data.clean_data import CleanPoseMeshWindows
from pose_mesh_data.dense_surface import apply_surface_overlay
from pose_mesh_data.v4_interaction import V4SamplingPolicy, retain_window
sys.path.insert(0, str(TASK / "tools/audit"))
from build_v4_window_filter_fast import _native
from audit_v4_surface_geometry import audit


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-recipe", type=Path, required=True)
    p.add_argument("--surface-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=1800)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    recipe = json.loads((args.base_recipe / "recipe.json").read_text())
    stats = json.loads((args.base_recipe / "horizon-stats.json").read_text())
    overlay = json.loads(args.surface_manifest.read_text())
    changed = {row["key"] for row in overlay["objects"]}
    recipe["surface_geometry_manifest"] = str(args.surface_manifest.resolve())
    recipe["data_profile"] = "nine-source-v4-target-object-dense512.v1"
    report = audit(recipe)
    write(root / "surface-audit.json", report)
    if not report["qualifies_512"]:
        raise ValueError("not all canonical target geometries have 512 distinct points and unit normals")
    print("all nine sources pass distinct512/unit-normal geometry audit", flush=True)
    main = json.loads((Path(recipe["main3_root"]) / "processed/manifest.json").read_text())
    transitions = {}
    for split in ("train", "dev"):
        old_filter = Path(recipe["window_filter_manifests"][split])
        filtered = json.loads(old_filter.read_text())
        policy_values = dict(filtered["policy"])
        policy_values.pop("schema", None)
        policy = V4SamplingPolicy(**policy_values)
        main_length = sum(source["indices"]["train" if split == "train" else "val"]["windows"]
                          for source in main["sources"])
        dataset = CleanPoseMeshWindows(recipe["native_root"], split, engineering=True, cache_size=8)
        apply_surface_overlay(dataset.geometry, args.surface_manifest)
        kept_blocked, restored = [], Counter()
        checked = 0
        for position, index in enumerate(filtered["blocked_indices"]):
            if position % 10000 == 0 and (time.monotonic() - started > args.seconds
                                          or shutil.disk_usage(root).free < 20 * 2**30):
                raise RuntimeError("dense recipe time or free-space bound")
            keep = False
            if index >= main_length:
                local = index - main_length
                row_id, _ = dataset.index[local]
                row = dataset.meta["sequences"][row_id]
                if row["objects"][row["target_slot"]] in changed:
                    kind, identity = _native(dataset, local, policy)
                    checked += 1
                    keep = retain_window(kind, identity, policy)
                    if keep:
                        restored[str(row["source_id"])] += 1
            if not keep:
                kept_blocked.append(index)
        filtered.update(blocked_indices=kept_blocked, blocked_windows=len(kept_blocked),
                        kept_windows=filtered["total_windows"]-len(kept_blocked), policy=policy.__dict__,
                        dense_geometry_recheck=dict(parent=str(old_filter), checked=checked,
                            restored_by_source=dict(restored), already_kept_windows_preserved=True,
                            surface_geometry_manifest=str(args.surface_manifest.resolve())))
        filtered["parent_counts_before_dense_recheck"] = filtered.pop("counts", {})
        dest = root / ("window-filter-" + split + ".json")
        write(dest, filtered)
        recipe["window_filter_manifests"][split] = str(dest)
        if split == "train":
            for sid, count in restored.items():
                recipe["source_counts"][sid] += count
        transitions[split] = filtered["dense_geometry_recheck"]
        print(split, "rechecked", checked, "restored", dict(restored), flush=True)
    recipe["admission"]["v4_filter"].update(
        train=recipe["window_filter_manifests"]["train"], dev=recipe["window_filter_manifests"]["dev"])
    recipe["limitations"] = ["v3 data/checkpoints untouched; derived dense CAD geometry overlay.",
        "Original 32 flow identities and frozen v3 TRAIN normalization preserved.",
        "Source probabilities remain fixed; underfilled target surfaces now contain 512 distinct points."]
    stats["recipe"] = recipe
    write(root / "recipe.json", recipe)
    write(root / "horizon-stats.json", stats)
    # Exercise actual v4 readers and masks on a real window per source/split.
    from ptv3_shared_wam_v4.data import NineWindows, collate
    panels = {}
    for split in ("train", "dev"):
        data = NineWindows(recipe["main3_root"], recipe["native_root"], split, recipe)
        panel = data.panel(1, 171)
        batch = collate([data[ids[0]] for ids in panel.values()])
        inputs = batch["model_inputs"]
        if inputs["xyz"].shape[1:] != (512, 3):
            raise ValueError("real v4 batch geometry mismatch")
        panels[split] = dict(sources=list(panel), windows=data.counts,
                             interaction_shape=list(inputs["interaction_target"].shape))
        print(split, "real reader/collate passed", flush=True)
    write(root / "result.json", dict(status="COMPLETED", run_id=root.name,
        elapsed_s=time.monotonic()-started, transitions=transitions, panels=panels,
        environment=dict(python=sys.version, executable=sys.executable),
        source_normalization="frozen v3 TRAIN values; not refitted", qualifies_dense512=True))


if __name__ == "__main__":
    main()
