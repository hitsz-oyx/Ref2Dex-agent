#!/usr/bin/env python3
"""Re-audit and combine the frozen 26-episode PointWAM expansion groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(TASK / "tools/run"))
from oakink_wm.pointwam_data import sha256
from qualify_pointwam_train import audit_pack


def combine(selection_path: Path, qualifications: list[Path]) -> dict:
    selection = json.loads(selection_path.read_text())
    selected = {r["member"]: r["partition"] for r in selection}
    if (len(selection) != 26 or len(selected) != 26
            or sum(p == "fit_candidate" for p in selected.values()) != 13
            or sum(p == "dev_candidate" for p in selected.values()) != 13):
        raise ValueError("expected the frozen 26-episode, 13-fit/13-dev selection")
    coverage = {}; clips = []; episodes = []; sources = []; packs = set()
    for path in qualifications:
        data = json.loads(path.read_text())
        if data.get("status") != "COMPLETED" or data.get("training_allowed") is not False:
            raise ValueError(f"incomplete or already-admitted qualification: {path}")
        group = {r["member"]: r["partition"] for r in data["selection"]}
        if len(group) != len(data["selection"]) or len(data["episodes"]) != len(group):
            raise ValueError("group does not account for every selected episode")
        for member, partition in group.items():
            if member in coverage or selected.get(member) != partition:
                raise ValueError("group selection overlaps or differs from frozen expansion")
            coverage[member] = partition
        episode_members = [r["member"] for r in data["episodes"]]
        if set(episode_members) != set(group) or len(set(episode_members)) != len(group):
            raise ValueError("episode report differs from group selection")
        episodes.extend(data["episodes"])
        sources.append(dict(manifest=str(path.resolve()), sha256=sha256(path),
                            git_commit=data["git_commit"], selection_sha256=data["selection_sha256"]))
        for row in data["clips"]:
            if (group.get(row["member"]) != row["partition"]
                    or row["episode_id"] != row["member"].removesuffix(".hdf5")
                    or row["task"] != Path(row["member"]).parent.name):
                raise ValueError("clip ownership differs from selected source")
            pack = Path(row["pack"]).resolve()
            if pack in packs or sha256(pack) != row["pack_sha256"]:
                raise ValueError("duplicate or changed pack")
            packs.add(pack)
            actual_audit = audit_pack(pack, "official_train", row["episode_id"])
            teacher = row["teacher_audit"]
            qualified = (actual_audit["passed"] and teacher["engineering_gates_pass"]
                         and teacher["all_future_valid_tracks"] >= 512
                         and teacher["scene_valid_fraction"] >= .5)
            if bool(qualified) != row["qualified"]:
                raise ValueError("stored-pack re-audit changed the qualification outcome")
            clips.append({**row, "pack": str(pack), "pack_audit": actual_audit})
    if coverage != selected or len(clips) > 52:
        raise ValueError("groups do not cover the frozen expansion exactly")
    passed = [c for c in clips if c["qualified"]]
    fit = [c for c in passed if c["partition"] == "fit_candidate"]
    dev = [c for c in passed if c["partition"] == "dev_candidate"]
    gates = dict(qualified_clips_ge40=len(passed) >= 40,
                 fit_records_ge10=len({c["episode_id"] for c in fit}) >= 10,
                 dev_records_ge10=len({c["episode_id"] for c in dev}) >= 10,
                 tasks_ge8=len({c["task"] for c in passed}) >= 8)
    return dict(status="COMPLETED", experiment_id="P-20261011-pointwam-data-expansion",
                training_allowed=False, selection=selection, selection_sha256=sha256(selection_path),
                qualification_sources=sources, episodes=episodes, clips=clips,
                qualified_clips=len(passed), fit_clips=len(fit), dev_clips=len(dev),
                fit_records=len({c["episode_id"] for c in fit}),
                dev_records=len({c["episode_id"] for c in dev}),
                tasks=len({c["task"] for c in passed}), gates=gates,
                gate_pass=all(gates.values()),
                result="PROMISING" if all(gates.values()) else "UNCLEAR")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--qualifications", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = combine(args.selection, args.qualifications)
    report["git_commit"] = subprocess.check_output(
        ["git", "-C", str(TASK.parents[2]), "rev-parse", "HEAD"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "result", "qualified_clips",
        "fit_clips", "dev_clips", "fit_records", "dev_records", "tasks", "gates")}))


if __name__ == "__main__":
    main()
