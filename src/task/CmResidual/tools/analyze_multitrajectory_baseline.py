"""Summarize first-episode lift by motion and object for a fixed run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "episodes": n,
        "held_lift_successes": sum(bool(row["lift_success"]) for row in rows),
        "held_lift_rate": sum(bool(row["lift_success"]) for row in rows) / n,
        "mean_contact_fraction": sum(row["hand_object_contact_fraction"] for row in rows) / n,
        "mean_max_contact_lift_m": sum(row["max_contact_lift_m"] for row in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "input_manifest.json").read_text())
    names = sorted(item["sequence"] for item in manifest["motions"])
    by_motion: dict[str, list[dict]] = defaultdict(list)
    by_object: dict[str, list[dict]] = defaultdict(list)
    by_seed: dict[str, list[dict]] = defaultdict(list)
    for seed in args.seeds:
        result_path = run / f"eval_s{seed}_e{args.epoch:03d}_full" / "results.json"
        result = json.loads(result_path.read_text())
        if (result["summary"]["num_episodes"] != 64 or
                result["summary"]["early_termination_disabled"] is not True):
            raise ValueError(f"incomplete evaluation: {result_path}")
        for row in result["per_episode"]:
            name = names[int(row["motion_id"])]
            obj = name.split("_")[1]
            by_motion[name].append(row)
            by_object[obj].append(row)
            by_seed[str(seed)].append(row)
    all_rows = [row for group in by_seed.values() for row in group]
    report = {"run_id": run.name, "seeds": args.seeds, "epoch": args.epoch,
              "overall": stats(all_rows),
              "by_seed": {key: stats(value) for key, value in sorted(by_seed.items())},
              "by_motion": {key: stats(value) for key, value in sorted(by_motion.items())},
              "by_object": {key: stats(value) for key, value in sorted(by_object.items())}}
    output = run / f"analysis_e{args.epoch}_s{'_'.join(map(str, args.seeds))}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
