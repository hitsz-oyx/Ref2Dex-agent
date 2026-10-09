"""Audit an episode-split history-preserving candidate panel offline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_evaluator.data import sha
from audit_history_candidate_bank import TICKS, query_report, reset_report, validate


SCHEMA = "ref2dex.history-candidate-panel-audit.v1"


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    expected_parent = (ROOT / "outputs/consequence-evaluator").resolve()
    if output.exists() or output.parent != expected_parent:
        raise ValueError("fresh task-owned audit output required")
    paths = {name: path.resolve() for name, path in (
        ("train", args.train), ("val", args.val), ("test", args.test))}
    loaded = {name: validate(path) for name, path in paths.items()}
    actor_hashes = {value[1].get("actor_sha256") for value in loaded.values()}
    if len(actor_hashes) != 1 or None in actor_hashes:
        raise ValueError("episode splits must use one frozen actor")
    if not all(value[1].get("pipeline") == "cpu" and value[1].get("physics_gpu") is True
               for value in loaded.values()):
        raise ValueError("episode split runtime identity mismatch")

    splits = {}
    for name, (_, manifest, data) in loaded.items():
        splits[name] = {
            "run_id": manifest["run_id"],
            "seed": int(manifest["seed"]),
            "episodes": int(data["actor_observation"].shape[1]),
            "frames": int(data["actor_observation"].shape[0]),
            "history_dim": int(data["actor_observation"].shape[2]),
            "clipped_steps": int(data["clipped"].sum()),
            "reset": reset_report(data),
            "ticks": {str(tick): query_report(data, tick) for tick in TICKS},
        }

    positive = []
    for split, report in splits.items():
        for tick, row in report["ticks"].items():
            screen = row["thresholds"]["0.03"]
            if screen["near_pairs"] >= 20 and screen["strict_label_pairs"] >= 10:
                positive.append({"split": split, "tick": int(tick), **screen})
    reset_exact = all(report["reset"]["exact"] for report in splits.values())
    result = {
        "status": "COMPLETED",
        "conclusion": "PROMISING" if reset_exact and positive else "UNPROMISING",
        "schema": SCHEMA,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "episode-split history-preserving structured panel; offline candidate-bank screen only",
        "splits": splits,
        "screen": {
            "history_threshold": 0.03,
            "state_near_rule": "hand RMS <=3 mm and q RMS <=.03",
            "strict_label_delta": 0.02,
            "reset_exact_all_splits": reset_exact,
            "positive_split_ticks": positive,
        },
        "limitations": [
            "one structured rollout per episode split; no evaluator fit or matched candidate panel replay",
            "feed-forward actor observation is used as H; hidden simulator-state equivalence is not established",
            "U32 labels are frozen outcome proxies and thresholds are engineering screens",
        ],
    }
    hashes = {str((path / name).resolve()): sha(path / name)
              for path in paths.values() for name in ("manifest.json", "trajectory.npz")}
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    result["input_sha256"] = hashes
    output.mkdir(parents=True)
    write(output / "result.json", result)
    write(output / "manifest.json", result)
    print(json.dumps({"conclusion": result["conclusion"],
                      "positive_split_ticks": positive}, indent=2), flush=True)


if __name__ == "__main__":
    main()
