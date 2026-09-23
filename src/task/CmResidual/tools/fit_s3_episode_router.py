"""Fit the pre-registered V1.40 start-frame-only, whole-episode PPO route.

This script deliberately reads only discovery seeds 81--84.  Heldout results
must not enter the route selection or its stopping decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


EXPERTS = ("source", "standard", "back240", "back260")
SEEDS = (81, 82, 83, 84)
SHA256 = {
    "source": "863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c",
    "standard": "3212bcc195d374a4d1cb31b21051e4a0a93f63fc345a0fe987279a562092dccb",
    "back240": "a88924a590966c964b029e067985fea41465f230d459454b8048899cbe8670c2",
    "back260": "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f",
}
RUNS = {
    "source": ("agent_v135_s3_cmoff_s70_e180", 180),
    "standard": ("agent_v139_s3_standard_s70_e260", 260),
    "back240": ("agent_v139_s3_backtrack_s70_e260", 240),
    "back260": ("agent_v139_s3_backtrack_s70_e260", 260),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_discovery(root: Path):
    records = []
    inputs = {}
    oracle = {}
    for seed in SEEDS:
        outcomes = {}
        for name in EXPERTS:
            run, epoch = RUNS[name]
            folder = root / run / f"eval_s{seed}_e{epoch}_full"
            manifest_path = folder / "run_manifest.json"
            results_path = folder / "results.json"
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get("run_status") != "COMPLETED"
                    or manifest.get("checkpoint_sha256") != SHA256[name]
                    or manifest.get("seed") != seed
                    or manifest.get("num_envs") != 64
                    or manifest.get("early_termination_disabled") is not True):
                raise ValueError(f"Invalid discovery run: {folder}")
            payload = json.loads(results_path.read_text())
            episodes = {int(row["env_id"]): row for row in payload["per_episode"]}
            if set(episodes) != set(range(64)) or len(payload["per_episode"]) != 64:
                raise ValueError(f"Expected 64 unique episodes: {results_path}")
            outcomes[name] = episodes
            inputs[f"s{seed}_{name}"] = {
                "manifest": str(manifest_path.resolve()),
                "results": str(results_path.resolve()),
                "results_sha256": _sha256(results_path),
                "checkpoint_sha256": SHA256[name],
            }
        oracle[seed] = 0
        for env_id in range(64):
            aligned = [outcomes[name][env_id] for name in EXPERTS]
            if len({(row["motion_id"], row["start_frame"]) for row in aligned}) != 1:
                raise ValueError(f"Unmatched env s{seed}/{env_id}")
            frame = int(aligned[0]["start_frame"])
            successes = tuple(bool(row["lift_success"]) for row in aligned)
            records.append((seed, env_id, frame, successes))
            oracle[seed] += any(successes)
    return records, inputs, oracle


def _fit(records):
    """Exhaustively optimize <=4 contiguous segments; deterministic tie breaks."""
    frames = sorted({row[2] for row in records})
    weights = [[sum(row[2] == frame and row[3][expert] for row in records)
                for expert in range(len(EXPERTS))] for frame in frames]
    prefix = [[0] * (len(frames) + 1) for _ in EXPERTS]
    for i, row in enumerate(weights):
        for expert in range(len(EXPERTS)):
            prefix[expert][i + 1] = prefix[expert][i] + row[expert]

    # State is (raw_successes, boundary_indices, expert_indices).
    states = {(0, 0): (0, (), ())}
    for segments in range(1, min(4, len(frames)) + 1):
        for end in range(segments, len(frames) + 1):
            best = None
            for start in range(segments - 1, end):
                previous = states.get((segments - 1, start))
                if previous is None:
                    continue
                for expert in range(len(EXPERTS)):
                    if previous[2] and previous[2][-1] == expert:
                        continue  # Identical adjacent experts are one segment.
                    candidate = (
                        previous[0] + prefix[expert][end] - prefix[expert][start],
                        previous[1] + (() if start == 0 else (start,)),
                        previous[2] + (expert,),
                    )
                    key = (candidate[0], tuple(-x for x in candidate[1]),
                           tuple(-x for x in candidate[2]))
                    if best is None or key > best[0]:
                        best = (key, candidate)
            if best is not None:
                states[(segments, end)] = best[1]
    finalists = []
    for segments in range(1, min(4, len(frames)) + 1):
        state = states.get((segments, len(frames)))
        if state is None:
            continue
        score = state[0] - 2 * (segments - 1)
        key = (score, -segments, tuple(-frames[x] for x in state[1]),
               tuple(-x for x in state[2]))
        finalists.append((key, state))
    _, (raw, boundaries, expert_ids) = max(finalists)
    starts = (0,) + boundaries
    ends = boundaries + (len(frames),)
    segments = [{"start_frame": frames[start], "end_frame": frames[end - 1],
                 "expert": EXPERTS[expert]}
                for start, end, expert in zip(starts, ends, expert_ids)]
    # Map every integer frame, including gaps, to its segment. The evaluator's
    # nearest-key fallback handles frames outside the discovery range.
    route = {}
    for frame in range(frames[0], frames[-1] + 1):
        segment = next((i for i in range(1, len(segments))
                        if frame < segments[i]["start_frame"]), len(segments)) - 1
        route[str(frame)] = segments[segment]["expert"]
    achieved = {seed: sum(successes[EXPERTS.index(route[str(frame)])]
                          for row_seed, _, frame, successes in records if row_seed == seed)
                for seed in SEEDS}
    assert sum(achieved.values()) == raw
    return frames, segments, route, raw, achieved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records, inputs, oracle = _load_discovery(args.discovery_root)
    if min(oracle.values()) < 58:
        raise ValueError(f"Pre-registered 58/64 oracle gate failed: {oracle}")
    frames, segments, route, raw, achieved = _fit(records)
    payload = {
        "run_status": "COMPLETED",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "work_version": "V1.40",
        "discovery_seeds": list(SEEDS),
        "heldout_seeds": [85, 86, 87, 88, 89],
        "expert_order": list(EXPERTS),
        "expert_sha256": SHA256,
        "discovery_inputs": inputs,
        "oracle_successes": oracle,
        "oracle_gate_per_seed": 58,
        "observed_start_frames": frames,
        "segments": segments,
        "route_map": route,
        "discovery_route_successes": achieved,
        "raw_successes": raw,
        "penalized_score": raw - 2 * (len(segments) - 1),
        "complexity_penalty_per_extra_segment": 2,
        "official_policy_checkpoint": None,
        "cm_used_by_router": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "oracle_successes", "segments", "discovery_route_successes", "penalized_score")},
        sort_keys=True))


if __name__ == "__main__":
    main()
