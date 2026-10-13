"""CPU-only timing of the exact stateless source draw and batch preparation."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))

import numpy as np
import torch
from ptv3_shared_wam.data import SharedWindows, shared_collate
from ptv3_shared_wam.training import DEFAULT_DATA
from oakink_wm.multisource import mixed_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--no-profile", action="store_true", help="unprofiled wall-clock control")
    args = parser.parse_args()
    if args.output.exists() or ROOT not in args.output.resolve().parents:
        raise ValueError("fresh repository-local profile output required")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    dataset = SharedWindows(DEFAULT_DATA, "train", 32)
    timings, profile = [], cProfile.Profile()
    if not args.no_profile:
        profile.enable()
    for step in range(101, 101 + args.repeats):
        started = time.perf_counter()
        seed = int(np.random.SeedSequence([100, step, 0]).generate_state(1)[0])
        indices = mixed_indices(dataset.base, args.batch_size, seed)
        samples = [dataset[int(i)] for i in indices]
        loaded = time.perf_counter()
        batch = shared_collate(samples)
        finished = time.perf_counter()
        timings.append(dict(step=step, load_s=loaded-started, collate_s=finished-loaded,
                            total_s=finished-started, points=batch["xyz"].shape[1],
                            objects=batch["scene_points"].shape[1]))
    if not args.no_profile:
        profile.disable()
    stream = io.StringIO()
    if not args.no_profile:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
        (args.output / "cpu_profile.txt").write_text(stream.getvalue())
    result = dict(scope="CPU input construction only; no neural computation", profiled=not args.no_profile,
                  batch_size=args.batch_size,
                  timings=timings, mean_s=float(np.mean([r["total_s"] for r in timings])))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
