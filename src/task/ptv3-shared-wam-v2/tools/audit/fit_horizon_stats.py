"""Fit one shared horizon artifact using exactly the TRAIN source/stratum sampler."""
import argparse
from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from ptv3_shared_wam_v2.paths import ROOT, DEFAULT_DATA, STATS_SCHEMA
from ptv3_shared_wam_v2.normalization import Moments
from ptv3_shared_wam.data import SharedWindows
from ptv3_shared_wam.checkpoint import atomic_json, git_commit
from oakink_wm.multisource import mixed_indices
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--std-floor-m", type=float, default=.001)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / "outputs" / TASK.name not in output.parents or output.exists():
        raise ValueError("fresh v2 output artifact required")
    if not 3 <= args.samples <= 32768:
        raise ValueError("bounded TRAIN sample count required")
    dataset = SharedWindows(args.data, "train", 32)
    for source in dataset.base.sources:
        source.sequence = lru_cache(maxsize=128)(source.sequence)
    indices = mixed_indices(dataset.base, args.samples, args.seed)
    hand, scene = Moments(), Moments()
    sources = Counter()
    started = time.monotonic()
    for n, index in enumerate(indices):
        sample = dataset[int(index)]
        hand_valid = sample["trajectory_valid"] & sample["start_hand_valid"][None]
        hand.update(sample["trajectory_target"], hand_valid)
        scene.update(sample["scene_target"].transpose(1, 0, 2, 3), sample["scene_valid"].transpose(1, 0, 2))
        sources[int(sample["source_id"]) ] += 1
        if (n + 1) % 1024 == 0:
            print(json.dumps(dict(samples=n+1, elapsed_s=time.monotonic()-started)), flush=True)
    if len(sources) != 3:
        raise ValueError("stats sample must cover all three TRAIN domains")
    result = dict(schema=STATS_SCHEMA, split="train", sources=["oakink2", "grab", "arctic"],
        data=str(args.data.resolve()), object_points=32, fps=30, history=4, horizon=24,
        sampler="mixed_indices: source_weights and source_draw strata .6/.2/.2; point-weighted moments",
        source_weights=dataset.base.source_weights.tolist(), sampled_source_counts=dict(sources),
        normalization_source_manifest_sha256=dataset.base.meta.get("normalization_source_manifest_sha256"),
        samples=args.samples, seed=args.seed, std_floor_m=args.std_floor_m,
        hand=hand.result(args.std_floor_m), scene=scene.result(args.std_floor_m),
        provenance=dict(git_commit=git_commit(), elapsed_s=time.monotonic()-started,
                        device="cpu: label arrays and streaming statistics only; no neural computation"))
    output.parent.mkdir(parents=True, exist_ok=False)
    np.save(output.parent / "train_indices.npy", indices)
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output), elapsed_s=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()
