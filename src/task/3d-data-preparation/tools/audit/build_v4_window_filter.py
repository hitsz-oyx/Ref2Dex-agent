"""Build a deterministic v4 static/approach thinning manifest.

The raw corpus is never edited.  The output is a small index consumed by the
v4 recipe, with contact windows always retained and idle/approach windows
thinned by the policy hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from pose_mesh_data.v4_data import V4Main3Windows, V4NativeWindows
from pose_mesh_data.v4_interaction import V4SamplingPolicy, retain_window


def _scan(dataset, offset, policy, counts, blocked):
    for index in range(len(dataset)):
        sample = dataset[index]
        kind = str(sample["interaction_kind"])
        counts[kind] = counts.get(kind, 0) + 1
        identity = f"{sample.get('source_id', 0)}:{np.asarray(sample.get('sample_id', [index])).tolist()}"
        if not retain_window(kind, identity, policy):
            blocked.append(offset + index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main3-root", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--split", choices=("train", "dev"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--static-keep-ratio", type=float, default=.10)
    p.add_argument("--approach-keep-ratio", type=float, default=.35)
    a = p.parse_args()
    policy = V4SamplingPolicy(static_keep_ratio=a.static_keep_ratio, approach_keep_ratio=a.approach_keep_ratio)
    main = V4Main3Windows(a.main3_root, "train" if a.split == "train" else "val", policy=policy)
    native = V4NativeWindows(a.native_root, a.split, policy=policy)
    counts = {}
    blocked = []
    _scan(main, 0, policy, counts, blocked)
    _scan(native, len(main), policy, counts, blocked)
    kept = len(main) + len(native) - len(blocked)
    result = dict(schema="ref2dex.wam-v4-window-filter.v1", split=a.split,
                  policy=policy.__dict__, total_windows=len(main) + len(native),
                  kept_windows=kept, blocked_windows=len(blocked), counts=counts,
                  blocked_indices=blocked,
                  source_roots=[str(a.main3_root.resolve()), str(a.native_root.resolve())])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
