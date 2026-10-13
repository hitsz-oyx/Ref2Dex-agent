#!/usr/bin/env python3
"""Inspect compact snapshot contracts and optionally render native RGB tracks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/"src"))
from video_data_preparation.io import atomic_json
from video_data_preparation.shards import ShardDataset
from video_data_preparation.video import VideoFrames
from video_data_preparation.native_hand import read_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_windows <= 48 or args.output.exists():
        raise ValueError("fresh audit output and bounded window count [1,48] required")
    if args.output.resolve() == Path("/tmp") or Path("/tmp") in args.output.resolve().parents:
        raise ValueError("audit output must not be in /tmp")
    dataset = ShardDataset(args.snapshot, args.split, engineering=args.engineering)
    args.output.mkdir(parents=True)
    rows = []
    try:
        for index in np.linspace(0, len(dataset)-1, min(args.max_windows, len(dataset)), dtype=int):
            sample = dataset[int(index)]
            audit = dataset.audit(int(index))
            meta = audit["metadata"]
            row = dict(index=int(index), episode_key=meta["episode_key"], tick=meta["tick"],
                target=meta["target_name"], scene_points=len(sample["xyz"]),
                context_points=int((sample["scene_object"] == -2).sum()),
                endpoint_valid=int(sample["object_goal_valid"].sum()),
                hand_current_valid=int(sample["start_hand_valid"].sum()),
                scene_per_horizon_valid=sample["scene_valid"][0].sum(-1).tolist(),
                current_depth_agreement=meta["current_depth_agreement"],
                causal_observation=meta["causal_observation"])
            if args.render:
                import cv2
                raw = Path(meta["raw_hdf5"])
                native = read_native(raw)
                tick = meta["tick"]
                with VideoFrames(raw.with_suffix(".mp4"), native.frames) as video:
                    rgb = video.get(range(tick, tick+25))
                tiles = []
                colors = [tuple(map(int, cv2.applyColorMap(np.array([[i*7 % 256]], np.uint8), cv2.COLORMAP_HSV)[0, 0])) for i in range(32)]
                for step in (0, 8, 16, 24):
                    image = cv2.cvtColor(rgb[step], cv2.COLOR_RGB2BGR)
                    for point in range(32):
                        if not audit["track_valid"][step, point]:
                            continue
                        uv = tuple(np.rint(audit["track_uv"][step, point]).astype(int))
                        cv2.circle(image, uv, 3, colors[point], -1)
                        cv2.putText(image, str(point), uv, cv2.FONT_HERSHEY_SIMPLEX, .3, colors[point], 1)
                    cv2.putText(image, "t+%d %s" % (step, meta["target_name"]), (12, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    tiles.append(image)
                contact = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
                name = "window-%06d.jpg" % index
                if not cv2.imwrite(str(args.output/name), contact):
                    raise IOError("failed to write audit overlay")
                row["overlay"] = name
            rows.append(row)
        atomic_json(args.output/"summary.json", dict(status="COMPLETED", snapshot=str(args.snapshot.resolve()),
            split=args.split, windows=rows, scope="contract/visual audit only; no training or geometry truth claim"))
        print(json.dumps(dict(windows=len(rows), output=str(args.output))))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
