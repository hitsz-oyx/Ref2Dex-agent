#!/usr/bin/env python3
"""Render unadmitted candidates; promote only explicit visually reviewed rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/"src"))
from video_data_preparation.io import atomic_json, confined_path
from video_data_preparation.sample_contract import FIXED_FIELDS
from video_data_preparation.shards import ShardWriter, freeze_snapshot
from video_data_preparation.native_hand import read_native
from video_data_preparation.video import VideoFrames
from video_data_preparation.automatic_masks import color_evidence


def read_row(path, row):
    import h5py
    with h5py.File(path, "r") as f:
        if not 0 <= row < len(f["sample_id"]):
            raise ValueError("review row out of bounds")
        lo, hi = map(int, f["offsets"][row:row+2])
        sample = {k: np.asarray(f[k][row]) for k in FIXED_FIELDS}
        sample.update({k: f[k][lo:hi] for k in ("xyz", "features", "scene_object")})
        meta = json.loads(f["metadata"][row])
        audit = {k: f[k][row] for k in ("track_uv", "track_valid")}
        return sample, meta, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="JSON {reviewer, rows:[{shard,row,decision,reason}]} to promote")
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--mask-audit", type=Path, help="use saved current SAM PNGs for a CPU color consistency screen")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_windows <= 48:
        raise ValueError("fresh bounded review output required")
    root = args.corpus.resolve()
    args.output.mkdir(parents=True)
    if args.review:
        review = json.loads(args.review.read_text())
        if not review.get("reviewer") or not 1 <= len(review["rows"]) <= 48:
            raise ValueError("named reviewer and at most48 explicit rows required")
        profiles, seen = set(), set()
        for entry in review["rows"]:
            if entry["decision"] != "APPROVED":
                continue
            if not entry.get("reason"):
                raise ValueError("visual approval needs a reason")
            path = confined_path(root/"shards", entry["shard"])
            receipt = json.loads(path.with_suffix(".json").read_text())
            sample, meta, audit = read_row(path, int(entry["row"]))
            identity = (meta["episode_key"], meta["tick"])
            if identity in seen or meta.get("target_verification") != "automatic-provisional":
                raise ValueError("duplicate or nonprovisional review")
            seen.add(identity)
            profiles.add(receipt["profile_id"])
            meta.update(training_allowed=True, target_verification="visual-reviewed",
                target_review=dict(reviewer=review["reviewer"], reason=entry["reason"], source_shard=str(path),
                                   source_row=int(entry["row"])))
            writer = ShardWriter(args.output/"shards"/(str(len(seen))+".h5"), receipt["split"],
                                 receipt["session"], receipt["profile_id"])
            writer.append(sample, meta, audit)
            writer.publish()
        if len(profiles) != 1:
            raise ValueError("one nonempty processing profile required")
        atomic_json(args.output/"visual-review.json", review, exclusive=True)
        result = freeze_snapshot(args.output, args.output/"snapshot.json", require_both=False)
        print(json.dumps(dict(counts=result["counts"], training_ready=all(result["counts"].values()))))
        return
    rows = []
    for receipt_path in sorted((root/"shards").glob("*.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("training_allowed"):
            continue
        path = confined_path(root/"shards", receipt["path"])
        for row in range(receipt["samples"]):
            if len(rows) >= args.max_windows:
                break
            sample, meta, audit = read_row(path, row)
            raw, tick = Path(meta["raw_hdf5"]), meta["tick"]
            native = read_native(raw)
            with VideoFrames(raw.with_suffix(".mp4"), native.frames) as video:
                rgb = video.get(range(tick, tick+25))
            import cv2
            color, decision, reason = None, "PENDING", ""
            if args.mask_audit:
                stem = hashlib.sha256(meta["episode_key"].encode()).hexdigest()[:16]+"-"+str(tick)
                folder = args.mask_audit/stem
                target = cv2.imread(str(folder/"target.png"), cv2.IMREAD_GRAYSCALE)
                hand = cv2.imread(str(folder/"hand.png"), cv2.IMREAD_GRAYSCALE)
                if target is None or hand is None or target.shape != rgb.shape[1:3] or hand.shape != target.shape:
                    raise ValueError("saved current mask evidence missing or wrong shape")
                try:
                    color = color_evidence(rgb[0], (target > 0) & ~(hand > 0), meta["target_name"])
                except ValueError as error:
                    decision, reason = "REJECTED", "Current-mask color screen: "+str(error)
            tiles = []
            for step in (0, 8, 16, 24):
                image = cv2.cvtColor(rgb[step], cv2.COLOR_RGB2BGR)
                for point in range(32):
                    if audit["track_valid"][step, point]:
                        uv = tuple(np.rint(audit["track_uv"][step, point]).astype(int))
                        cv2.circle(image, uv, 3, (0, 255, 0), -1)
                cv2.putText(image, "t+%d %s" % (step, meta["target_name"]), (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                tiles.append(image)
            sheet = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
            image_name = "candidate-%03d.jpg" % len(rows)
            cv2.imwrite(str(args.output/image_name), sheet)
            rows.append(dict(shard=path.name, row=row, key=meta["episode_key"], tick=tick, split=receipt["split"],
                target=meta["target_name"], target_evidence=meta["target_evidence"],
                endpoint_valid=int(sample["object_goal_valid"].sum()), per_horizon=meta["per_horizon_valid"],
                image=image_name, color_audit=color, decision=decision, reason=reason))
        if len(rows) >= args.max_windows:
            break
    atomic_json(args.output/"candidates.json", dict(corpus=str(root), rows=rows), exclusive=True)
    print(json.dumps(dict(numeric_candidates=len(rows), output=str(args.output))))


if __name__ == "__main__":
    main()
