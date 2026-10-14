#!/usr/bin/env python3
"""Freeze a small named-rigid-object cohort from downloaded official TRAIN pairs."""
import argparse
import json
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/"src"))
from video_data_preparation.automatic_masks import target_phrase
from video_data_preparation.io import atomic_json


def main():
    import h5py
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.episodes <= 50:
        raise ValueError("fresh bounded cohort required")
    rows, details, seen, sessions = [], [], set(), {}
    skipped, scanned = {}, 0
    with args.ready.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            scanned += 1
            if scanned > 300000:
                break
            record = json.loads(line)
            if (record.get("release_split") != "official_train" or "/basic_pick_place/" not in record.get("member", "")
                    or record["key"] in seen):
                continue
            seen.add(record["key"])
            try:
                with h5py.File(record["raw_hdf5"], "r") as raw:
                    frames = raw["transforms/camera"].shape[0]
                    if not 52 <= frames <= 240 or "confidences" not in raw:
                        raise ValueError("NATIVE_CLOCK_OR_CONFIDENCE_SCOPE")
                    session = str(raw.attrs.get("session_name", ""))
                    key = "llm_description2" if str(raw.attrs.get("which_llm_description")) == "2" else "llm_description"
                    description = str(raw.attrs.get(key, ""))
                    phrase = target_phrase(description)
                    if not session or sessions.get(session, 0) >= 3:
                        raise ValueError("SESSION_DIVERSITY_QUOTA")
                if not Path(record["raw_hdf5"]).with_suffix(".mp4").is_file():
                    raise ValueError("MP4_NOT_READY")
            except (ValueError, OSError, KeyError) as error:
                reason = str(error)
                skipped[reason] = skipped.get(reason, 0)+1
                continue
            rows.append(record)
            sessions[session] = sessions.get(session, 0)+1
            details.append(dict(key=record["key"], session=session, frames=frames, target=phrase, description=description))
            if len(rows) >= args.episodes:
                break
    if not rows:
        raise ValueError("no eligible downloaded pairs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for row in rows:
            out.write(json.dumps(row)+"\n")
    atomic_json(args.output.with_suffix(".manifest.json"), dict(source=str(args.ready.resolve()),
        episodes=details, scanned_rows=scanned, skipped=skipped, future_used_for_selection=False), exclusive=True)
    print(json.dumps(dict(episodes=len(rows), sessions=len(sessions), scanned=scanned, skipped=skipped)))


if __name__ == "__main__":
    main()
