#!/usr/bin/env python3
"""Freeze admitted shards without changing an existing training snapshot."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from video_data_preparation.shards import freeze_snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--name", default="snapshot.json")
    args = parser.parse_args()
    snapshot = freeze_snapshot(args.corpus, args.corpus/args.name)
    print(snapshot["counts"])
