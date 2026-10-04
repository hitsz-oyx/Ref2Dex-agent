#!/usr/bin/env python3
"""Explicit post-run source supplement; never rewrite the original manifest."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    began = datetime.fromisoformat(manifest["created_at"]).timestamp()
    command = manifest["command"]
    motion_root = Path(command[command.index("--motion_file")+1])
    rows = []
    for directory in sorted(motion_root.iterdir()):
        motion = directory / "interaction_hand_inspire.pt"
        if not motion.is_file():
            continue
        resolved = motion.resolve()
        stat = resolved.stat()
        rows.append(dict(alias=str(motion), resolved=str(resolved), bytes=stat.st_size,
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(), mtime=stat.st_mtime,
            mtime_before_run=stat.st_mtime <= began))
    assert len(rows) == 3 and all(row["mtime_before_run"] for row in rows)
    report = dict(timing="POST_RUN_SUPPLEMENT", recorded_at=datetime.now(timezone.utc).isoformat(),
        original_manifest_sha256=hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
        sources=rows, limitation="original rglob missed linked motion directories; current hashes and old mtimes support provenance but are not a pre-run hash capture")
    output = run_dir / "input_provenance_supplement.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+"\n")
    log = ROOT / "tmp/ref3" / (run_dir.name+".log")
    if log.is_file():
        shutil.copyfile(log, run_dir / "run.log")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
