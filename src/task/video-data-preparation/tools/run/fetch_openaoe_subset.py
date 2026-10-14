#!/usr/bin/env python3
"""Fetch an explicitly named, bounded ModelScope subset; no SDK/cache mutation."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

BASE = "https://modelscope.cn/api/v1/datasets/inclusionAI/OpenAoE-2000h/repo"
REQUIRED = {"raw_video.mp4", "video_info.json", "ego_annotation/ego_action_annotation.json",
            "ego_process/ego_hands_reconstruction/hands.npz",
            "ego_process/ego_hands_reconstruction/camera_traj.npz",
            "ego_process/ego_undistorted_video/raw_video_undistorted.mp4",
            "ego_process/ego_undistorted_video/undistorted_video_info.json"}


def request(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "Ref2Dex-bounded-subset"}), timeout=30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-mib", type=int, default=512)
    p.add_argument("--seconds", type=int, default=1200)
    a = p.parse_args()
    if not 1 <= len(a.samples) <= 4 or len(set(a.samples)) != len(a.samples):
        p.error("choose one to four distinct sample IDs")
    if a.max_mib <= 0 or not 0 < a.seconds <= 1800:
        p.error("positive byte budget and deadline <= 1800 seconds required")
    if any(Path(s).name != s or s in {".", ".."} for s in a.samples):
        p.error("sample ID must be a single directory name")
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest = dict(source="ModelScope inclusionAI/OpenAoE-2000h", revision="master",
                    fetched_at=time.time(), samples=a.samples, max_bytes=a.max_mib*2**20,
                    max_seconds=a.seconds, files=[], status="PLANNED")
    for sample in a.samples:
        url = BASE + "/tree?" + urllib.parse.urlencode(dict(
            Revision="master", Root=sample, Recursive="true", PageNumber=1, PageSize=100))
        with request(url) as f:
            tree = json.load(f)
        if tree.get("Code") != 200:
            raise RuntimeError("ModelScope tree response unsuccessful")
        files = {r["Path"][len(sample)+1:]: r for r in tree["Data"]["Files"]
                 if r["Type"] == "blob" and r["Path"].startswith(sample + "/")}
        missing = REQUIRED - set(files)
        if missing:
            raise ValueError("missing complete sample inputs: " + repr(sorted(missing)))
        for rel in sorted(REQUIRED):
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("unsafe repository file path")
            size = int(files[rel]["Size"])
            if size <= 0:
                raise ValueError("unknown source file size")
            manifest["files"].append(dict(path=sample+"/"+rel, bytes=size))
    total = sum(r["bytes"] for r in manifest["files"])
    if total > manifest["max_bytes"]:
        raise ValueError("selected sample bytes exceed budget: %d" % total)
    def save():
        (a.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["expected_bytes"] = total
    manifest["status"] = "RUNNING"
    save()
    print("Download plan", len(a.samples), "samples", total, "bytes", flush=True)
    def fetch(row):
        dest = a.output / "raw" / row["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_size != row["bytes"]:
                raise ValueError("existing input has unexpected size; refusing overwrite")
            return dict(path=row["path"], status="REUSED", bytes=row["bytes"])
        partial = dest.with_name(dest.name + ".partial")
        url = BASE + "?" + urllib.parse.urlencode(dict(Revision="master", FilePath=row["path"]))
        for attempt in range(2):
            if time.monotonic()-start >= a.seconds:
                raise TimeoutError("download deadline")
            try:
                with request(url) as stream, partial.open("wb") as out:
                    count = 0
                    while True:
                        if time.monotonic()-start >= a.seconds:
                            raise TimeoutError("download deadline")
                        chunk = stream.read(1024*1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > row["bytes"]:
                            raise ValueError("source file grew beyond planned size")
                        out.write(chunk)
                if count != row["bytes"]:
                    raise ValueError("incomplete transfer")
                os.chmod(partial, 0o644)
                os.replace(partial, dest)
                return dict(path=row["path"], status="DOWNLOADED", bytes=count)
            except Exception:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(fetch, row) for row in manifest["files"]]
            for job in concurrent.futures.as_completed(jobs):
                result = job.result()
                next(r for r in manifest["files"] if r["path"] == result["path"]).update(result)
                manifest["elapsed_s"] = time.monotonic()-start
                save()
                print(result, flush=True)
        manifest["status"] = "COMPLETE"
    except Exception:
        manifest["status"] = "FAILED"
        raise
    finally:
        manifest["elapsed_s"] = time.monotonic()-start
        save()


if __name__ == "__main__":
    main()
