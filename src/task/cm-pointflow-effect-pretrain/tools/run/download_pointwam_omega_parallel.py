#!/usr/bin/env python3
"""Download a pinned VGGT-Omega file through bounded parallel byte ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import requests


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=64 * 2**20)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600 or not 1 <= args.workers <= 16:
        parser.error("seconds must be 1..3600 and workers must be 1..16")
    if not 1 << 20 <= args.chunk_bytes <= 512 * 2**20:
        parser.error("chunk-bytes must be between 1 MiB and 512 MiB")

    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / args.filename
    manifest_path = args.output / "download-manifest.json"
    chunk_dir = args.output / "chunks"
    chunk_dir.mkdir()
    start_time = time.monotonic()
    deadline = start_time + args.seconds
    manifest = dict(status="RUNNING", url=args.url, filename=args.filename,
                    expected_bytes=args.expected_bytes,
                    expected_sha256=args.expected_sha256, seconds_cap=args.seconds,
                    workers=args.workers, chunk_bytes=args.chunk_bytes,
                    chunks=[], bytes=0)
    write_json(manifest_path, manifest)

    session = requests.Session()
    session.trust_env = False
    thread_state = threading.local()

    def worker_session() -> requests.Session:
        """Keep one direct session per worker; requests sessions are not shared."""
        current = getattr(thread_state, "session", None)
        if current is None:
            current = requests.Session()
            current.trust_env = False
            thread_state.session = current
        return current
    lock = threading.Lock()
    try:
        head = session.head(args.url, allow_redirects=True, timeout=(15, 30))
        head.raise_for_status()
        if int(head.headers.get("Content-Length", -1)) != args.expected_bytes:
            raise ValueError("HEAD size differs from pinned expected size")
        etag = head.headers.get("ETag")
        manifest["etag"] = etag
        ranges = [(start, min(start + args.chunk_bytes, args.expected_bytes) - 1)
                  for start in range(0, args.expected_bytes, args.chunk_bytes)]
        manifest["chunk_count"] = len(ranges)
        write_json(manifest_path, manifest)

        def fetch(index_and_range):
            index, (lo, hi) = index_and_range
            if time.monotonic() >= deadline:
                raise TimeoutError("download wall-time cap reached before range")
            path = chunk_dir / f"{index:05d}.part"
            if path.exists():
                raise FileExistsError(path)
            response = worker_session().get(
                args.url, headers={"Range": f"bytes={lo}-{hi}"},
                stream=True, timeout=(20, 120)
            )
            try:
                response.raise_for_status()
                expected_range = f"bytes {lo}-{hi}/{args.expected_bytes}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError(f"range identity mismatch: {response.status_code} "
                                     f"{response.headers.get('Content-Range')} != {expected_range}")
                if etag and response.headers.get("ETag") != etag:
                    raise ValueError("remote ETag changed during parallel download")
                written = 0
                with path.open("xb") as handle:
                    for data in response.iter_content(1 << 20):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("download wall-time cap reached")
                        written += len(data)
                        if written > hi - lo + 1:
                            raise ValueError("range returned too many bytes")
                        handle.write(data)
                if written != hi - lo + 1:
                    raise ValueError(f"truncated range {index}: {written} bytes")
            finally:
                response.close()
            with lock:
                manifest["chunks"].append(dict(index=index, start=lo, end=hi,
                                                bytes=written, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                manifest["bytes"] = sum(row["bytes"] for row in manifest["chunks"])
                write_json(manifest_path, manifest)
            print(json.dumps(dict(chunk_done=index + 1, chunk_total=len(ranges),
                                  bytes=manifest["bytes"])), flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, item) for item in enumerate(ranges)]
            for future in as_completed(futures):
                future.result()

        if len(manifest["chunks"]) != len(ranges):
            raise ValueError("not all ranges completed")
        ordered = sorted(manifest["chunks"], key=lambda row: row["index"])
        digest = hashlib.sha256()
        written = 0
        partial = target.with_suffix(target.suffix + ".part")
        with partial.open("xb") as destination:
            for row in ordered:
                path = chunk_dir / f"{row['index']:05d}.part"
                with path.open("rb") as source:
                    for data in iter(lambda: source.read(1 << 20), b""):
                        destination.write(data)
                        digest.update(data)
                        written += len(data)
        actual = digest.hexdigest()
        if written != args.expected_bytes or actual != args.expected_sha256:
            raise ValueError(f"assembled file identity mismatch: {written} bytes, {actual}")
        partial.replace(target)
        for row in ordered:
            (chunk_dir / f"{row['index']:05d}.part").unlink()
        chunk_dir.rmdir()
        manifest.update(status="COMPLETED", bytes=written, sha256=actual,
                        sha256_verified=True, elapsed_s=time.monotonic() - start_time)
    except Exception as exc:
        manifest.update(status="FAILED", error=repr(exc),
                        elapsed_s=time.monotonic() - start_time)
        raise
    finally:
        session.close()
        write_json(manifest_path, manifest)
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
