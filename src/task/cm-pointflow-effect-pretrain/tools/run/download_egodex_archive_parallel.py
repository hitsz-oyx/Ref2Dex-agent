#!/usr/bin/env python3
"""Resumable direct-range acquisition of a pinned EgoDex ZIP archive."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
import zipfile


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def direct_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items()
            if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}


def head_identity(url: str) -> tuple[int, str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, method="HEAD")
    with opener.open(request, timeout=60) as response:
        size = int(response.headers.get("Content-Length", "-1"))
        etag = response.headers.get("ETag", "")
    if size <= 0 or not etag:
        raise ValueError("archive HEAD lacks Content-Length or ETag")
    return size, etag


def parse_headers(path: Path) -> dict[str, str | int]:
    blocks = path.read_text(errors="replace").replace("\r\n", "\n").split("\n\n")
    for block in reversed(blocks):
        lines = [line for line in block.splitlines() if line]
        if not lines or not lines[0].startswith("HTTP/"):
            continue
        status = int(lines[0].split()[1])
        values: dict[str, str | int] = {"status": status}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.lower().strip()] = value.strip()
        return values
    raise ValueError("curl response did not contain an HTTP status")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_directory_audit(archive: Path, inventory_path: Path) -> dict:
    inventory = json.loads(inventory_path.read_text())
    expected = {row["name"]: row for row in inventory["files"]}
    with zipfile.ZipFile(archive) as source:
        infos = {info.filename: info for info in source.infolist()}
        if len(infos) != len(expected) or set(infos) != set(expected):
            raise ValueError("ZIP central directory differs from pinned inventory names")
        rows = []
        for name in sorted(expected):
            expected_row = expected[name]
            info = infos[name]
            actual = dict(name=name, bytes=info.file_size,
                          compressed_bytes=info.compress_size, crc32=info.CRC)
            if actual != expected_row:
                raise ValueError(f"ZIP member identity differs: {name}")
            rows.append(actual)
    directory_sha = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return dict(zip_entries=len(rows), zip_directory_sha256=directory_sha,
                zip_directory_verified=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filename", default="part2.zip")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-etag", required=True)
    parser.add_argument("--seconds", type=int, default=172800)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-bytes", type=int, default=64 * 2**20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 172800 or not 1 <= args.workers <= 32:
        parser.error("seconds must be 1..172800 and workers must be 1..32")
    if not 16 * 2**20 <= args.chunk_bytes <= 256 * 2**20:
        parser.error("chunk-bytes must be between16MiB and256MiB")
    if args.expected_bytes <= 0:
        parser.error("expected-bytes must be positive")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download-manifest.json"
    target = output / (args.filename + ".part")
    final = output / args.filename
    chunks_dir = output / "chunks"
    if final.exists():
        raise FileExistsError(f"refuse to overwrite published archive: {final}")
    prior_elapsed = 0.0
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("manifest exists; pass --resume for the same acquisition")
        manifest = json.loads(manifest_path.read_text())
        identity = (manifest.get("url"), manifest.get("expected_bytes"),
                    manifest.get("expected_etag"), manifest.get("filename"),
                    manifest.get("chunk_bytes"))
        expected_identity = (args.url, args.expected_bytes, args.expected_etag,
                             args.filename, args.chunk_bytes)
        if identity != expected_identity:
            raise ValueError("resume parameters differ from the frozen archive identity")
        prior_elapsed = float(manifest.get("elapsed_s", 0.0))
    else:
        if target.exists() or chunks_dir.exists():
            raise FileExistsError("partial output exists without a manifest")
        manifest = dict(status="RUNNING", url=args.url, filename=args.filename,
                        expected_bytes=args.expected_bytes, expected_etag=args.expected_etag,
                        inventory=str(args.inventory.resolve()), seconds_cap=args.seconds,
                        workers=args.workers, chunk_bytes=args.chunk_bytes,
                        chunks=[], completed_bytes=0, chunk_count=0)
    if prior_elapsed >= args.seconds:
        raise TimeoutError("archive wall-time cap already exhausted")

    actual_size, actual_etag = head_identity(args.url)
    if (actual_size, actual_etag) != (args.expected_bytes, args.expected_etag):
        raise ValueError("archive HEAD differs from the pinned size/ETag")
    ranges = [(start, min(start + args.chunk_bytes, args.expected_bytes) - 1)
              for start in range(0, args.expected_bytes, args.chunk_bytes)]
    manifest["chunk_count"] = len(ranges)
    rows = {int(row["index"]): row for row in manifest.get("chunks", [])
            if row.get("status") == "COMPLETED"}
    if any(index < 0 or index >= len(ranges) for index in rows):
        raise ValueError("manifest contains an invalid completed range")

    chunks_dir.mkdir()
    if not target.exists():
        with target.open("wb") as handle:
            handle.truncate(args.expected_bytes)
    elif target.stat().st_size != args.expected_bytes:
        raise ValueError("partial archive size differs from the pinned size")

    manifest.update(status="RUNNING", resumed=bool(rows), pid=os.getpid())
    run_started = time.monotonic()
    deadline = run_started + (args.seconds - prior_elapsed)
    lock = threading.Lock()
    stop = threading.Event()

    def save() -> None:
        completed = sorted(rows.values(), key=lambda row: row["index"])
        manifest.update(chunks=completed,
                        completed_bytes=sum(row["bytes"] for row in completed),
                        elapsed_s=prior_elapsed + time.monotonic() - run_started)
        write_json(manifest_path, manifest)

    def stop_handler(signum, _frame) -> None:
        stop.set()
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    save()

    def fetch(index_and_range: tuple[int, tuple[int, int]]) -> None:
        index, (lo, hi) = index_and_range
        if index in rows:
            return
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("archive wall-time cap reached")
        temporary = chunks_dir / f"{index:05d}.part"
        header_path = chunks_dir / f"{index:05d}.headers"
        temporary.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
        remaining = max(60, int(deadline - time.monotonic()))
        command = ["curl", "--location", "--fail", "--silent", "--show-error",
                   "--retry", "5", "--retry-delay", "5", "--connect-timeout", "30",
                   "--max-time", str(remaining), "--range", f"{lo}-{hi}",
                   "--dump-header", str(header_path), "--output", str(temporary), args.url]
        result = subprocess.run(command, env=direct_environment(),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, timeout=remaining + 30)
        if result.returncode:
            raise RuntimeError(f"range {index} curl failed: {result.stderr[-500:]}")
        headers = parse_headers(header_path)
        expected_range = f"bytes {lo}-{hi}/{args.expected_bytes}"
        if (headers.get("status") != 206
                or headers.get("content-range") != expected_range
                or headers.get("etag") != args.expected_etag):
            raise ValueError(f"range {index} identity mismatch: {headers}")
        expected_length = hi - lo + 1
        if temporary.stat().st_size != expected_length:
            raise ValueError(f"range {index} length mismatch")
        digest = sha256_file(temporary)
        with target.open("r+b") as destination, temporary.open("rb") as source:
            destination.seek(lo)
            for block in iter(lambda: source.read(1 << 20), b""):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.unlink()
        header_path.unlink(missing_ok=True)
        with lock:
            rows[index] = dict(index=index, start=lo, end=hi, bytes=expected_length,
                               sha256=digest, status="COMPLETED")
            save()
            done = len(rows)
            print(json.dumps(dict(chunk_done=done, chunk_total=len(ranges),
                                  bytes=manifest["completed_bytes"],
                                  mib_s=manifest["completed_bytes"] / 2**20 /
                                  max(manifest["elapsed_s"], .001))), flush=True)

    try:
        pending = [(index, bounds) for index, bounds in enumerate(ranges) if index not in rows]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, item) for item in pending]
            for future in as_completed(futures):
                future.result()
        if len(rows) != len(ranges):
            raise ValueError("not all archive ranges completed")
        if target.stat().st_size != args.expected_bytes:
            raise ValueError("assembled archive size mismatch")
        manifest.update(stage="ZIP_DIRECTORY_AUDIT")
        save()
        manifest.update(zip_directory_audit(target, args.inventory))
        target.replace(final)
        shutil.rmtree(chunks_dir)
        manifest.update(status="COMPLETED", archive=str(final), archive_published=True,
                        elapsed_s=prior_elapsed + time.monotonic() - run_started)
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error),
                        elapsed_s=prior_elapsed + time.monotonic() - run_started)
        raise
    finally:
        if manifest.get("status") != "COMPLETED":
            manifest["stage"] = manifest.get("stage", "RANGE_DOWNLOAD")
        write_json(manifest_path, manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
