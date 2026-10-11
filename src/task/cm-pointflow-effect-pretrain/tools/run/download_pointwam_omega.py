"""Bounded, SHA256-verified VGGT-Omega acquisition from a user-selected mirror."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1200:
        parser.error("maximum download wall time is 1200 seconds")
    metadata = json.loads(args.files_manifest.read_text())
    file = next(f for f in metadata["Data"]["Files"] if f["Path"] == "vggt_omega_1b_512.pt")
    expected_bytes, expected_sha = file["Size"], file["Sha256"]
    if not 0 < expected_bytes < 5 * 2**30:
        raise ValueError("unexpected checkpoint size")
    url = "https://modelscope.cn/api/v1/models/facebook/VGGT-Omega/repo?" + urlencode(
        {"Revision": file["Revision"], "FilePath": file["Path"]})
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / file["Path"]
    partial = target.with_suffix(".pt.part")
    manifest_path = args.output / "omega-download.json"
    if target.exists() or partial.exists() or manifest_path.exists():
        raise FileExistsError("refuse to overwrite a checkpoint or earlier acquisition")
    start = time.monotonic();digest = hashlib.sha256();size = 0
    manifest = dict(status="RUNNING", model="VGGT-Omega-1B-512", url=url,
                    expected_bytes=expected_bytes, expected_sha256=expected_sha,
                    revision=file["Revision"], seconds_cap=args.seconds,
                    source_note="user-selected ModelScope mirror; HF metadata masks the gated LFS identity")

    def record():
        manifest.update(bytes=size, elapsed_s=time.monotonic()-start)
        temp = manifest_path.with_suffix(".json.part")
        temp.write_text(json.dumps(manifest, indent=2)+"\n");temp.replace(manifest_path)

    try:
        session = requests.Session();session.trust_env = False
        session.proxies = {"http": args.proxy, "https": args.proxy}
        with session.get(url, timeout=(15, 60), stream=True) as response:
            response.raise_for_status()
            if int(response.headers.get("Content-Length", -1)) != expected_bytes:
                raise ValueError("mirror response size differs from pinned manifest")
            previous = 0
            with partial.open("xb") as handle:
                for chunk in response.iter_content(1 << 20):
                    if time.monotonic()-start > args.seconds:
                        raise TimeoutError("download wall-time cap")
                    size += len(chunk)
                    if size > expected_bytes:
                        raise ValueError("download byte cap exceeded")
                    handle.write(chunk);digest.update(chunk)
                    if size-previous >= 128*2**20:
                        record();elapsed = time.monotonic()-start
                        print(json.dumps(dict(bytes=size, total=expected_bytes,
                              mib_s=size/2**20/elapsed, eta_s=elapsed*(expected_bytes-size)/size)), flush=True)
                        previous = size
        actual = digest.hexdigest()
        if size != expected_bytes or actual != expected_sha:
            raise ValueError("checkpoint byte/SHA256 verification failed")
        partial.replace(target)
        manifest.update(status="COMPLETED", sha256=actual, sha256_verified=True)
    except Exception as exc:
        manifest.update(status="FAILED", error=repr(exc))
        raise
    finally:
        record();print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
