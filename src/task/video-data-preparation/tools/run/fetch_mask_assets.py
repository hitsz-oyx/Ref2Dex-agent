#!/usr/bin/env python3
"""Fetch explicit small mask assets into an owned scope, without env mutation."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume only this scope's pinned asset files")
    args = parser.parse_args()
    root = args.output.resolve()
    if root == Path("/tmp") or Path("/tmp") in root.parents:
        raise ValueError("owned project output required")
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+1200
    total, rows = 0, []

    def download(urls, path, cap):
        nonlocal total
        if path.exists():
            if not args.resume or not 0 < path.stat().st_size <= cap:
                raise FileExistsError(path)
            rows.append(dict(path=str(path), bytes=path.stat().st_size, source="owned-pinned-resume"))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        part = path.with_suffix(path.suffix+".partial")
        # A prior domestic mirror attempt can be resumed from the same pinned
        # upstream revision. Do not request a mutable model revision.
        if args.resume and part.exists() and len(urls) > 1:
            urls = list(reversed(urls))
        for url in urls:
            try:
                offset = part.stat().st_size if args.resume and part.exists() else 0
                headers = {"Range": "bytes=%d-" % offset} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(15, 20)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith("bytes %d-" % offset):
                            raise ValueError("download resume range mismatch")
                    elif offset:
                        offset = 0  # Server supplies the complete immutable asset.
                    n = offset
                    with part.open("ab" if offset else "wb") as out:
                        for chunk in response.iter_content(2**20):
                            n += len(chunk)
                            total += len(chunk)
                            if n > cap or total > 2*2**30 or time.monotonic() > deadline:
                                raise RuntimeError("bounded asset download budget")
                            out.write(chunk)
                expected = {"model.safetensors": 689359096, "sam2.1_hiera_tiny.pt": 156008466}.get(path.name)
                if expected is not None and n != expected:
                    raise ValueError("incomplete pinned binary asset: %d/%d" % (n, expected))
                part.rename(path)
                row = dict(path=str(path), bytes=n, source=url)
                rows.append(row)
                print(json.dumps(row), flush=True)
                return
            except (requests.RequestException, OSError) as error:
                errors.append(type(error).__name__+": "+str(error)[:120])
                if not args.resume:
                    part.unlink(missing_ok=True)
        raise RuntimeError("asset unavailable: "+str(path)+" "+str(errors))

    revision = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "vocab.txt", "model.safetensors"):
        download(["https://"+host+"/IDEA-Research/grounding-dino-tiny/resolve/"+revision+"/"+name
                  for host in ("hf-mirror.com", "huggingface.co")], root/"grounding-dino-tiny"/name, 900*2**20)
    download(["https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
              "https://hf-mirror.com/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt"],
             root/"sam2.1_hiera_tiny.pt", 200*2**20)
    if not (root/"sam2/.git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam2.git",
                        str(root/"sam2")], check=True, timeout=min(120, max(1, deadline-time.monotonic())))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root/"sam2", text=True).strip()
    (root/"asset-manifest.json").write_text(json.dumps(dict(assets=rows, sam2_commit=commit,
        grounding_revision=revision, network_bytes=total), indent=2)+"\n")
    print(json.dumps(dict(status="COMPLETED", sam2_commit=commit, bytes=total)), flush=True)


if __name__ == "__main__":
    main()
