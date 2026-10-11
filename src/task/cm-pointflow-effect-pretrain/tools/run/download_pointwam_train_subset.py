"""Acquire a frozen, small official EgoDex TRAIN subset without staging its ZIP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "ref5_data_expansion"))
from download_egodex import RemoteZip, sha


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=1200)
    a = p.parse_args()
    if not 1 <= a.seconds <= 1200:
        p.error("acquisition is capped at20minutes")
    inventory = json.loads(a.inventory.read_text())
    selection = json.loads(a.selection.read_text())
    url = inventory["url"]
    prefix = url.rsplit("/", 1)[-1].removesuffix(".zip")
    if url not in {f"https://ml-site.cdn-apple.com/datasets/egodex/part{i}.zip" for i in range(1, 6)}:
        raise ValueError("only official EgoDex TRAIN archives are accepted")
    if (not 1 <= len(selection) <= 6 or len({r["member"] for r in selection}) != len(selection)
            or any(r["partition"] not in ("fit_candidate", "dev_candidate") for r in selection)):
        raise ValueError("expected at most six distinct, partitioned episodes")
    names = {r["name"]: r for r in inventory["files"]}
    members = []
    for episode in selection:
        name = episode["member"]
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or len(relative.parts) != 3 or relative.parts[0] != prefix
                or not name.endswith(".hdf5")):
            raise ValueError("unsafe or non-TRAIN archive member")
        members.extend((name, name[:-5] + ".mp4"))
    if any(n not in names for n in members):
        raise ValueError("frozen selection is absent from pinned inventory")
    cap = 2**30
    if sum(names[n]["bytes"] for n in members) > cap:
        raise ValueError("raw selected data exceed1GiB cap")
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status="RUNNING", url=url, split="official_train", training_allowed=False,
                  selection=selection, selection_sha256=sha(a.selection),
                  inventory_sha256=sha(a.inventory), max_bytes=cap, seconds_cap=a.seconds,
                  files=[], archive_etag=inventory["archive_etag"], archive_size=inventory["archive_size"])
    remote = None
    session = requests.Session(); session.trust_env = False
    session.proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}

    def save() -> None:
        report.update(elapsed_s=time.monotonic() - start,
                      transferred_bytes=remote.transferred if remote else 0)
        temporary = a.output / "download_manifest.json.part"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(a.output / "download_manifest.json")

    try:
        remote = RemoteZip(url, session, cap, start + a.seconds)
        if (not remote.etag or remote.etag != report["archive_etag"]
                or remote.size != report["archive_size"]):
            raise ValueError("archive identity changed from the pinned inventory")
        save()
        with zipfile.ZipFile(remote) as archive:
            for count, name in enumerate(members, 1):
                info = archive.getinfo(name); expected = names[name]
                if (info.file_size != expected["bytes"] or info.compress_size != expected["compressed_bytes"]
                        or info.CRC != expected["crc32"]):
                    raise ValueError("archive member changed from the pinned inventory")
                dest = a.output / "raw" / name; dest.parent.mkdir(parents=True, exist_ok=True)
                partial = dest.with_suffix(dest.suffix + ".part")
                if dest.exists() or partial.exists():
                    raise FileExistsError(dest)
                written = 0
                with archive.open(name) as source, partial.open("xb") as target:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        written += len(chunk)
                        if written > info.file_size or time.monotonic() > start + a.seconds:
                            raise RuntimeError("member/time cap exceeded")
                        target.write(chunk)
                if written != info.file_size:
                    raise ValueError("truncated member")
                # ZipExtFile has checked CRC at EOF; never publish a partial file.
                partial.rename(dest)
                report["files"].append(dict(member=name, bytes=written, sha256=sha(dest), crc32=info.CRC))
                save()
                print(json.dumps(dict(member_done=count, member_total=len(members), member=name,
                                      transferred_bytes=remote.transferred)), flush=True)
        report["status"] = "COMPLETED"
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc))
        raise
    finally:
        save()
        session.close()


if __name__ == "__main__":
    main()
