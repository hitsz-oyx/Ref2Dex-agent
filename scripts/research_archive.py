#!/usr/bin/env python3
"""Build a traceable research index and archive manifest.

The tool is intentionally additive. It never edits experiment cards, removes Git
refs, or assumes that a branch name is an immutable experiment identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BEGIN = "<!-- BEGIN GENERATED EXPERIMENT INVENTORY -->"
END = "<!-- END GENERATED EXPERIMENT INVENTORY -->"
CARD_DIRS = (Path("docs/experiments/probes"), Path("docs/experiments/validations"))


def _git(root: Path, *args: str, timeout: float = 20.0) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


def _metadata(text: str) -> dict[str, str]:
    """Extract the small common metadata subset without parsing YAML."""

    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(
            r"^\s*(probe_id|validation_id|hypothesis_family|family|branch|git_commit|status|conclusion|implementation_status|replay_status|scientific_status|budget_decision)\s*:\s*(.*?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            fields[match.group(1).lower()] = _clean_value(match.group(2))
        match = re.match(r"^\s*(Family|Status)\s*:\s*(.*?)\s*$", line)
        if match:
            fields[match.group(1).lower()] = _clean_value(match.group(2))
    return fields


def _card_id(path: Path, fields: dict[str, str]) -> str:
    return fields.get("probe_id") or fields.get("validation_id") or path.stem


def _card_record(root: Path, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fields = _metadata(text)
    title = next(
        (line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem
    )
    family = fields.get("family") or fields.get("hypothesis_family") or "UNKNOWN"
    status = fields.get("status") or fields.get("conclusion") or "UNKNOWN"
    relative = path.relative_to(root).as_posix()
    return {
        "id": _card_id(path, fields),
        "kind": "validation" if "validations" in path.parts else "probe",
        "title": title,
        "family": family,
        "status": status,
        "implementation_status": fields.get("implementation_status") or "UNCLASSIFIED",
        "replay_status": fields.get("replay_status") or "UNCLASSIFIED",
        "scientific_status": fields.get("scientific_status") or "UNCLASSIFIED",
        "budget_decision": fields.get("budget_decision") or "UNCLASSIFIED",
        "branch": fields.get("branch"),
        "git_commit": fields.get("git_commit"),
        "path": relative,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def collect_cards(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory in CARD_DIRS:
        absolute = root / directory
        if not absolute.is_dir():
            continue
        for path in sorted(absolute.glob("*.md")):
            if path.name.upper().endswith("_TEMPLATE.MD"):
                continue
            records.append(_card_record(root, path))
    return records


def _escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_inventory(records: Iterable[dict[str, Any]], links: dict[str, str] | None = None) -> str:
    lines = [
        BEGIN,
        "",
        "| ID | Kind | Family | Raw status | Implementation | Replay | Scientific | Budget | Code commit | Card |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {id} | {kind} | {family} | {status} | {implementation} | {replay} | {scientific} | {budget} | {commit} | [{id}]({path}) |".format(
                id=_escape(record["id"]),
                kind=_escape(record["kind"]),
                family=_escape(record["family"]),
                status=_escape(record["status"]),
                implementation=_escape(record.get("implementation_status", "UNCLASSIFIED")),
                replay=_escape(record.get("replay_status", "UNCLASSIFIED")),
                scientific=_escape(record.get("scientific_status", "UNCLASSIFIED")),
                budget=_escape(record.get("budget_decision", "UNCLASSIFIED")),
                commit=_escape(record.get("git_commit") or "UNKNOWN"),
                path=_escape((links or {}).get(record["path"], record["path"])),
            )
        )
    lines.extend(["", END, ""])
    return "\n".join(lines)


def update_index(path: Path, records: list[dict[str, Any]], root: Path | None = None) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else (
        "# Research index\n\n"
        "This is the sole current research entry. The generated inventory below links "
        "to immutable experiment cards; archive summaries define scientific scope.\n\n"
    )
    links: dict[str, str] = {}
    if root is not None:
        start = path.parent.relative_to(root).as_posix()
        for record in records:
            links[record["path"]] = Path(
                os.path.relpath(record["path"], start=start)
            ).as_posix()
    generated = render_inventory(records, links)
    if BEGIN in existing and END in existing:
        prefix = existing.split(BEGIN, 1)[0].rstrip() + "\n\n"
        suffix = existing.split(END, 1)[1].lstrip()
        content = prefix + generated + suffix
    else:
        content = existing.rstrip() + "\n\n" + generated
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _status_lines(root: Path) -> list[str]:
    return _git(root, "status", "--short", "--untracked-files=all").splitlines()


def _parse_worktrees(raw: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in raw.strip().split("\n\n") if raw.strip() else []:
        record: dict[str, Any] = {}
        for line in block.splitlines():
            if " " in line:
                key, value = line.split(" ", 1)
                record[key] = value
            else:
                record[line] = True
        if "worktree" in record:
            result.append(record)
    return result


def _worktree_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for record in _parse_worktrees(_git(root, "worktree", "list", "--porcelain")):
        path = Path(str(record["worktree"]))
        item = {
            "path": str(path),
            "head": record.get("HEAD"),
            "branch": record.get("branch"),
        }
        try:
            status = subprocess.run(
                ["git", "-C", str(path), "status", "--short", "--untracked-files=all"],
                text=True,
                capture_output=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            item["status"] = "unavailable"
            item["status_error"] = str(exc)
        else:
            item["status"] = "dirty" if status.stdout else "clean"
            item["status_paths"] = status.stdout.splitlines()
            if status.returncode:
                item["status_error"] = status.stderr.strip()
        records.append(item)
    return records


def _refs(root: Path, namespace: str) -> list[dict[str, str]]:
    raw = _git(root, "for-each-ref", "--format=%(refname:short)\t%(objectname)", namespace)
    result = []
    for line in raw.splitlines():
        if "\t" in line:
            name, commit = line.split("\t", 1)
            result.append({"name": name, "commit": commit})
    return result


def _dangling_commits(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "fsck", "--no-reflogs", "--unreachable"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return sorted(
        line.split()[2]
        for line in result.stdout.splitlines()
        if len(line.split()) == 3 and line.split()[1] == "commit"
    )


def build_manifest(root: Path, index: Path, archive_dir: Path, cards: list[dict[str, Any]]) -> dict[str, Any]:
    experiment_files = []
    experiment_root = root / "docs/experiments"
    tracked = set(_git(root, "ls-files", "--", "docs/experiments").splitlines())
    for path in sorted(p for p in experiment_root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        experiment_files.append(
            {
                "path": relative,
                "tracked": relative in tracked,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema": "ref2dex.research_archive.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "index": index.relative_to(root).as_posix(),
        "archive_dir": archive_dir.relative_to(root).as_posix(),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "branch": _git(root, "branch", "--show-current").strip() or None,
        "status_porcelain": _status_lines(root),
        "branches": _refs(root, "refs/heads"),
        "remote_tracking_refs": _refs(root, "refs/remotes"),
        "tags": _refs(root, "refs/tags"),
        "dangling_commits": _dangling_commits(root),
        "worktrees": _worktree_inventory(root),
        "cards": cards,
        "experiment_files": experiment_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    index = (root / args.index).resolve() if not args.index.is_absolute() else args.index.resolve()
    archive_dir = (root / args.archive_dir).resolve() if not args.archive_dir.is_absolute() else args.archive_dir.resolve()
    cards = collect_cards(root)
    update_index(index, cards, root)
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root, index, archive_dir, cards)
    old_manifest = archive_dir / "manifest.json"
    if old_manifest.exists():
        try:
            previous = json.loads(old_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        for key in ("validation", "notes"):
            if key in previous:
                manifest[key] = previous[key]
    (archive_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cards": len(cards), "experiment_files": len(manifest["experiment_files"]), "manifest": str(archive_dir / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
