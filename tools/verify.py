#!/usr/bin/env python3
"""Run the small, repository-local verification gate.

The gate is deliberately about contracts that are useful on every research
branch: the three current context documents, structured workflow state,
experiment-card identity/schema, and navigable Markdown.  Historical cards
remain readable without being silently rewritten; cards that opt into the v2
schema are checked strictly.

The feature-branch diff is compared with ``main`` (or ``origin/main``) and
includes deleted paths.  Deletions matter because an otherwise unchanged
document can start pointing at a dead target when a file is removed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

CHANGE_FILTER = "ACMRD"
STABLE_REFS = ("origin/main", "main")
REQUIRED_CONTEXT_FILES = (
    "docs/MISSION.md",
    "docs/CAMPAIGN.md",
    "docs/STATE.md",
)
PROBE_DIR = Path("docs/experiments/probes")
VALIDATION_DIR = Path("docs/experiments/validations")
QUEUE_PATH = Path("docs/RESEARCH_QUEUE.yaml")
SEED_LEDGER_PATH = Path("docs/SEED_LEDGER.yaml")

MARKDOWN_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|([^\s)\n]+))"
)
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$")
PROBE_ID_RE = re.compile(r"^(?:P|PROBE)-[A-Za-z0-9][A-Za-z0-9._-]*$")
VALIDATION_ID_RE = re.compile(r"^VAL-[A-Za-z0-9][A-Za-z0-9._-]*$")

CLAIM_STATUSES = {"OPEN", "PARTIAL", "SUPPORTED", "REFUTED", "KILLED"}
HYPOTHESIS_STATUSES = {
    "OPEN",
    "ACTIVE",
    "PAUSED",
    "PROMISING",
    "KILLED",
    "REFUTED",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _nul_paths(value: str) -> set[str]:
    return {path for path in value.split("\0") if path}


def _stable_branch() -> str:
    """Return the local stable ref used for feature-branch diffs."""

    for candidate in STABLE_REFS:
        if subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            cwd=ROOT,
            capture_output=True,
        ).returncode == 0:
            return candidate
    raise RuntimeError("无法找到稳定集成分支 main 或 origin/main")


def _diff_paths(*args: str) -> set[str]:
    return _nul_paths(
        _git("diff", "--name-only", "-z", f"--diff-filter={CHANGE_FILTER}", *args)
    )


def _changed_paths(base: str | None = None) -> set[str]:
    """Collect committed feature diff and explicitly staged paths.

    Unstaged files are intentionally not swept into a feature verification:
    users can keep unrelated scratch artifacts in the worktree.  A staged
    deletion is included just like a staged addition.
    """

    branch = _git("branch", "--show-current").strip()
    changed: set[str] = set()
    if base:
        changed.update(_diff_paths(f"{base}...HEAD"))
    elif branch and branch not in {"main", "origin/main"}:
        stable = _stable_branch()
        merge_base = _git("merge-base", "HEAD", stable).strip()
        changed.update(_diff_paths(f"{merge_base}...HEAD"))
    changed.update(
        _nul_paths(
            _git(
                "diff",
                "--cached",
                "--name-only",
                "-z",
                f"--diff-filter={CHANGE_FILTER}",
            )
        )
    )
    return changed


def _repo_path(path: str | Path) -> Path:
    return (ROOT / Path(path)).resolve()


def _relative_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _is_historical(path: str) -> bool:
    return bool({"logs", "archive"}.intersection(Path(path).parts))


def _is_tracked(path: Path) -> bool:
    """Compatibility helper for callers that want to inspect Git tracking."""

    relative = _relative_path(path)
    if relative is None:
        return False
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def _markdown_targets(text: str) -> Iterable[str]:
    for match in MARKDOWN_LINK_RE.finditer(text):
        yield (match.group(1) or match.group(2) or "").strip()


def _resolve_target(target: str, source: Path) -> Path | None:
    decoded = unquote(target).strip()
    if not decoded or decoded.startswith(("#", "//")) or URI_SCHEME_RE.match(decoded):
        return None
    location = decoded.split("#", 1)[0].split("?", 1)[0]
    if not location:
        return None
    candidate = Path(location)
    resolved = candidate if candidate.is_absolute() else (source.parent / candidate)
    try:
        return resolved.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return Path("__outside_repository__")


def _active_markdown_files() -> set[str]:
    files: set[str] = set()
    if not ROOT.is_dir():
        return files
    for candidate in ROOT.rglob("*.md"):
        if ".git" in candidate.parts:
            continue
        relative = _relative_path(candidate)
        if relative and not _is_historical(relative):
            files.add(relative)
    return files


def _workflow_markdown_files() -> set[str]:
    """Return current workflow docs whose links should always be live."""

    candidates = set(REQUIRED_CONTEXT_FILES) | {"AGENTS.md", "docs/README.md"}
    for directory in (ROOT / PROBE_DIR, ROOT / VALIDATION_DIR):
        if directory.is_dir():
            candidates.update(
                (_relative_path(path) or path.as_posix())
                for path in directory.glob("*.md")
                if not path.name.upper().endswith("_TEMPLATE.MD")
            )
    return {
        relative
        for relative in candidates
        if (_repo_path(relative).is_file() and not _is_historical(relative))
    }


def _deleted_paths(paths: Iterable[str]) -> set[str]:
    deleted: set[str] = set()
    for relative in paths:
        candidate = _repo_path(relative)
        if candidate.exists():
            continue
        try:
            deleted.add(Path(relative).as_posix())
        except (TypeError, ValueError):
            continue
    return deleted


def _check_deleted_links(paths: Iterable[str], failures: list[str]) -> None:
    """Report links to paths removed by the current diff.

    Only targets that are part of this diff are considered here.  That keeps
    old, intentionally archived artifact references from turning a new
    deletion into an unrelated repository-wide documentation migration.
    """

    deleted = _deleted_paths(paths)
    if not deleted:
        return
    for relative in sorted(_active_markdown_files()):
        source = _repo_path(relative)
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for target in _markdown_targets(text):
            resolved = _resolve_target(target, source)
            if resolved is None or resolved == Path("__outside_repository__"):
                continue
            if resolved.as_posix() in deleted:
                failures.append(f"{relative}: Markdown 链接目标已删除: {target}")


def _check_markdown(paths: Iterable[str], failures: list[str]) -> None:
    """Check changed Markdown and make deletions fail closed.

    Current workflow docs are checked on every run, and other changed
    Markdown is checked when it is in the diff.  When a file is deleted, all
    active Markdown is searched for a reference to that exact deleted path,
    so an unchanged index cannot hide a dead link.
    """

    scope: set[str] = _workflow_markdown_files()
    for relative in paths:
        candidate = _repo_path(relative)
        if candidate.is_file() and candidate.suffix.lower() == ".md" and not _is_historical(relative):
            scope.add(Path(relative).as_posix())

    for relative in sorted(scope):
        source = _repo_path(relative)
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{relative}: Markdown 读取失败: {exc}")
            continue
        for target in _markdown_targets(text):
            resolved = _resolve_target(target, source)
            if resolved is None:
                continue
            if resolved == Path("__outside_repository__"):
                failures.append(f"{relative}: Markdown 链接越出仓库: {target}")
            elif not _repo_path(resolved).exists():
                failures.append(f"{relative}: Markdown 链接目标不存在: {target}")

    _check_deleted_links(paths, failures)


def _check_python(paths: Iterable[str], failures: list[str]) -> None:
    for relative in sorted(paths):
        if not relative.endswith(".py"):
            continue
        candidate = _repo_path(relative)
        if not candidate.is_file():
            continue
        try:
            compile(candidate.read_text(encoding="utf-8"), str(candidate), "exec")
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append(f"{relative}: Python 编译失败: {exc}")


def _check_structured(paths: Iterable[str], failures: list[str]) -> None:
    yaml_module = None
    for relative in sorted(paths):
        candidate = _repo_path(relative)
        if not candidate.is_file():
            continue
        try:
            if relative.endswith(".json"):
                json.loads(candidate.read_text(encoding="utf-8"))
            elif relative.endswith((".yaml", ".yml")):
                if yaml_module is None:
                    import yaml as yaml_module

                yaml_module.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception as exc:  # Parsing diagnostics are user-facing gate output.
            failures.append(f"{relative}: 结构化文件解析失败: {exc}")


def _read_yaml(path: Path) -> tuple[Any | None, str | None]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # Parsing diagnostics are reported by the gate.
        return None, str(exc)


def _check_required_context(failures: list[str]) -> None:
    for relative in REQUIRED_CONTEXT_FILES:
        path = _repo_path(relative)
        if not path.is_file():
            failures.append(f"{relative}: 当前研究上下文文件不存在")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(f"{relative}: 当前研究上下文不可读: {exc}")
            continue
        if not text.strip():
            failures.append(f"{relative}: 当前研究上下文不能为空")


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
            value = value[1:-1].strip()
    return value


def _card_fields(text: str) -> dict[str, Any]:
    """Read simple key/value headers from both v2 cards and legacy cards."""

    fields: dict[str, Any] = {}
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        try:
            end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
        except StopIteration:
            end = -1
        if end > 0:
            try:
                import yaml

                frontmatter = yaml.safe_load("\n".join(lines[1:end]))
                if isinstance(frontmatter, Mapping):
                    fields.update({str(k): _clean_scalar(v) for k, v in frontmatter.items()})
            except Exception:
                # The structured checker will report malformed YAML when the
                # card is part of the diff; continue so ID diagnostics remain
                # useful too.
                pass
    for line in lines:
        match = KEY_VALUE_RE.match(line)
        if not match:
            continue
        key, raw = match.groups()
        if key not in fields:
            fields[key] = _clean_scalar(raw)
    return fields


def _card_id(kind: str, path: Path, fields: Mapping[str, Any]) -> str | None:
    keys = ("probe_id", "experiment_id", "id") if kind == "probe" else (
        "validation_id",
        "experiment_id",
        "id",
    )
    for key in keys:
        value = fields.get(key)
        if value is not None and str(value).strip():
            return str(_clean_scalar(value)).strip()
    stem = path.stem
    if kind == "probe" and PROBE_ID_RE.fullmatch(stem):
        return stem
    if kind == "validation" and VALIDATION_ID_RE.fullmatch(stem):
        return stem
    return None


def _card_files(kind: str) -> list[Path]:
    directory = ROOT / (PROBE_DIR if kind == "probe" else VALIDATION_DIR)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.md")
        if path.name.upper() not in {"PROBE_TEMPLATE.MD", "VALIDATION_TEMPLATE.MD"}
    )


def _nonempty_field(fields: Mapping[str, Any], name: str) -> bool:
    value = fields.get(name)
    return value is not None and bool(str(value).strip())


def _queue_hypotheses() -> Mapping[str, Mapping[str, Any]]:
    path = _repo_path(QUEUE_PATH)
    if not path.is_file():
        return {}
    data, error = _read_yaml(path)
    if error or not isinstance(data, Mapping):
        return {}
    hypotheses = data.get("hypotheses")
    return hypotheses if isinstance(hypotheses, Mapping) else {}


def _queue_claims() -> Mapping[str, Mapping[str, Any]]:
    path = _repo_path(QUEUE_PATH)
    if not path.is_file():
        return {}
    data, error = _read_yaml(path)
    if error or not isinstance(data, Mapping):
        return {}
    claims = data.get("claims")
    return claims if isinstance(claims, Mapping) else {}


def _check_experiment_cards(failures: list[str]) -> None:
    seen: dict[str, list[str]] = defaultdict(list)
    v2_families: dict[str, list[tuple[str, int]]] = defaultdict(list)
    hypotheses = _queue_hypotheses()
    claims = _queue_claims()

    for kind in ("probe", "validation"):
        expected_schema = f"ref2dex.{kind}.v2"
        id_pattern = PROBE_ID_RE if kind == "probe" else VALIDATION_ID_RE
        for path in _card_files(kind):
            relative = _relative_path(path) or path.as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                failures.append(f"{relative}: 实验卡不可读: {exc}")
                continue
            fields = _card_fields(text)
            identifier = _card_id(kind, path, fields)
            if identifier is None:
                failures.append(f"{relative}: 缺少合法 {kind}_id（可由文件名推断）")
                continue
            identifier = str(identifier)
            seen[identifier].append(relative)
            if not id_pattern.fullmatch(identifier):
                failures.append(f"{relative}: {kind} ID 格式无效: {identifier}")

            schema = fields.get("schema")
            if schema is None or str(schema).strip() == "":
                # Historical cards predate the machine-readable contract.
                # They remain valid evidence and are not rewritten here.
                continue
            if str(schema) != expected_schema:
                failures.append(f"{relative}: schema 应为 {expected_schema}")
                continue
            if path.stem != identifier:
                failures.append(f"{relative}: 文件名必须与 ID 一致: {identifier}")
            explicit_id_key = "probe_id" if kind == "probe" else "validation_id"
            if not _nonempty_field(fields, explicit_id_key):
                failures.append(f"{relative}: v2 实验卡必须显式填写 {explicit_id_key}")

            claim_id = str(fields.get("claim_id", "")).strip()
            if claim_id and claim_id not in claims:
                failures.append(f"{relative}: claim_id 未在 RESEARCH_QUEUE 中登记: {claim_id}")
            family_id = str(fields.get("hypothesis_family", "")).strip()
            family_state = hypotheses.get(family_id)
            if family_id and not isinstance(family_state, Mapping):
                failures.append(
                    f"{relative}: hypothesis_family 未在 RESEARCH_QUEUE 中登记: {family_id}"
                )

            required = (
                (
                    "date",
                    "branch",
                    "git_commit",
                    "claim_id",
                    "hypothesis_family",
                    "decision_changed_if_positive",
                    "decision_changed_if_negative",
                    "probe_index_in_family",
                    "seed_pool",
                )
                if kind == "probe"
                else (
                    "date",
                    "branch",
                    "git_commit",
                    "claim_id",
                    "hypothesis_family",
                    "frozen_method_commit",
                    "development_seed_pool",
                    "validation_seed_pool",
                    "matched_control",
                )
            )
            for field in required:
                if not _nonempty_field(fields, field):
                    failures.append(f"{relative}: v2 实验卡缺少 {field}")

            if kind == "probe":
                raw_index = fields.get("probe_index_in_family")
                try:
                    index = int(raw_index)
                    if index < 1:
                        raise ValueError
                except (TypeError, ValueError):
                    failures.append(f"{relative}: probe_index_in_family 必须是正整数")
                    index = 0
                family = family_id
                if family:
                    v2_families[family].append((relative, index))
                    if isinstance(family_state, Mapping):
                        status = str(family_state.get("status", "")).upper()
                        card_status = str(fields.get("status", "")).upper()
                        # Closing a family must not invalidate its completed cards.
                        # Only new or unfinished probes would consume another slot.
                        if status in {"KILLED", "REFUTED", "PAUSED"} and card_status not in {
                            "PROMISING", "UNPROMISING", "UNCLEAR"
                        }:
                            failures.append(f"{relative}: hypothesis_family {family} 当前不可继续消费（{status}）")
                        budget = family_state.get("probe_budget")
                        if isinstance(budget, int) and not isinstance(budget, bool) and index > budget:
                            failures.append(f"{relative}: probe_index_in_family 超过 {family} 的预算 {budget}")
                        used = family_state.get("probes_used")
                        if isinstance(used, int) and not isinstance(used, bool) and index > used:
                            failures.append(
                                f"{relative}: queue 中 {family}.probes_used={used}，"
                                "先登记并递增队列状态再运行该 Probe"
                            )
                        expected_branch = family_state.get("branch")
                        card_branch = fields.get("branch")
                        if expected_branch and card_branch and str(card_branch) != str(expected_branch):
                            failures.append(
                                f"{relative}: branch {card_branch} 与 {family} 登记的 branch "
                                f"{expected_branch} 不一致"
                            )

            # Validation seed-pool names are checked against the ledger below;
            # keeping their presence in this schema check makes the contract
            # visible even when the ledger itself is malformed.

    for identifier, locations in sorted(seen.items()):
        if len(locations) > 1:
            failures.append(f"实验 ID 重复 {identifier}: {', '.join(sorted(locations))}")

    for family, entries in sorted(v2_families.items()):
        indexes = [index for _, index in entries if index]
        if len(indexes) != len(set(indexes)):
            failures.append(f"hypothesis_family {family}: probe_index_in_family 重复")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_research_queue(failures: list[str]) -> None:
    path = _repo_path(QUEUE_PATH)
    if not path.is_file():
        failures.append(f"{QUEUE_PATH.as_posix()}: 研究队列不存在")
        return
    data, error = _read_yaml(path)
    if error:
        failures.append(f"{QUEUE_PATH.as_posix()}: YAML 无法解析: {error}")
        return
    if not isinstance(data, Mapping) or data.get("schema") != "ref2dex.research_queue.v1":
        failures.append(f"{QUEUE_PATH.as_posix()}: schema 必须为 ref2dex.research_queue.v1")
        return

    claims = data.get("claims")
    if not isinstance(claims, Mapping) or not claims:
        failures.append(f"{QUEUE_PATH.as_posix()}: claims 必须是非空映射")
        claims = {}
    for claim_id, claim in claims.items():
        if not isinstance(claim, Mapping):
            failures.append(f"{QUEUE_PATH.as_posix()}: claim {claim_id} 必须是映射")
            continue
        status = str(claim.get("status", "")).upper()
        if status not in CLAIM_STATUSES:
            failures.append(f"{QUEUE_PATH.as_posix()}: claim {claim_id} status 无效: {status}")
        if not _nonempty_field(claim, "name"):
            failures.append(f"{QUEUE_PATH.as_posix()}: claim {claim_id} 缺少 name")

    hypotheses = data.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or not hypotheses:
        failures.append(f"{QUEUE_PATH.as_posix()}: hypotheses 必须是非空映射")
        return
    for family, hypothesis in hypotheses.items():
        if not isinstance(hypothesis, Mapping):
            failures.append(f"{QUEUE_PATH.as_posix()}: hypothesis {family} 必须是映射")
            continue
        claim = str(hypothesis.get("claim", ""))
        if claim not in claims:
            failures.append(f"{QUEUE_PATH.as_posix()}: {family} 引用了未知 claim {claim}")
        if not _nonempty_field(hypothesis, "name"):
            failures.append(f"{QUEUE_PATH.as_posix()}: {family} 缺少 name")
        status = str(hypothesis.get("status", "")).upper()
        if status not in HYPOTHESIS_STATUSES:
            failures.append(f"{QUEUE_PATH.as_posix()}: {family} status 无效: {status}")
        budget = hypothesis.get("probe_budget")
        used = hypothesis.get("probes_used")
        if not _is_int(budget) or budget < 1:
            failures.append(f"{QUEUE_PATH.as_posix()}: {family}.probe_budget 必须为正整数")
        if not _is_int(used) or used < 0:
            failures.append(f"{QUEUE_PATH.as_posix()}: {family}.probes_used 必须为非负整数")
        if _is_int(budget) and _is_int(used):
            if used > budget:
                failures.append(f"{QUEUE_PATH.as_posix()}: {family} probes_used 超过预算")
            if used >= budget and status in {"ACTIVE", "OPEN"}:
                failures.append(
                    f"{QUEUE_PATH.as_posix()}: {family} 已耗尽预算但仍为 {status}，必须换高层假设或更新状态"
                )
        branch = hypothesis.get("branch")
        if branch is not None and (not isinstance(branch, str) or not branch.startswith("agent/")):
            failures.append(f"{QUEUE_PATH.as_posix()}: {family}.branch 必须以 agent/ 开头")


def _seed_values(node: Any) -> set[int]:
    """Expand the compact seed-ledger representation."""

    values: set[int] = set()
    if isinstance(node, Mapping):
        seeds = node.get("seeds", [])
        if isinstance(seeds, list):
            for seed in seeds:
                if _is_int(seed):
                    values.add(seed)
        ranges = node.get("ranges", [])
        if isinstance(ranges, list):
            for interval in ranges:
                if (
                    isinstance(interval, (list, tuple))
                    and len(interval) == 2
                    and _is_int(interval[0])
                    and _is_int(interval[1])
                    and interval[0] <= interval[1]
                ):
                    values.update(range(interval[0], interval[1] + 1))
        for key in ("development", "holdout", "validation"):
            if key in node:
                values.update(_seed_values(node[key]))
    elif isinstance(node, list):
        for value in node:
            if _is_int(value):
                values.add(value)
    elif _is_int(node):
        values.add(node)
    return values


def _seed_pool(data: Mapping[str, Any], name: str) -> set[int] | None:
    current: Any = data.get("pools", {})
    for part in name.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return _seed_values(current)


def _validate_seed_spec(node: Any, label: str, failures: list[str]) -> None:
    """Reject malformed seed entries instead of silently dropping them."""

    if not isinstance(node, Mapping):
        if not isinstance(node, (list, tuple)) and not _is_int(node):
            failures.append(f"{SEED_LEDGER_PATH.as_posix()}: {label} 必须是 seed 映射")
        return
    seeds = node.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or not all(_is_int(seed) for seed in seeds):
            failures.append(f"{SEED_LEDGER_PATH.as_posix()}: {label}.seeds 必须是整数列表")
    ranges = node.get("ranges")
    if ranges is not None:
        valid_ranges = (
            isinstance(ranges, list)
            and all(
                isinstance(interval, (list, tuple))
                and len(interval) == 2
                and _is_int(interval[0])
                and _is_int(interval[1])
                and interval[0] <= interval[1]
                for interval in ranges
            )
        )
        if not valid_ranges:
            failures.append(f"{SEED_LEDGER_PATH.as_posix()}: {label}.ranges 必须是递增整数区间")
    for key in ("development", "holdout", "validation"):
        if key in node:
            _validate_seed_spec(node[key], f"{label}.{key}", failures)


def _check_seed_ledger(failures: list[str]) -> None:
    path = _repo_path(SEED_LEDGER_PATH)
    if not path.is_file():
        failures.append(f"{SEED_LEDGER_PATH.as_posix()}: seed ledger 不存在")
        return
    data, error = _read_yaml(path)
    if error:
        failures.append(f"{SEED_LEDGER_PATH.as_posix()}: YAML 无法解析: {error}")
        return
    if not isinstance(data, Mapping) or data.get("schema") != "ref2dex.seed_ledger.v1":
        failures.append(f"{SEED_LEDGER_PATH.as_posix()}: schema 必须为 ref2dex.seed_ledger.v1")
        return
    pools = data.get("pools")
    if not isinstance(pools, Mapping):
        failures.append(f"{SEED_LEDGER_PATH.as_posix()}: pools 必须是映射")
        return
    required = ("debug", "probe", "validation")
    for name in required:
        if name not in pools:
            failures.append(f"{SEED_LEDGER_PATH.as_posix()}: 缺少 {name} seed pool")
        else:
            _validate_seed_spec(pools[name], name, failures)
    pool_sets: dict[str, set[int]] = {}
    for name in required:
        if name in pools:
            pool_sets[name] = _seed_values(pools[name])
            if not pool_sets[name]:
                failures.append(f"{SEED_LEDGER_PATH.as_posix()}: {name} pool 不能为空")
    names = sorted(pool_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = pool_sets[left] & pool_sets[right]
            if overlap:
                failures.append(
                    f"{SEED_LEDGER_PATH.as_posix()}: seed pool {left}/{right} 重叠: {sorted(overlap)[:8]}"
                )
    validation = pools.get("validation")
    if isinstance(validation, Mapping):
        development = _seed_values(validation.get("development", {}))
        holdout = _seed_values(validation.get("holdout", {}))
        overlap = development & holdout
        if overlap:
            failures.append(
                f"{SEED_LEDGER_PATH.as_posix()}: validation development/holdout 重叠: {sorted(overlap)[:8]}"
            )


def _check_card_seed_pools(failures: list[str]) -> None:
    """Cross-check v2 card pool references after the ledger is parsed."""

    ledger_path = _repo_path(SEED_LEDGER_PATH)
    data, error = _read_yaml(ledger_path) if ledger_path.is_file() else (None, "missing")
    if error or not isinstance(data, Mapping):
        return
    for kind in ("probe", "validation"):
        for path in _card_files(kind):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                relative = _relative_path(path) or path.as_posix()
                failures.append(f"{relative}: 实验卡 seed 字段不可读: {exc}")
                continue
            fields = _card_fields(text)
            schema = str(fields.get("schema", ""))
            if schema != f"ref2dex.{kind}.v2":
                continue
            names = (
                ("seed_pool",)
                if kind == "probe"
                else ("development_seed_pool", "validation_seed_pool")
            )
            relative = _relative_path(path) or path.as_posix()
            for name in names:
                pool_name = str(fields.get(name, "")).strip()
                if pool_name and _seed_pool(data, pool_name) is None:
                    failures.append(f"{relative}: {name} 未在 SEED_LEDGER 中登记: {pool_name}")
                if kind == "probe" and pool_name.startswith("validation"):
                    failures.append(
                        f"{relative}: Probe 不得消费 validation seed pool: {pool_name}"
                    )
                if kind == "validation":
                    if name == "development_seed_pool" and pool_name != "validation.development":
                        failures.append(
                            f"{relative}: development_seed_pool 必须指向 validation.development"
                        )
                    if name == "validation_seed_pool" and pool_name != "validation.holdout":
                        failures.append(
                            f"{relative}: validation_seed_pool 必须指向 validation.holdout"
                        )


def _select_tests(paths: Iterable[str]) -> list[str]:
    selected: set[str] = set()
    needs_governance = False
    needs_shared = False
    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            selected.add(path)
        if path.startswith("src/base/"):
            needs_shared = True
        if path == "AGENTS.md" or path.startswith(
            ("docs/", ".agents/", ".github/")
        ) or path == "tools/verify.py":
            needs_governance = True
    if needs_shared:
        candidate = "tests/test_run_manifest.py"
        if (ROOT / candidate).is_file():
            selected.add(candidate)
    if needs_governance:
        # The retired work-version migration test is deliberately not part of
        # current workflow verification.
        candidate = "tests/governance/test_verify.py"
        if (ROOT / candidate).is_file():
            selected.add(candidate)
    return sorted(selected)


def _run_tests(paths: Sequence[str], failures: list[str]) -> None:
    if not paths:
        return
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        failures.append(f"pytest 失败，退出码 {result.returncode}: {(result.stdout + result.stderr)[-4000:]}")


def verify_changed(paths: set[str]) -> int:
    failures: list[str] = []
    _check_python(paths, failures)
    _check_structured(paths, failures)
    _check_required_context(failures)
    _check_research_queue(failures)
    _check_seed_ledger(failures)
    _check_experiment_cards(failures)
    _check_card_seed_pools(failures)
    _check_markdown(paths, failures)
    if not failures:
        _run_tests(_select_tests(paths), failures)
    if failures:
        print("VERIFY FAIL")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("VERIFY PASS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed", action="store_true", required=True)
    parser.parse_args(argv)
    base = os.environ.get("VERIFY_BASE", "").strip() or None
    try:
        paths = _changed_paths(base)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"VERIFY FAIL: 无法读取 Git 变更: {exc}")
        return 1
    print("VERIFY PATHS:", *sorted(paths), sep="\n  ")
    return verify_changed(paths)


if __name__ == "__main__":
    raise SystemExit(main())
