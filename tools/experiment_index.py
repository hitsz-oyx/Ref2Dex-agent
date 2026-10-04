#!/usr/bin/env python3
"""Generate a route-grouped reading index; original experiment cards remain authoritative."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[1]
IGNORED_CARD_NAMES = {
    "README.MD",
    "INDEX.MD",
    "PROBE_TEMPLATE.MD",
    "VALIDATION_TEMPLATE.MD",
}


def _card_paths(directory: Path, kind: str) -> list[Path]:
    """Find active cards below a root and its Task-local experiment trees.

    A caller may pass an arbitrary standalone directory for a small index.  In
    that case discovery stays local to that directory.  The repository-wide
    ``docs/experiments`` index additionally includes each
    ``src/task/<Task>/docs/experiments`` tree.
    """

    roots = [directory / kind]
    if directory.name == "experiments" and directory.parent.name == "docs":
        repo_root = directory.parent.parent
        task_root = repo_root / "src" / "task"
        if task_root.is_dir():
            roots.extend(
                task / "docs" / "experiments" / kind
                for task in sorted(task_root.iterdir())
                if task.is_dir()
            )

    cards: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        cards.extend(
            path
            for path in root.rglob("*.md")
            if path.is_file()
            and path.name.upper() not in IGNORED_CARD_NAMES
            and "archive" not in path.relative_to(root).parts
        )
    return sorted(set(cards))


def _relative_link(path: Path, directory: Path) -> str:
    """Return a portable Markdown link from ``directory`` to ``path``."""

    return Path(os.path.relpath(path, directory)).as_posix()


def render(directory: Path) -> str:
    groups = defaultdict(list)
    for kind in ('probes', 'validations'):
        for path in _card_paths(directory, kind):
            text = path.read_text()
            metadata = {}
            if text.startswith('---\n'):
                parts = text.split('---', 2)
                if len(parts) == 3:
                    value = yaml.safe_load(parts[1])
                    if isinstance(value, dict):
                        metadata = value
            title = next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith('# ')), path.stem)
            identifier = metadata.get('probe_id', metadata.get('validation_id', path.stem))
            family = str(metadata.get('hypothesis_family', '未分类历史记录'))
            state = str(metadata.get('status', '见原卡'))
            result = next((line.split(':', 1)[1].strip() for line in text.splitlines() if line.startswith('Result:')), '见原卡')
            next_step = next((line.split(':', 1)[1].strip() for line in text.splitlines() if line.startswith('Decision:')), '见原卡')
            groups[family].append((_relative_link(path, directory), str(identifier), title, state, kind, result, next_step))
    lines = ['# 实验阅读索引', '', '由 tools/experiment_index.py 生成；状态照录原卡，不重判科研结论。',
             '问题、结果与下一步见原卡开头；详细配置和执行版本见其 manifest。', '']
    for family, cards in sorted(groups.items()):
        lines.extend(['## ' + family, '', '| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |', '| --- | --- | --- | --- | --- |'])
        for relative, identifier, title, state, kind, result, next_step in cards:
            clean = lambda value: value.replace('|', '\\|').replace('\n', ' ')
            lines.append('| [' + clean(identifier) + '](' + quote(relative) + ') | ' + clean(title) + ' | ' + clean(result[:140]) + ' | ' + clean(next_step[:140]) + ' | ' + kind + '/' + clean(state) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'docs/experiments')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = render(args.directory)
    output = args.directory / 'INDEX.md'
    if args.check:
        if not output.is_file() or output.read_text() != content:
            print('Experiment index is stale; run tools/experiment_index.py')
            return 1
    else:
        output.write_text(content)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
