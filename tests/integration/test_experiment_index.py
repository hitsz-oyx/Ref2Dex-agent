from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

CLI = Path(__file__).resolve().parents[2] / 'tools/experiment_index.py'


def test_index_groups_routes_links_original_evidence_and_reports_actual_decision(tmp_path: Path) -> None:
    probes = tmp_path / 'probes'
    probes.mkdir()
    card = probes / 'P-example.md'
    text = '---\nprobe_id: P-example\nhypothesis_family: HF-test\nstatus: COMPLETED\n---\n# Is the route useful?\n\nResult: UNPROMISING; no gain\nDecision: close this route\n'
    card.write_text(text)
    subprocess.run([sys.executable, str(CLI), '--directory', str(tmp_path)], check=True)
    index = (tmp_path / 'INDEX.md').read_text()
    assert 'HF-test' in index and '[P-example](probes/P-example.md)' in index
    assert 'UNPROMISING; no gain' in index and 'close this route' in index
    assert card.read_text() == text
    assert subprocess.run([sys.executable, str(CLI), '--directory', str(tmp_path), '--check']).returncode == 0
    card.write_text(text.replace('close this route', 'revisit after new evidence'))
    assert subprocess.run([sys.executable, str(CLI), '--directory', str(tmp_path), '--check']).returncode == 1


@pytest.mark.parametrize('module_mode', [False, True])
def test_recursive_global_index_links_root_and_task_cards(tmp_path: Path, module_mode: bool) -> None:
    directory = tmp_path / 'docs/experiments'
    cards = {
        'docs/experiments/probes/group/P-root.md': '# Root probe\n',
        'src/task/TaskA/docs/experiments/probes/topic/P-local.md': (
            '---\nexperiment_id: P-local\nhypothesis_family: HF-local\n---\n'
            '# Local probe\nResult: UNCLEAR\nDecision: collect more data\n'
        ),
        'src/task/Task B/docs/experiments/validations/VAL-local.md': '# Local validation\n',
    }
    ignored = [
        'docs/experiments/probes/group/PROBE_TEMPLATE.md',
        'docs/experiments/probes/group/README.md',
        'docs/experiments/probes/archive/P-ignored.md',
        'src/task/TaskA/docs/experiments/probes/topic/INDEX.md',
        'src/task/TaskA/docs/experiments/probes/topic/PROBE_TEMPLATE.md',
        'src/task/TaskA/docs/experiments/probes/archive/P-ignored.md',
        'src/task/TaskA/docs/archive/experiments/probes/P-ignored.md',
        'src/task/TaskA/research/trial/docs/experiments/probes/P-ignored.md',
    ]
    for relative, content in {**cards, **dict.fromkeys(ignored, '# IGNORE ME\n')}.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    invocation = ['-m', 'tools.experiment_index'] if module_mode else [str(CLI)]
    command = [sys.executable, *invocation, '--directory', str(directory)]
    subprocess.run(command, cwd=CLI.parents[1], check=True)
    index_path = directory / 'INDEX.md'
    index = index_path.read_text()
    expected_links = [
        'probes/group/P-root.md',
        '../../src/task/TaskA/docs/experiments/probes/topic/P-local.md',
        '../../src/task/Task%20B/docs/experiments/validations/VAL-local.md',
    ]
    for link in expected_links:
        assert f']({link})' in index
        assert (index_path.parent / unquote(link)).is_file()
    assert 'HF-local' in index and 'collect more data' in index
    assert 'IGNORE ME' not in index
    assert all((tmp_path / path).read_text() == content for path, content in cards.items())
    assert subprocess.run(command + ['--check'], cwd=CLI.parents[1]).returncode == 0

    local_card = tmp_path / 'src/task/TaskA/docs/experiments/probes/topic/P-local.md'
    local_card.write_text(cards[str(local_card.relative_to(tmp_path))].replace('UNCLEAR', 'PROMISING'))
    assert subprocess.run(command + ['--check'], cwd=CLI.parents[1]).returncode == 1
    subprocess.run(command, cwd=CLI.parents[1], check=True)
    local_card.unlink()
    assert subprocess.run(command + ['--check'], cwd=CLI.parents[1]).returncode == 1


def test_custom_directory_stays_scoped_and_recurses(tmp_path: Path) -> None:
    directory = tmp_path / 'standalone'
    card = directory / 'probes/route/P-standalone.md'
    card.parent.mkdir(parents=True)
    card.write_text('# Standalone probe\n')
    sibling = tmp_path / 'src/task/TaskA/docs/experiments/probes/P-unrelated.md'
    sibling.parent.mkdir(parents=True)
    sibling.write_text('# Unrelated\n')
    subprocess.run([sys.executable, str(CLI), '--directory', str(directory)], check=True)
    index = (directory / 'INDEX.md').read_text()
    assert '[P-standalone](probes/route/P-standalone.md)' in index
    assert 'P-unrelated' not in index
