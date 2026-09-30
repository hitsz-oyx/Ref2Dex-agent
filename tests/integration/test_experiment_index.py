from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
