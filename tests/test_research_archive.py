from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("research_archive", ROOT / "scripts/research_archive.py")
assert SPEC is not None and SPEC.loader is not None
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


def test_card_metadata_and_filename_fallback(tmp_path: Path) -> None:
    card = tmp_path / "docs/experiments/probes/P-20261003-example.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "---\nprobe_id: P-20261003-example\nhypothesis_family: HF33\n"
        "git_commit: abc123\n---\n# Example route\n\nStatus: COMPLETED — `UNPROMISING`\n",
        encoding="utf-8",
    )
    record = ARCHIVE._card_record(tmp_path, card)
    assert record["id"] == "P-20261003-example"
    assert record["family"] == "HF33"
    assert record["status"] == "COMPLETED — `UNPROMISING`"
    assert record["git_commit"] == "abc123"


def test_update_index_preserves_manual_section(tmp_path: Path) -> None:
    index = tmp_path / "INDEX.md"
    index.write_text(
        "# Research index\n\n## Current route\n\nKeep this text.\n\n"
        "<!-- BEGIN GENERATED EXPERIMENT INVENTORY -->\nold\n"
        "<!-- END GENERATED EXPERIMENT INVENTORY -->\n",
        encoding="utf-8",
    )
    record = {
        "id": "P-1",
        "kind": "probe",
        "family": "HF1",
        "status": "UNPROMISING",
        "git_commit": "abc",
        "path": "docs/experiments/probes/P-1.md",
    }
    ARCHIVE.update_index(index, [record], tmp_path)
    text = index.read_text(encoding="utf-8")
    assert "Keep this text." in text
    assert "| P-1 | probe | HF1 | UNPROMISING | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | UNCLASSIFIED | abc |" in text
    assert "docs/experiments/probes/P-1.md" in text
    assert "old" not in text
