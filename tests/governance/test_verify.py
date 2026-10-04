from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ref2dex_verify", ROOT / "tools/verify.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ledger() -> str:
    return """
schema: ref2dex.seed_ledger.v1
pools:
  debug:
    ranges: [[0, 9]]
  probe:
    ranges: [[10, 19]]
  validation:
    development:
      ranges: [[20, 29]]
    holdout:
      ranges: [[30, 39]]
"""


def _valid_probe(identifier: str = "P-20260925-temporal") -> str:
    return f"""---
schema: ref2dex.probe.v2
probe_id: {identifier}
date: 2026-09-25
branch: agent/cm-temporal
git_commit: abc123
claim_id: C3
hypothesis_family: HF02
decision_changed_if_positive: continue temporal credit
decision_changed_if_negative: change high-level hypothesis
probe_index_in_family: 1
seed_pool: probe
---
# Probe
"""


def _valid_validation() -> str:
    return """---
schema: ref2dex.validation.v2
validation_id: VAL-20260925-temporal
date: 2026-09-25
branch: agent/cm-temporal
git_commit: abc123
claim_id: C3
hypothesis_family: HF02
frozen_method_commit: def456
development_seed_pool: validation.development
validation_seed_pool: validation.holdout
matched_control: identical Cm-off arm
---
# Validation
"""


def test_feature_branch_diff_uses_main_and_includes_deleted_and_staged(monkeypatch) -> None:
    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "agent/workflow-v2.1\n"
        if args == ("merge-base", "HEAD", "origin/main"):
            return "base\n"
        if "--cached" in args:
            return "docs/staged.md\0"
        return "tools/verify.py\0docs/removed.md\0"

    monkeypatch.setattr(VERIFY, "_git", fake_git)
    monkeypatch.setattr(VERIFY, "_stable_branch", lambda: "origin/main")
    assert VERIFY._changed_paths() == {
        "tools/verify.py",
        "docs/removed.md",
        "docs/staged.md",
    }


def test_required_context_documents_are_readable() -> None:
    failures: list[str] = []
    VERIFY._check_required_context(failures)
    assert not failures


def test_required_context_rejects_missing_or_empty_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/MISSION.md", "mission\n")
    _write(tmp_path / "docs/CAMPAIGN.md", "campaign\n")
    _write(tmp_path / "docs/STATE.md", "   \n")
    failures: list[str] = []
    VERIFY._check_required_context(failures)
    assert any("STATE.md" in failure for failure in failures)


def test_markdown_link_check_and_deleted_target_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/index.md", "[gone](gone.md)\n")

    failures: list[str] = []
    VERIFY._check_markdown({"docs/index.md"}, failures)
    assert any("目标不存在" in failure for failure in failures)

    failures = []
    VERIFY._check_deleted_links({"docs/gone.md"}, failures)
    assert any("目标已删除" in failure for failure in failures)


def test_reusable_skill_markdown_is_outside_project_link_checks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    path = ".agents/skills/example/SKILL.md"
    _write(tmp_path / path, "[generic example](./hypothetical-project/file.md)\n")

    failures: list[str] = []
    VERIFY._check_markdown({path}, failures)
    assert not failures
    assert path not in VERIFY._active_markdown_files()


def test_experiment_ids_are_unique_and_legacy_ids_are_inferred(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/experiments/probes/P-20260925-a.md", "# old card\n")
    _write(tmp_path / "docs/experiments/probes/P-20260925-b.md", "probe_id: P-20260925-a\n")

    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    assert any("实验 ID 重复 P-20260925-a" in failure for failure in failures)


@pytest.mark.parametrize("directory", [
    "docs/experiments/probes",
    "docs/experiments/probes/nested",
    "src/task/TaskA/docs/experiments/probes/nested",
])
def test_v2_probe_schema_and_seed_contract(tmp_path: Path, monkeypatch, directory: str) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / directory / "P-20260925-temporal.md",
        _valid_probe(),
    )

    failures: list[str] = []
    VERIFY._check_seed_ledger(failures)
    VERIFY._check_experiment_cards(failures)
    VERIFY._check_card_seed_pools(failures)
    assert not failures
    assert len(VERIFY._card_files("probe")) == 1


@pytest.mark.parametrize("directory", [
    "docs/experiments/validations",
    "src/task/TaskB/docs/experiments/validations/nested",
])
def test_v2_validation_requires_frozen_method_and_seed_pools(tmp_path: Path, monkeypatch, directory: str) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / directory / "VAL-20260925-temporal.md",
        _valid_validation(),
    )
    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    VERIFY._check_card_seed_pools(failures)
    assert not failures
    assert len(VERIFY._card_files("validation")) == 1

    card = tmp_path / directory / "VAL-20260925-temporal.md"
    card.write_text(_valid_validation().replace("frozen_method_commit: def456\n", ""))
    VERIFY._check_experiment_cards(failures)
    assert any("缺少 frozen_method_commit" in failure for failure in failures)


def test_seed_ledger_rejects_pool_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(
        tmp_path / "docs/SEED_LEDGER.yaml",
        _ledger().replace("ranges: [[10, 19]]", "ranges: [[9, 19]]"),
    )
    failures: list[str] = []
    VERIFY._check_seed_ledger(failures)
    assert any("重叠" in failure for failure in failures)


@pytest.mark.parametrize("directory", [
    "docs/experiments/probes",
    "src/task/TaskA/docs/experiments/probes/nested",
])
def test_probe_cannot_reference_validation_holdout(tmp_path: Path, monkeypatch, directory: str) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / directory / "P-20260925-temporal.md",
        _valid_probe().replace("seed_pool: probe", "seed_pool: validation.holdout"),
    )
    failures: list[str] = []
    VERIFY._check_card_seed_pools(failures)
    assert any("不得消费 validation" in failure for failure in failures)


def test_duplicate_ids_across_root_and_tasks_are_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    paths = [
        "docs/experiments/probes/nested/P-shared.md",
        "src/task/TaskA/docs/experiments/probes/P-shared.md",
        "src/task/TaskB/docs/experiments/probes/route/P-shared.md",
    ]
    for path in paths:
        _write(tmp_path / path, "# Legacy card\n")
    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    assert len(failures) == 1
    assert "实验 ID 重复 P-shared" in failures[0]
    assert all(path in failures[0] for path in paths)


def test_recursive_cards_exclude_templates_navigation_and_archives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    card = "src/task/TaskA/docs/experiments/probes/topic/P-active.md"
    _write(tmp_path / card, "# Active\n")
    excluded = [
        "docs/experiments/probes/PROBE_TEMPLATE.md",
        "docs/experiments/probes/archive/P-old.md",
        "docs/archive/experiments/probes/P-old.md",
        "src/task/TaskA/docs/experiments/probes/topic/README.md",
        "src/task/TaskA/docs/experiments/probes/topic/INDEX.md",
        "src/task/TaskA/docs/experiments/probes/topic/PROBE_TEMPLATE.md",
        "src/task/TaskA/docs/experiments/probes/archive/P-old.md",
        "src/task/TaskA/docs/archive/experiments/probes/P-old.md",
        "src/task/TaskA/research/trial/docs/experiments/probes/P-old.md",
    ]
    for path in excluded:
        _write(tmp_path / path, "[not an active card](missing.md)\n")
    assert VERIFY._card_files("probe") == [tmp_path / card]
    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    VERIFY._check_markdown(set(), failures)
    assert not failures


def test_task_card_links_are_checked_even_when_unchanged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    card = "src/task/TaskA/docs/experiments/probes/topic/P-link.md"
    _write(tmp_path / card, "# Probe\n[evidence](result.json)\n")
    failures: list[str] = []
    VERIFY._check_markdown(set(), failures)
    assert any(card in failure and "目标不存在" in failure for failure in failures)
    _write(tmp_path / card.replace("P-link.md", "result.json"), "{}")
    failures = []
    VERIFY._check_markdown(set(), failures)
    assert not failures


def test_task_validation_checks_global_seed_ledger(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / "src/task/TaskA/docs/experiments/validations/VAL-20260925-temporal.md",
        _valid_validation().replace("validation_seed_pool: validation.holdout", "validation_seed_pool: unknown"),
    )
    failures: list[str] = []
    VERIFY._check_card_seed_pools(failures)
    assert any("未在 SEED_LEDGER 中登记" in failure for failure in failures)
    assert any("必须指向 validation.holdout" in failure for failure in failures)


def test_seed_ledger_rejects_malformed_range(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(
        tmp_path / "docs/SEED_LEDGER.yaml",
        _ledger().replace("ranges: [[0, 9]]", "ranges: [[9, 0]]"),
    )
    failures: list[str] = []
    VERIFY._check_seed_ledger(failures)
    assert any("递增整数区间" in failure for failure in failures)


def test_governance_selection_is_narrow() -> None:
    selected = VERIFY._select_tests({"AGENTS.md", "docs/SEED_LEDGER.yaml"})
    assert selected == ["tests/governance/test_verify.py"]


def test_historical_markdown_is_not_a_current_link_scope() -> None:
    assert VERIFY._is_historical("docs/archive/2026-10-04-research-governance/logs/activity_log.md")
    assert VERIFY._is_historical("docs/archive/old.md")
    assert not VERIFY._is_historical("docs/experiments/P-001.md")


def test_changed_test_rename_selects_only_existing_destination(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, 'ROOT', tmp_path)
    destination = 'tests/integration/test_moved.py'
    _write(tmp_path / destination, 'def test_ok(): pass\n')
    assert VERIFY._select_tests({'tests/test_moved.py', destination}) == [destination]
