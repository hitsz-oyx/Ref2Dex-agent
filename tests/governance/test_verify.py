from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ref2dex_verify", ROOT / "tools/verify.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _queue() -> str:
    return """
schema: ref2dex.research_queue.v1
claims:
  C3:
    name: cm_policy_utility
    status: OPEN
hypotheses:
  HF02:
    claim: C3
    name: temporal_cm
    status: ACTIVE
    probe_budget: 3
    probes_used: 1
"""


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


def test_experiment_ids_are_unique_and_legacy_ids_are_inferred(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/experiments/probes/P-20260925-a.md", "# old card\n")
    _write(tmp_path / "docs/experiments/probes/P-20260925-b.md", "probe_id: P-20260925-a\n")

    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    assert any("实验 ID 重复 P-20260925-a" in failure for failure in failures)


def test_v2_probe_schema_and_family_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/RESEARCH_QUEUE.yaml", _queue())
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / "docs/experiments/probes/P-20260925-temporal.md",
        _valid_probe(),
    )

    failures: list[str] = []
    VERIFY._check_research_queue(failures)
    VERIFY._check_seed_ledger(failures)
    VERIFY._check_experiment_cards(failures)
    VERIFY._check_card_seed_pools(failures)
    assert not failures


def test_closed_family_keeps_completed_probe_but_rejects_new_probe(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/RESEARCH_QUEUE.yaml", _queue().replace("status: ACTIVE", "status: KILLED"))
    card = tmp_path / "docs/experiments/probes/P-20260925-temporal.md"
    _write(card, _valid_probe().replace("seed_pool: probe", "seed_pool: probe\nstatus: UNPROMISING"))
    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    assert not failures

    _write(card, _valid_probe().replace("seed_pool: probe", "seed_pool: probe\nstatus: PLANNED"))
    failures = []
    VERIFY._check_experiment_cards(failures)
    assert any("当前不可继续消费" in failure for failure in failures)


def test_supported_family_is_valid_and_closed_to_unfinished_probes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(
        tmp_path / "docs/RESEARCH_QUEUE.yaml",
        _queue().replace("status: ACTIVE", "status: SUPPORTED"),
    )
    card = tmp_path / "docs/experiments/probes/P-20260925-temporal.md"
    _write(card, _valid_probe().replace("seed_pool: probe", "seed_pool: probe\nstatus: PROMISING"))
    failures: list[str] = []
    VERIFY._check_research_queue(failures)
    VERIFY._check_experiment_cards(failures)
    assert not failures

    _write(card, _valid_probe().replace("seed_pool: probe", "seed_pool: probe\nstatus: PLANNED"))
    failures = []
    VERIFY._check_experiment_cards(failures)
    assert any("当前不可继续消费（SUPPORTED）" in failure for failure in failures)


def test_v2_validation_requires_frozen_method_and_seed_pools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/RESEARCH_QUEUE.yaml", _queue())
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / "docs/experiments/validations/VAL-20260925-temporal.md",
        _valid_validation(),
    )
    failures: list[str] = []
    VERIFY._check_experiment_cards(failures)
    VERIFY._check_card_seed_pools(failures)
    assert not failures


def test_queue_rejects_exhausted_active_family(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/RESEARCH_QUEUE.yaml", _queue().replace("probes_used: 1", "probes_used: 3"))
    failures: list[str] = []
    VERIFY._check_research_queue(failures)
    assert any("已耗尽预算" in failure for failure in failures)


def test_seed_ledger_rejects_pool_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(
        tmp_path / "docs/SEED_LEDGER.yaml",
        _ledger().replace("ranges: [[10, 19]]", "ranges: [[9, 19]]"),
    )
    failures: list[str] = []
    VERIFY._check_seed_ledger(failures)
    assert any("重叠" in failure for failure in failures)


def test_probe_cannot_reference_validation_holdout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, "ROOT", tmp_path)
    _write(tmp_path / "docs/RESEARCH_QUEUE.yaml", _queue())
    _write(tmp_path / "docs/SEED_LEDGER.yaml", _ledger())
    _write(
        tmp_path / "docs/experiments/probes/P-20260925-temporal.md",
        _valid_probe().replace("seed_pool: probe", "seed_pool: validation.holdout"),
    )
    failures: list[str] = []
    VERIFY._check_card_seed_pools(failures)
    assert any("不得消费 validation" in failure for failure in failures)


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
    selected = VERIFY._select_tests({"AGENTS.md", "docs/RESEARCH_QUEUE.yaml"})
    assert selected == ["tests/governance/test_verify.py"]


def test_historical_markdown_is_not_a_current_link_scope() -> None:
    assert VERIFY._is_historical("docs/logs/activity_log.md")
    assert VERIFY._is_historical("docs/archive/old.md")
    assert not VERIFY._is_historical("docs/experiments/P-001.md")


def test_changed_test_rename_selects_only_existing_destination(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(VERIFY, 'ROOT', tmp_path)
    destination = 'tests/integration/test_moved.py'
    _write(tmp_path / destination, 'def test_ok(): pass\n')
    assert VERIFY._select_tests({'tests/test_moved.py', destination}) == [destination]
