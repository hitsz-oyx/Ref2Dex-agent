"""CPU/static tests for the canonical support collector launcher."""
from __future__ import annotations

from pathlib import Path

from scripts.run_six_expert_support_collection import (
    SPLITS,
    build_command,
    source_preflight,
    validate_command,
)


def test_real_parser_and_internal_cm_off_contract_are_cpu_ready():
    audit = source_preflight()
    assert audit["status"] == "PASS"
    assert audit["legacy_cm_flags_in_real_parser"] == []
    assert audit["internal_cm_off"] is True
    assert audit["adapter_before_save"] is True
    assert audit["isaacgym_imported"] is False


def test_launcher_emits_no_legacy_cm_flags_and_exact_fit_holdout_contract(tmp_path):
    for split, expected in SPLITS.items():
        output_root = tmp_path / split
        command = build_command(split=split, output_root=output_root)
        validation = validate_command(command, split=split, output_root=output_root)
        assert validation["status"] == "PASS"
        assert validation["unsupported_cm_flags"] == []
        assert command[command.index("--assignment-seed") + 1] == str(
            expected["assignment_seed"])
        assert command[command.index("--seed") + 1] == str(expected["simulator_seed"])
        assert command[command.index("--num_envs") + 1] == "192"
        assert "--disable-early-termination" in command
        assert all(not item.startswith("--cm-") for item in command)
        motion = Path(command[command.index("--motion_file") + 1])
        assert motion == motion.resolve()
