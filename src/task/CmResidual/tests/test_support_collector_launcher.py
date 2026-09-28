"""CPU/static tests for the canonical support collector launcher."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.run_six_expert_support_collection as launcher
from scripts.run_six_expert_support_collection import (
    SPLITS,
    build_command,
    check_gpu_for_execute,
    execute_environment,
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


def test_cpu_preflight_never_queries_gpu(monkeypatch):
    def fail(_args):
        raise AssertionError("CPU preflight must not query nvidia-smi")

    monkeypatch.setattr(launcher, "_nvidia_smi", fail)
    assert source_preflight()["status"] == "PASS"


def test_execute_requires_explicit_gpu_index_without_gpu_probe(monkeypatch, tmp_path):
    def fail(_gpu_index):
        raise AssertionError("missing --gpu-index must fail before GPU probing")

    monkeypatch.setattr(launcher, "check_gpu_for_execute", fail)
    monkeypatch.setattr(sys, "argv", [
        "run_six_expert_support_collection.py", "--split", "fit",
        "--output-root", str(tmp_path / "out"), "--execute",
    ])
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_gpu_probe_rejects_occupied_memory(monkeypatch):
    monkeypatch.setattr(
        launcher, "_nvidia_smi",
        lambda _args: subprocess.CompletedProcess(
            [], 0, "2, 513, 16384, GPU-test\n", ""))
    with pytest.raises(RuntimeError, match="occupied"):
        check_gpu_for_execute(2)


def test_gpu_probe_rejects_compute_owner(monkeypatch):
    responses = iter([
        subprocess.CompletedProcess([], 0, "2, 128, 16384, GPU-test\n", ""),
        subprocess.CompletedProcess([], 0, "4321, python, 128\n", ""),
    ])
    monkeypatch.setattr(launcher, "_nvidia_smi", lambda _args: next(responses))
    with pytest.raises(RuntimeError, match="compute owners"):
        check_gpu_for_execute(2)


def test_execute_environment_sets_physical_gpu_visibility(monkeypatch):
    monkeypatch.setattr(
        launcher, "check_gpu_for_execute",
        lambda index: {"physical_gpu_index": index, "compute_owners": []})
    env, evidence = execute_environment(2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["REF2DEX_MAIN_ROOT"] == str(launcher.MAIN_ROOT)
    assert evidence["physical_gpu_index"] == 2
