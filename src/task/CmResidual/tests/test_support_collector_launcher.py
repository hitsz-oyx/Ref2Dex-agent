"""CPU/static tests for the canonical support collector launcher."""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pytest

import scripts.run_six_expert_support_collection as launcher
from scripts.run_six_expert_support_collection import (
    SPLITS,
    build_command,
    check_gpu_for_execute,
    execute_environment,
    runtime_cwd,
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


def test_launcher_uses_absolute_real_dexplore_config_paths(tmp_path):
    command = build_command(split="fit", output_root=tmp_path / "fit")
    cfg_env = Path(command[command.index("--cfg_env") + 1])
    cfg_train = Path(command[command.index("--cfg_train") + 1])
    assert cfg_env.is_absolute() and cfg_env.is_file()
    assert cfg_train.is_absolute() and cfg_train.is_file()
    assert os.path.join(str(Path.cwd()), str(cfg_env)) == str(cfg_env)
    assert os.path.join(str(Path.cwd()), str(cfg_train)) == str(cfg_train)
    assert command[command.index("--cfg_env") + 1] != \
        "dexplore/data/cfg/inspire_object_balanced.yaml"
    assert command[command.index("--cfg_train") + 1] != \
        "dexplore/data/cfg/train/rlg/inspire.yaml"


def test_runtime_cwd_resolves_relative_airplane_asset_root():
    audit = source_preflight()
    dplore_cwd = runtime_cwd()
    airplane_asset = dplore_cwd / launcher.AIRPLANE_ASSET_RELATIVE
    assert audit["runtime_cwd_error"] is None
    assert audit["runtime_cwd"] == str(dplore_cwd)
    assert audit["resolved_airplane_asset"] == str(airplane_asset.resolve())
    assert airplane_asset.is_file()
    assert os.path.realpath(str(dplore_cwd / launcher.AIRPLANE_ASSET_RELATIVE)) == \
        str(airplane_asset.resolve())


def test_execute_subprocess_uses_canonical_dexplore_cwd(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(launcher, "check_gpu_for_execute",
                        lambda index: {"physical_gpu_index": index})

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=Path(cwd), env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_six_expert_support_collection.py", "--split", "fit",
        "--output-root", str(tmp_path / "fresh-output"),
        "--gpu-index", "3", "--execute",
    ])
    assert launcher.main() == 0
    assert captured["cwd"] == runtime_cwd()
    assert (captured["cwd"] / launcher.AIRPLANE_ASSET_RELATIVE).is_file()
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert not (tmp_path / "fresh-output").exists()


def test_cpu_preflight_never_queries_gpu(monkeypatch):
    def fail(_args):
        raise AssertionError("CPU preflight must not query nvidia-smi")

    monkeypatch.setattr(launcher, "_nvidia_smi", fail)
    audit = source_preflight()
    assert audit["status"] == "PASS"
    assert audit["config_paths_missing"] == []
    assert audit["runtime_cwd_error"] is None
    assert all(Path(value).is_absolute() and Path(value).is_file()
               for value in audit["config_paths"].values())


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
