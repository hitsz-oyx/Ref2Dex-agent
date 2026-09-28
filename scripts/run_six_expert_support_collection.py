#!/usr/bin/env python3
"""Build and preflight the canonical six-expert collection command.

This launcher is source/CPU-only unless ``--execute`` is explicitly passed by
an separately authorized collection task.  It intentionally does not expose
the legacy ``--cm-*`` flags: temporal collection fixes Cm-off internally in
``evaluate_temporal_expert_option.py`` before installing the routed player.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(os.environ.get(
    "REF2DEX_MAIN_ROOT", "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent")).resolve()
EVALUATOR = ROOT / "third_party/DExplore/dexplore/evaluate_temporal_expert_option.py"
REAL_EVALUATE = ROOT / "third_party/DExplore/dexplore/evaluate.py"
CONFIG_PARSER = ROOT / "third_party/DExplore/dexplore/utils/config.py"
COLLECTOR_CONFIG = ROOT / "src/task/CmResidual/configs/airplane_temporal_expert_probe.json"
ROUTE_CONFIG = ROOT / "src/task/CmResidual/configs/hf02_temporal_canonical_route.json"
MOTION_RELATIVE = Path("outputs/CmResidual/agent_contact_option_airplane_motions")
CHECKPOINT_RELATIVE = Path(
    "outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/"
    "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/"
    "GRAB_00000260.pth")
ROUTER_RELATIVE = Path(
    "outputs/CmResidual/agent_six_expert_router_model_20260925/router.joblib")

SPLITS: Mapping[str, Mapping[str, int]] = {
    "fit": {"simulator_seed": 256, "assignment_seed": 20260928256},
    "holdout": {"simulator_seed": 257, "assignment_seed": 20260928257},
}
LEGACY_CM_FLAGS = frozenset(
    {"--cm-mode", "--cm-candidate-mode", "--cm-contact-gate", "--cm-checkpoint"})
REQUIRED_REAL_PARSER_FLAGS = frozenset(
    {"--task", "--cfg_env", "--cfg_train", "--motion_file", "--checkpoint",
     "--headless", "--num_envs", "--seed", "--disable-early-termination"})
MAX_PREEXISTING_GPU_MEMORY_MIB = 512


def _canonical_main(relative: Path) -> Path:
    path = (MAIN_ROOT / relative).resolve()
    if not path.exists():
        raise FileNotFoundError(f"canonical input is missing: {path}")
    return path


def _literal_add_argument_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and key.value == "name" and
                        isinstance(value, ast.Constant) and
                        isinstance(value.value, str) and value.value.startswith("--")):
                    flags.add(value.value)
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            flags.add(first.value)
    return flags


def source_preflight() -> dict[str, Any]:
    """Audit real parser and evaluator source without importing Isaac Gym."""
    evaluator_source = EVALUATOR.read_text(encoding="utf-8")
    real_source = REAL_EVALUATE.read_text(encoding="utf-8")
    config_source = CONFIG_PARSER.read_text(encoding="utf-8")
    for path, source in ((EVALUATOR, evaluator_source),
                         (REAL_EVALUATE, real_source),
                         (CONFIG_PARSER, config_source)):
        ast.parse(source, filename=str(path))
    parser_flags = _literal_add_argument_flags(REAL_EVALUATE) | \
        _literal_add_argument_flags(CONFIG_PARSER)
    parser_missing = sorted(REQUIRED_REAL_PARSER_FLAGS - parser_flags)
    # sim_device/graphics_device_id are supplied by Isaac Gym's gymutil parser;
    # they are kept as launcher flags but are not imported during this audit.
    internal_cm_off = all(marker in evaluator_source for marker in (
        "routed.CM_MODEL = None", 'routed.CM_MODE = "off"',
        'routed.CM_CANDIDATE_MODE = "experts"',
        'routed.CM_CONTACT_GATE = "instant"'))
    gate = evaluator_source.find("validate_support_payload(")
    save = evaluator_source.find('torch.save(payload, runtime["record_output"])')
    adapter_before_save = gate >= 0 and save > gate
    return {
        "status": "PASS" if (not parser_missing and internal_cm_off and
                              adapter_before_save and "--cm-mode" not in real_source
                              and "--cm-candidate-mode" not in real_source
                              and "--cm-contact-gate" not in real_source) else "FAIL",
        "real_parser": str(REAL_EVALUATE),
        "parser_flags": sorted(parser_flags),
        "missing_required_flags": parser_missing,
        "legacy_cm_flags_in_real_parser": sorted(
            flag for flag in LEGACY_CM_FLAGS if flag in real_source),
        "internal_cm_off": internal_cm_off,
        "adapter_before_save": adapter_before_save,
        "isaacgym_imported": False,
    }


def build_command(*, split: str, output_root: Path,
                  python_executable: str = "/home2/wyy/miniconda3/envs/graspenv/bin/python",
                  main_root: Path = MAIN_ROOT) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"split must be fit or holdout: {split}")
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"collection output root already exists: {output_root}")
    motion = (Path(main_root) / MOTION_RELATIVE).resolve()
    checkpoint = (Path(main_root) / CHECKPOINT_RELATIVE).resolve()
    router = (Path(main_root) / ROUTER_RELATIVE).resolve()
    for name, path in (("motion root", motion), ("base checkpoint", checkpoint),
                       ("C1 router", router)):
        if not path.exists():
            raise FileNotFoundError(f"{name} is missing: {path}")
    spec = SPLITS[split]
    command = [
        str(python_executable), str(EVALUATOR),
        "--collector-config", str(COLLECTOR_CONFIG.resolve()),
        "--route-config", str(ROUTE_CONFIG.resolve()),
        "--split", split,
        "--assignment-seed", str(spec["assignment_seed"]),
        "--record-output", str(output_root / split / "records.pt"),
        "--observation-router-model", str(router),
        "--observation-router-sha256",
        "1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14",
        "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire_object_balanced.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(motion),
        "--checkpoint", str(checkpoint),
        "--disable-early-termination", "--headless",
        "--sim_device", "cuda:0", "--rl_device", "cuda:0",
        "--graphics_device_id", "0", "--num_envs", "192",
        "--seed", str(spec["simulator_seed"]),
        "--output", str(output_root / split / "results.json"),
    ]
    unsupported = sorted(flag for flag in LEGACY_CM_FLAGS if flag in command)
    if unsupported:
        raise AssertionError(f"canonical launcher contains unsupported flags: {unsupported}")
    return command


def _contains_flag(command: Iterable[str], flag: str) -> bool:
    return any(value == flag or value.startswith(flag + "=") for value in command)


def validate_command(command: list[str], *, split: str, output_root: Path) -> dict[str, Any]:
    unsupported = sorted(flag for flag in LEGACY_CM_FLAGS if _contains_flag(command, flag))
    output_root = Path(output_root).resolve()
    output_values = [
        command[index + 1] for index, value in enumerate(command[:-1])
        if value in {"--output", "--record-output"}
    ]
    valid_output_scope = all(str(output_root) in value for value in output_values)
    result = {
        "status": "PASS" if not unsupported and valid_output_scope else "FAIL",
        "unsupported_cm_flags": unsupported,
        "output_scope": valid_output_scope,
        "split": split,
        "assignment_seed": SPLITS[split]["assignment_seed"],
        "simulator_seed": SPLITS[split]["simulator_seed"],
        "num_envs": command[command.index("--num_envs") + 1],
        "adapter_before_save": source_preflight()["adapter_before_save"],
        "cm_off_internal": source_preflight()["internal_cm_off"],
    }
    if unsupported or not valid_output_scope:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def _nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a read-only nvidia-smi query for an explicitly selected GPU.

    This helper is only called from the explicit ``--execute`` path.  Keeping
    it out of ``source_preflight`` makes the default launcher invocation safe
    on CPU-only workers and prevents an accidental CUDA/Isaac Gym probe.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"nvidia-smi unavailable: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"nvidia-smi query failed: {detail or result.returncode}")
    return result


def check_gpu_for_execute(gpu_index: int) -> dict[str, Any]:
    """Fail closed unless the selected physical GPU is free and attributable.

    The query verifies that the requested physical index exists, has bounded
    pre-existing memory, and has no compute process owned by another task.
    No fallback GPU or implicit environment selection is permitted.
    """
    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise RuntimeError("--gpu-index must be a non-negative physical GPU index")
    gpu = _nvidia_smi([
        f"--id={gpu_index}",
        "--query-gpu=index,memory.used,memory.total,uuid",
        "--format=csv,noheader,nounits",
    ])
    rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"physical GPU{gpu_index} ownership query returned {len(rows)} rows")
    fields = [value.strip() for value in rows[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]}")
    try:
        reported_index, memory_used, memory_total = (
            int(fields[0]), int(fields[1]), int(fields[2]))
    except ValueError as exc:
        raise RuntimeError(f"non-numeric nvidia-smi GPU row: {rows[0]}") from exc
    if reported_index != gpu_index:
        raise RuntimeError(
            f"nvidia-smi selected GPU{gpu_index} but reported GPU{reported_index}")
    if memory_used > MAX_PREEXISTING_GPU_MEMORY_MIB:
        raise RuntimeError(
            f"physical GPU{gpu_index} occupied: {memory_used}MiB used "
            f"(limit {MAX_PREEXISTING_GPU_MEMORY_MIB}MiB)")
    apps = _nvidia_smi([
        f"--id={gpu_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    app_rows = [line.strip() for line in apps.stdout.splitlines()
                if line.strip() and "no running processes" not in line.lower()]
    if app_rows:
        raise RuntimeError(
            f"physical GPU{gpu_index} has compute owners: {'; '.join(app_rows)}")
    return {
        "physical_gpu_index": gpu_index,
        "uuid": fields[3],
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "compute_owners": [],
    }


def execute_environment(gpu_index: int) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate ownership and return the child environment for explicit execute."""
    evidence = check_gpu_for_execute(gpu_index)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["REF2DEX_MAIN_ROOT"] = str(MAIN_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS))
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/P-20260928-six-expert-support-collection-r2")
    parser.add_argument("--emit-command", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly run the command; never used by CPU preflight")
    parser.add_argument("--gpu-index", type=int,
                        help="physical GPU index; required for --execute")
    args = parser.parse_args()
    audit = source_preflight()
    if audit["status"] != "PASS":
        print(json.dumps({"status": "NOT_READY", "source_preflight": audit}, indent=2))
        return 2
    if not args.split:
        if args.execute:
            parser.error("--execute requires --split")
        print(json.dumps({"status": "READY_FOR_COLLECTION", "source_preflight": audit}, indent=2))
        return 0
    if args.execute and args.gpu_index is None:
        parser.error("--execute requires an explicit --gpu-index")
    command = build_command(split=args.split, output_root=args.output_root)
    validation = validate_command(command, split=args.split, output_root=args.output_root)
    result = {"status": "READY_FOR_COLLECTION", "source_preflight": audit,
              "command_validation": validation, "argv": command}
    if args.emit_command:
        print(" ".join(command))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.execute:
        env, gpu_evidence = execute_environment(args.gpu_index)
        result["gpu_preflight"] = gpu_evidence
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
