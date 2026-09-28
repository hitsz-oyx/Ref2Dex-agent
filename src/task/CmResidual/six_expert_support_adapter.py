"""CPU-only bridge from the six-expert evaluator to the main support contract.

The support validator is owned by the main worktree.  This adapter loads that
validator without importing Isaac Gym or touching collection artifacts.  It
accepts only canonical rows; legacy temporal-option rows are audited and fail
closed when they cannot provide the provenance-complete fields.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from collections.abc import Sequence
from typing import Any, Iterable, Mapping


_LOCAL_ROOT = Path(__file__).resolve().parents[3]
_MAIN_ROOT = Path(os.environ.get("REF2DEX_MAIN_ROOT", "/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent"))
_CONTRACT_RELATIVE = Path("src/task/CmResidual/six_expert_support_contract.py")


def _contract_path() -> Path:
    candidates = (_LOCAL_ROOT / _CONTRACT_RELATIVE,
                  _MAIN_ROOT / _CONTRACT_RELATIVE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "six_expert_support_contract.py is unavailable in the local or main worktree"
    )


def load_main_contract() -> ModuleType:
    """Load the main validator with no simulator, torch, or GPU dependency."""
    # The validator is read-only input from the main worktree.  Do not create
    # a bytecode artifact beside it while the adapter is used for preflight.
    sys.dont_write_bytecode = True
    path = _contract_path()
    spec = importlib.util.spec_from_file_location(
        "ref2dex_main_six_expert_support_contract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load support contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_records(
        records: Iterable[Mapping[str, Any]], *,
        expected_provenance: Mapping[str, Any] | None = None,
        min_rows_per_arm: int = 30) -> dict[str, Any]:
    """Delegate canonical rows to the main validator."""
    contract = load_main_contract()
    return contract.validate_records(
        (adapt_record(record) for record in records),
        expected_provenance=expected_provenance,
        min_rows_per_arm=min_rows_per_arm)


def build_manifest(
        records: Iterable[Mapping[str, Any]], *,
        expected_provenance: Mapping[str, Any] | None = None,
        min_rows_per_arm: int = 30) -> dict[str, Any]:
    """Build the main contract manifest without writing any file."""
    contract = load_main_contract()
    return contract.build_manifest(
        (adapt_record(record) for record in records),
        expected_provenance=expected_provenance,
        min_rows_per_arm=min_rows_per_arm)


def _to_python(value: Any) -> Any:
    """Convert CPU tensors in the evaluator batch without importing torch."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _expected_provenance(
        expected: Mapping[str, Any] | None,
        experts: Sequence[Mapping[str, Any]],
        contract: ModuleType) -> dict[str, Any]:
    """Normalize evaluator provenance to the main validator's row summary."""
    value = dict(expected or {})
    checkpoint_map = value.pop("expert_checkpoint_sha256", None)
    if checkpoint_map is not None:
        if not isinstance(checkpoint_map, Mapping):
            raise contract.ContractError("expert checkpoint provenance must be a mapping")
        value["candidate_experts"] = tuple(
            (entry["name"], checkpoint_map.get(entry["name"]))
            for entry in experts)
    return value


def validate_payload(
        payload: Mapping[str, Any], *,
        expected_provenance: Mapping[str, Any] | None = None,
        min_rows_per_arm: int = 30) -> dict[str, Any]:
    """Admit the evaluator's batched payload through the main row validator.

    The main contract's aggregate validator expects both fit and holdout rows,
    while a real collection invocation owns one split.  Every row is therefore
    sent through ``six_expert_support_contract.validate_record`` here, followed
    by the same provenance, split, disjointness and per-arm checks.  If a
    payload contains both splits, the main aggregate validator is also called.
    No tensor or record is written by this function.
    """
    contract = load_main_contract()
    if not isinstance(payload, Mapping):
        raise contract.ContractError("payload must be an object")
    if payload.get("run_status") != "COMPLETED":
        raise contract.ContractError("payload is incomplete")
    if payload.get("base_expert") != "source_e260":
        raise contract.ContractError("post-option base expert must be source_e260")
    if payload.get("post_option_policy") != "canonical_route_expert":
        raise contract.ContractError("post-option policy must remain canonical_route_expert")
    payload_provenance = payload.get("provenance")
    if not isinstance(payload_provenance, Mapping):
        raise contract.ContractError("payload provenance missing")
    expected_axis = getattr(contract, "OBJECT_LIFT_AXIS_METADATA", None)
    if not isinstance(expected_axis, Mapping):
        raise contract.ContractError("support contract lacks object_lift_axis provenance")
    if payload_provenance.get("object_lift_axis") != dict(expected_axis):
        raise contract.ContractError("payload object_lift_axis provenance differs")
    if (expected_provenance is not None and
            "object_lift_axis" in expected_provenance and
            expected_provenance["object_lift_axis"] != dict(expected_axis)):
        raise contract.ContractError("expected object_lift_axis provenance differs")
    records = payload.get("records")
    if not isinstance(records, Mapping):
        raise contract.ContractError("payload records missing")

    required = sorted(contract.REQUIRED_FIELDS)
    episode_ids = _to_python(records.get("episode_id"))
    if not isinstance(episode_ids, Sequence) or isinstance(episode_ids, (str, bytes)):
        raise contract.ContractError("record episode_id batch missing")
    n = len(episode_ids)
    if n <= 0:
        raise contract.ContractError("at least one support record is required")
    columns: dict[str, Any] = {}
    missing = [name for name in required if name not in records]
    if missing:
        raise contract.ContractError(f"canonical payload fields missing: {missing}")
    for name in required:
        value = _to_python(records[name])
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise contract.ContractError(f"record field is not a batch: {name}")
        if len(value) != n:
            raise contract.ContractError(f"record batch length differs: {name}")
        if name == "assignment_propensity":
            # The simulator stores 1/6 in float32.  Check that representation
            # before normalizing to the main validator's exact Python value;
            # arbitrary propensity drift remains fail-closed.
            if any(not isinstance(item, (int, float)) or
                   not math.isclose(float(item), 1.0 / 6.0,
                                    rel_tol=0.0, abs_tol=1e-6)
                   for item in value):
                raise contract.ContractError(
                    "assignment_propensity must equal 1/6 before admission")
            value = [1.0 / 6.0] * n
        columns[name] = value

    route_experts = _to_python(records.get("route_expert"))
    if (not isinstance(route_experts, Sequence) or isinstance(route_experts, (str, bytes))
            or len(route_experts) != n or any(name != "source_e260" for name in route_experts)):
        raise contract.ContractError(
            "post-option route must remain source_e260 for every canonical row")
    teacher_sources = columns["router_teacher_source"]
    if any(source != "c1_observation_router" for source in teacher_sources):
        raise contract.ContractError("C1 teacher source cannot use source_e260 fallback")

    rows = [
        {name: columns[name][index] for name in required}
        for index in range(n)
    ]
    summaries = [contract.validate_record(row) for row in rows]
    first = summaries[0]

    def _consistent(key: str) -> Any:
        values = {summary[key] for summary in summaries}
        if len(values) != 1:
            raise contract.ContractError(f"{key} drift across canonical rows")
        return next(iter(values))

    experts = _consistent("candidate_experts")
    router_model = _consistent("router_model_sha256")
    route_hash = _consistent("route_config_sha256")
    collector_hash = _consistent("collector_config_sha256")
    expected = _expected_provenance(expected_provenance, rows[0][
        "candidate_expert_names_and_checkpoint_sha256"], contract)
    for key, actual in (
            ("candidate_experts", experts),
            ("router_model_sha256", router_model),
            ("route_config_sha256", route_hash),
            ("collector_config_sha256", collector_hash)):
        if key in expected and expected[key] != actual:
            raise contract.ContractError(f"{key} does not match expected provenance")

    split_episode_ids: dict[str, set[str]] = {"fit": set(), "holdout": set()}
    arm_counts: dict[str, dict[str, int]] = {
        split: {str(arm): 0 for arm in range(contract.EXPERT_COUNT)}
        for split in split_episode_ids}
    for summary in summaries:
        split_episode_ids[summary["split"]].add(summary["episode_id"])
        arm_counts[summary["split"]][str(summary["assignment"])] += 1
    present_splits = {summary["split"] for summary in summaries}
    for split in present_splits:
        underflow = {arm: count for arm, count in arm_counts[split].items()
                     if count < min_rows_per_arm}
        if underflow:
            raise contract.ContractError(
                f"{split} arm support below {min_rows_per_arm}: {underflow}")
    if present_splits == {"fit", "holdout"} and not split_episode_ids["fit"].isdisjoint(
            split_episode_ids["holdout"]):
        raise contract.ContractError("fit and holdout episode IDs overlap")

    if present_splits == {"fit", "holdout"}:
        contract.validate_records(rows, expected_provenance=expected,
                                  min_rows_per_arm=min_rows_per_arm)
    names = [name for name, _ in experts]
    flat_counts = {
        names[index]: sum(arm_counts[split][str(index)] for split in present_splits)
        for index in range(contract.EXPERT_COUNT)
    }
    return {
        "rows": n,
        "split": sorted(present_splits),
        "arm_counts": flat_counts,
        "split_arm_counts": arm_counts,
        "main_contract": contract.SCHEMA,
        "main_row_validation": True,
        "router_teacher_source": "c1_observation_router",
        "post_option_route": "source_e260",
        "candidate_experts": [
            {"name": name, "checkpoint_sha256": digest}
            for name, digest in experts],
        "router_model_sha256": router_model,
        "route_config_sha256": route_hash,
        "collector_config_sha256": collector_hash,
        "object_lift_axis": dict(expected_axis),
    }


def adapt_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Map only lossless legacy aliases; reject unverifiable fields.

    The old temporal evaluator has a twenty-step future mask and ten-step
    executed-action sequence.  Their first entries can be mapped losslessly,
    while reduced ``state``/history tensors cannot stand in for the required
    1442-dimensional observation or object poses.  Missing fields therefore
    remain a hard error instead of being synthesized.
    """
    contract = load_main_contract()
    if not isinstance(raw, Mapping):
        raise contract.ContractError("evaluator row must be a mapping")
    row = dict(raw)
    if "episode_id" not in row and "globally_unique_episode_id" in row:
        row["episode_id"] = row["globally_unique_episode_id"]
    if "contact_mask_t_plus_1_to_t_plus_5" not in row:
        future_contact = row.get("future_contact_mask")
        if isinstance(future_contact, (list, tuple)) and len(future_contact) >= 5:
            row["contact_mask_t_plus_1_to_t_plus_5"] = list(future_contact[:5])
    if "executed_action" not in row:
        option_actions = row.get("option_executed_action")
        if (isinstance(option_actions, (list, tuple)) and option_actions and
                isinstance(option_actions[0], (list, tuple))):
            row["executed_action"] = list(option_actions[0])
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        for field in ("route_config_sha256", "collector_config_sha256"):
            if field not in row and field in provenance:
                row[field] = provenance[field]
    missing = sorted(contract.REQUIRED_FIELDS.difference(row))
    if missing:
        raise contract.ContractError(
            f"adapter cannot construct canonical row; missing fields: {missing}")
    return row


def audit_legacy_evaluator(source: str) -> dict[str, Any]:
    """Report whether a legacy evaluator can emit the canonical row fields."""
    contract = load_main_contract()
    required = sorted(contract.REQUIRED_FIELDS)
    missing_source_markers = [field for field in required if field not in source]
    router_disabled = "routed.MODEL_PATH = None" in source
    return {
        "contract_path": str(_contract_path()),
        "required_fields": required,
        "missing_source_markers": missing_source_markers,
        "router_disabled_or_rejected": router_disabled,
        "ready": not missing_source_markers and not router_disabled,
    }


__all__ = [
    "adapt_record",
    "audit_legacy_evaluator",
    "build_manifest",
    "load_main_contract",
    "validate_payload",
    "validate_records",
]
