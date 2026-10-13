"""Independent array audit of actual commands, PD conversion and chunk clock."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / "src"), str(ROOT / "src/task/trajectory-policy/tools/audit"),
                str(ROOT / "src/task/consequence-evaluator/src")]
from audit_competent_pd import independent_pd, independent_features, ACTIVE
from ptv3_shared_wam.checkpoint import atomic_json


def audit(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    result = json.loads((path / "result.json").read_text())
    if manifest["status"] != "COMPLETED" or result["status"] != "COMPLETED":
        raise ValueError("complete native execution required")
    with np.load(path / "trajectory.npz", allow_pickle=False) as stream:
        d = {key: stream[key].copy() for key in stream.files}
    controls, steps = manifest["controls"], manifest["execution_steps"]
    if len(d["q"]) != controls + 1 or len(d["applied"]) != controls:
        raise ValueError("terminal measured-state boundary differs")
    errors = {}
    def check(name, actual, expected):
        if actual.shape != expected.shape:
            raise ValueError("shape differs: " + name)
        difference = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
        errors[name] = difference
        if difference > 1e-6:
            raise ValueError("contract differs: " + name)
    if not all(np.isfinite(value).all() for value in d.values()):
        raise ValueError("nonfinite execution arrays")
    check("clamped_commands", d["applied"], d["intended"].clip(-1, 1))
    check("native_PD", d["pd_targets"], independent_pd(d["applied"], d["q"][:-1], manifest["native_controller"]))
    check("clipping_flags", d["clipped"], np.abs(d["intended"] - d["applied"]).max(-1) > 1e-6)
    ticks = np.arange(controls)
    query_ticks = ticks - ticks % steps
    check("query_tick", d["query_tick"], np.repeat(query_ticks[:, None], 2, axis=1))
    check("chunk_index", d["chunk_index"], np.repeat((ticks % steps)[:, None], 2, axis=1))
    past = np.maximum(query_ticks[:, None] + np.arange(-3, 1), 0)
    check("own_H4_hand_at_query", d["wam_hand_history"], d["hand"][past, 2])
    check("own_H4_object_at_query", d["wam_object_history"], d["obj"][past, 2])
    dummy_future = np.repeat(d["hand"][query_ticks, 2][:, None], 24, axis=1)
    state, _ = independent_features(d, query_ticks, np.full(controls, 2), dummy_future)
    check("own_R_state_at_query", d["R_state"], state)
    import torch
    # Metadata-only historical audit also accepts old goal contracts. It never
    # instantiates a model or reinterprets legacy weights under the new goal.
    packet = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    if packet.get("schema") != manifest["schema"] or packet.get("stage") != "robot":
        raise ValueError("matching robot checkpoint metadata required")
    with np.load(packet["provenance"]["reference"], allow_pickle=False) as stream:
        reference = stream["obj"].copy()
    check("planned_goal_at_query", d["wam_goal"], reference[np.minimum(query_ticks + 24, len(reference) - 1)])
    passive = [i for i in range(18) if i not in ACTIVE]
    check("passive_commands_zero", d["applied"][:, 1:, passive], np.zeros_like(d["applied"][:, 1:, passive]))
    outcomes = []
    for env, saved in enumerate(result["outcomes"]):
        height = d["obj"][:, env, 2, 3] - d["obj"][0, env, 2, 3]
        held = (d["surface_gap"][:, env] <= .01) & (height >= .03)
        held &= ~(d["table_footprint"][:, env].astype(bool) & (np.abs(d["support_gap"][:, env]) <= .02))
        held[0] = False
        run = longest = 0
        for value in held:
            run = run + 1 if value else 0
            longest = max(longest, run)
        if longest != saved["maximum_held_frames"]:
            raise ValueError("independent held count differs")
        outcomes.append(dict(role=saved["role"], maximum_held_frames=int(longest)))
    return dict(pass_=True, errors=errors, outcomes=outcomes,
                scope="execution wiring only; no trained-policy utility claim")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    a = p.parse_args()
    destination = a.native / "independent_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = audit(a.native)
    report["pass"] = report.pop("pass_")
    atomic_json(destination, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
