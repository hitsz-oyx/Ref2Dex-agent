"""Object-level H10 randomized wrist-z action-value screen with env CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
from src.task.CmResidual.tools.probe_train8_causal_head import MOTION_NAMES, OBJECT_NAMES, SPLIT_SHA


SOURCE = Path("outputs/CmResidual/agent_expert_train8_randomized_s188_h10")
SOURCE_SHA = "58f3b687544b36f3b1833aed286ef39c2af5ebb6913141c64cc54c88fdec7e7c"
OFFICIAL_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    path = SOURCE / "transitions.pt"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("source_actor_role") != "official_data_collector" or
            manifest.get("checkpoint_sha256") != OFFICIAL_SHA or
            manifest.get("motion_manifest_sha256") != SPLIT_SHA or
            manifest.get("transitions_sha256") != SOURCE_SHA or
            manifest.get("source_objects") != OBJECT_NAMES or sha256(path) != SOURCE_SHA):
        raise ValueError("train8 H10 source drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.randomized_action_followup.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("followup_horizon") != 10 or
            payload.get("intervention_axes") != [2] or
            payload.get("delta_z_action") != .1):
        raise ValueError("H10 randomized schema drift")
    r = payload["records"]
    action = r["executed_action"] - r["base_action"]
    assignment = r["assignment"].reshape(-1).numpy()
    expected = torch.zeros_like(action)
    expected[:, 2] = r["assignment"] * .1
    # The legacy single-axis collector clips the executed z action to [-1, 1].
    # Preserve randomized assignment (intention-to-treat) while rejecting wrong
    # axes, directions, excessive doses, or material non-compliance.
    assigned = assignment != 0
    realized = action[:, 2].numpy()
    compliant = np.isclose(realized, expected[:, 2].numpy(), atol=1e-5)
    if (not torch.allclose(action[:, :2], expected[:, :2], atol=1e-5) or
            not torch.allclose(action[:, 3:], expected[:, 3:], atol=1e-5) or
            np.any(realized[assigned] * assignment[assigned] <= 0) or
            np.any(np.abs(realized[assigned]) > .1 + 1e-5) or
            compliant[assigned].mean() < .99):
        raise ValueError("executed randomized action drift")
    motion = r["motion_id"].reshape(-1).numpy()
    steps = r["global_step"].reshape(-1).numpy()
    env_id = np.arange(len(assignment)) % 64
    survived = (((r["followup_progress"] - r["progress"]).reshape(-1) == 10) &
                (r["followup_reset"].reshape(-1) == 0)).numpy()
    dz = ((r["followup_object_state"][:, 2] - r["object_state"][:, 2]) * 1000).numpy()
    contact_fraction = (r["followup_contact_count"].reshape(-1).numpy() / 10)
    outcome = np.where(survived, dz * contact_fraction, 0.0)
    if not np.isfinite(outcome).all() or set(motion.tolist()) != set(range(len(MOTION_NAMES))):
        raise ValueError("invalid H10 outcome or motion mapping")
    by_object = {}
    for object_name in OBJECT_NAMES:
        keep = np.isin(motion, [i for i, name in enumerate(MOTION_NAMES) if name == object_name])
        treated = keep & (assignment != 0)
        strata = steps + motion * 1000
        effect = weighted_step_difference(outcome[treated], assignment[treated], strata[treated])[0]
        clusters = [np.flatnonzero(treated & (env_id == i)) for i in range(64)
                    if np.any(treated & (env_id == i))]
        rng = np.random.default_rng(240994 + OBJECT_NAMES.index(object_name))
        draws = []
        for _ in range(1000):
            indices = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
            try:
                draws.append(weighted_step_difference(outcome[indices], assignment[indices],
                                                      strata[indices])[0])
            except ValueError:
                continue
        if len(draws) < 900:
            raise ValueError(f"too few valid env bootstraps for {object_name}")
        ci = np.quantile(draws, [.025, .975]).tolist()
        reliable = abs(effect) >= 10 and (ci[0] > 0 or ci[1] < 0)
        by_object[object_name] = {"treated": int(treated.sum()),
                                  "env_clusters": len(clusters),
                                  "plus_minus_effect_mm": effect,
                                  "env_cluster_95ci_mm": ci,
                                  "reliable_ge10mm": bool(reliable),
                                  "mean_contact_fraction": float(contact_fraction[treated].mean()),
                                  "survival_fraction": float(survived[treated].mean())}
    reliable_effects = [v["plus_minus_effect_mm"] for v in by_object.values()
                        if v["reliable_ge10mm"]]
    gate = (len(reliable_effects) >= 3 and
            any(v > 0 for v in reliable_effects) and any(v < 0 for v in reliable_effects))
    report = {"schema": "ref2dex.train8_h10_action_value.v1",
              "classification": "Probe", "source_sha256": SOURCE_SHA,
              "split_sha256": SPLIT_SHA, "by_object": by_object,
              "action_compliance": {
                  "assigned_rows": int(assigned.sum()),
                  "full_dose_rows": int(compliant[assigned].sum()),
                  "partial_clipped_rows": int((~compliant[assigned]).sum()),
                  "minimum_realized_abs_dose": float(np.abs(realized[assigned]).min()),
                  "analysis": "intention_to_treat",
              },
              "reliable_object_count": len(reliable_effects),
              "mixed_sign_reliable_effects": bool(gate),
              "continue_z_option_selector": bool(gate),
              "limits": ["Official actor only collected data; no self-trained policy utility measured.",
                         "H10 contact-supported lift is not held-lift success."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"by_object": by_object, "continue_z_option_selector": bool(gate)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
