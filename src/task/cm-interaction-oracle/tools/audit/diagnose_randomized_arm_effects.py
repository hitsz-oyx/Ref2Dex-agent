#!/usr/bin/env python3
"""Post-hoc randomized-arm diagnostic; never changes the learned Probe gates."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from intervention import physical_targets, window_outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    p = torch.load(args.collection / "interventions.pt", map_location="cpu", weights_only=False)
    arm = p["arm"].numpy()
    before = p["before"]
    e = physical_targets(before, p["trajectory"][:, 7]).numpy()
    y, _ = window_outcomes(before, p["trajectory"], p["rest_height"])
    y = y.numpy()
    episodes = json.loads((args.collection / "episode_summary.json").read_text())
    manifest = json.loads((args.collection / "manifest.json").read_text())
    command = manifest["command"]
    num_envs = int(command[command.index("--num_envs")+1])
    keyed = {row["wave"]*num_envs+row["env_id"]: row for row in episodes}
    full = np.array([[keyed[int(i)]["stable_success"], keyed[int(i)]["drop_after_success"],
                      keyed[int(i)]["stable_success"] and not keyed[int(i)]["drop_after_success"]]
                     for i in p["episode_id"]], dtype=float)
    names = ["E_dx", "E_dy", "E_dz", "E_rx", "E_ry", "E_rz", "E_dvx", "E_dvy", "E_dvz", "E_dwx", "E_dwy", "E_dwz"]
    names += ["I_hand_force_%d" % i for i in range(5)] + ["I_object_force_%d" % i for i in range(3)]
    names += ["I_body_surface_distance_%d" % i for i in range(5)] + ["I_pair_proxy"]
    names += ["Y%d_%s" % (h, axis) for h in (16, 32) for axis in ("dz", "retention", "held", "at_risk_drop")]
    names += ["full_stable45", "full_later_drop", "full_stable_no_later_drop"]
    target = np.concatenate((e, y, full), -1).astype(float)
    phase = p["context"][:, 3].numpy()
    motion = p["motion_id"].numpy()
    blocks = motion*8 + np.minimum((phase*4).astype(int), 3)*2 + (p["decision_tick"].numpy()<20).astype(int)
    block_values = np.unique(blocks)
    block_design = (blocks[:, None] == block_values[None]).astype(float)
    features = torch.cat((before[:, :7], p["history"][:, -1, :18],
                          before[:, 13:48].reshape(-1, 5, 7)[:, :, :3].flatten(1),
                          before[:, 48:63].reshape(-1, 5, 3).norm(dim=-1), before[:, 63:66],
                          p["hand_root"][:, :3], p["context"][:, 3:4], p["decision_tick"][:, None].float()), -1).numpy().astype(float)
    scale = features.std(0)
    variable = scale > 1e-5
    controls = np.concatenate((block_design, (features[:, variable]-features[:, variable].mean(0))/scale[variable]), -1)
    # QR-like orthonormal nuisance basis, dropping exact linear dependencies.
    u, singular, _ = np.linalg.svd(controls, full_matrices=False)
    q = u[:, singular > singular[0]*1e-10]
    adjusted_y = target-q@(q.T@target)
    energy = (adjusted_y**2).sum(0)
    supported = energy > 1e-10
    def statistic(labels):
        x = (labels[:, None] == np.arange(1, 7)[None]).astype(float)
        adjusted_x = x-q@(q.T@x)
        coef = np.linalg.pinv(adjusted_x, rcond=1e-10)@adjusted_y
        fitted = adjusted_x@coef
        value = np.divide((fitted**2).sum(0), energy, out=np.zeros_like(energy), where=supported)
        return value, coef
    observed, coefficients = statistic(arm)
    rng = np.random.default_rng(211)
    null = np.zeros((1999, len(names)))
    block_rows = [np.flatnonzero(blocks == block) for block in block_values]
    for draw in range(len(null)):
        perm = arm.copy()
        for rows in block_rows:
            perm[rows] = rng.permutation(arm[rows])
        null[draw], _ = statistic(perm)
    families = dict(physical_EI8=np.arange(26), task16_32=np.arange(26, 34), full_episode=np.arange(34, 37))
    family_reports = {}
    adjusted_p = np.ones(len(names))
    for name, indices in families.items():
        indices = indices[supported[indices]]
        null_max = null[:, indices].max(-1)
        for index in indices:
            adjusted_p[index] = (1+(null_max>=observed[index]).sum())/2000
        family_reports[name] = dict(max_partial_explained=float(observed[indices].max()),
                                    permutation_tail=float((1+(null_max>=observed[indices].max()).sum())/2000),
                                    supported_axes=len(indices))
    result = dict(kind="POST_HOC_DECISION_DIAGNOSTIC", trials=len(arm), nuisance_rank=q.shape[1],
                  blocks=len(block_values), permutations=1999, seed=211, families=family_reports,
                  axes=[dict(name=name, supported=bool(supported[i]), partial_explained=float(observed[i]),
                             family_max_tail=float(adjusted_p[i]), adjusted_arm_minus_zero=coefficients[:, i].tolist())
                        for i, name in enumerate(names)], elapsed_seconds=time.monotonic()-started,
                  limits="diagnostic conditional randomization/coarse linear adjustment; repeated-environment carryover not eliminated; no Validation or changed primary gate")
    output = args.diagnostic / "randomized_arm_diagnostic.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "axes"}, indent=2))


if __name__ == "__main__":
    main()
