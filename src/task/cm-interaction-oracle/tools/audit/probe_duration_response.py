#!/usr/bin/env python3
"""Physics-only randomized duration Decision; no learned models or selectors."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from intervention import physical_targets, residuals
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits, dexplore_action_to_native_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_targets(trajectory, rest_height):
    """Common-time readouts, without filtering failures before step16."""
    pair = trajectory[:, :, 71] > .5
    lift = trajectory[:, :, 2]-rest_height[:, None]
    lost = torch.zeros(len(pair), dtype=torch.long)
    proxy_failure = torch.zeros_like(pair)
    for step in range(32):
        lost = torch.where(pair[:, step], 0, lost+1)
        proxy_failure[:, step] = lost >= 6
    height_failure = lift < .02
    y = torch.stack((pair[:, :16].float().mean(1), pair[:, 16:].float().mean(1),
        height_failure[:, 16:].any(1).float(),
        (height_failure | proxy_failure)[:, 16:].any(1).float(),
        height_failure.any(1).float()), -1)
    return y, dict(early_failure=(height_failure | proxy_failure)[:, :16].any(1),
        all32_failure=(height_failure | proxy_failure).any(1),
        height_failure32=height_failure.any(1), proxy_failure32=proxy_failure.any(1))


def current_design(p, waves):
    before = p["before"]
    phase = p["context"][:, 3].numpy()
    groups = waves*12+p["motion_id"].numpy()*4+np.minimum((phase*4).astype(int), 3)
    features = torch.cat((before[:, :7], p["history"][:, -1, :18],
        before[:, 13:48].reshape(-1, 5, 7)[:, :, :3].flatten(1),
        before[:, 48:63].reshape(-1, 5, 3).norm(dim=-1), before[:, 63:66],
        p["hand_root"][:, :3], p["context"][:, 3:4],
        p["decision_tick"][:, None].float()), -1).numpy().astype(float)
    scale = features.std(0)
    variable = scale > 1e-5
    design = np.concatenate(((groups[:, None] == np.unique(groups)[None]).astype(float),
        (features[:, variable]-features[:, variable].mean(0))/scale[variable]), -1)
    return design, groups


def residual_fit(design, labels, target, n_cells=18):
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    q = u[:, singular > singular[0]*1e-10]
    y = target-q@(q.T@target)
    x = (labels[:, None] == np.arange(1, n_cells+1)[None]).astype(float)
    x -= q@(q.T@x)
    coefficient = np.linalg.pinv(x, rcond=1e-10)@y
    return coefficient, q, y, int(np.linalg.matrix_rank(x))


def randomization(design, labels, target, groups, seed, family_ids=None, n_cells=18):
    coefficient, q, y, rank = residual_fit(design, labels, target, n_cells)
    energy = (y*y).sum(0)
    supported = energy > 1e-10
    def statistic(assignment):
        x = (assignment[:, None] == np.arange(1, n_cells+1)[None]).astype(float)
        x -= q@(q.T@x)
        beta = np.linalg.pinv(x, rcond=1e-10)@y
        explained = (beta*((x.T@x)@beta)).sum(0)
        return np.divide(explained, energy, out=np.zeros_like(energy), where=supported)
    observed = statistic(labels)
    null = np.zeros((1999, target.shape[1]))
    rng = np.random.default_rng(seed)
    rows_per_block = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for index in range(len(null)):
        permuted = labels.copy()
        for rows in rows_per_block:
            permuted[rows] = rng.permutation(labels[rows])
        null[index] = statistic(permuted)
    family_ids = family_ids or dict(short_contact=[0], continuation_contact=[1],
        continuation_height_failure=[2], continuation_combined_failure=[3],
        all32_height_failure=[4], I16=list(range(5, 19)))
    tails, families = np.ones(target.shape[1]), {}
    for name, indices in family_ids.items():
        active = np.array(indices)[supported[indices]]
        if not len(active):
            families[name] = dict(supported_axes=0, max_tail=None)
            continue
        null_max = null[:, active].max(1)
        for axis in active:
            tails[axis] = (1+(null_max >= observed[axis]).sum())/2000
        families[name] = dict(supported_axes=len(active),
            max_tail=float((1+(null_max >= observed[active].max()).sum())/2000),
            max_partial_explained=float(observed[active].max()))
    return coefficient, tails, families, dict(nuisance_rank=q.shape[1],
        treatment_rank=rank, blocks=len(rows_per_block), permutations=len(null),
        partial_explained=observed.tolist(), axis_family_tail=tails.tolist())


def retention_candidates(coefficient, tails, half_coefficients):
    """Registered duration growth, monotonicity, late alignment and repetition."""
    candidates = []
    if tails[0] > .10:
        return candidates
    for arm in range(1, 7):
        rows = [arm-1, 6+arm-1, 12+arm-1]
        sign = np.sign(coefficient[rows[-1], 0])
        short = coefficient[rows, 0]*sign
        if short[2] < .10 or short[2]-short[0] < .05:
            continue
        if not (short[0] >= -.02 and short[1] >= 0 and short[1] >= short[0]-.02 and short[2] >= short[1]-.02):
            continue
        repeated_short = [float(beta[rows[-1], 0]*sign) for beta in half_coefficients]
        if min(repeated_short) < .03:
            continue
        alignments = []
        for axis, threshold, orientation, name in ((1, .05, sign, "continuation_contact"),
                                                    (2, .10, -sign, "continuation_height_failure")):
            repeated = [float(beta[rows[-1], axis]*orientation) for beta in half_coefficients]
            if tails[axis] <= .10 and coefficient[rows[-1], axis]*orientation >= threshold and min(repeated) > 0:
                alignments.append(dict(quantity=name, half_signed_effects=repeated))
        if alignments:
            candidates.append(dict(arm=arm, direction="beneficial" if sign > 0 else "harmful",
                signed_short_by_duration=short.tolist(), half_signed_short=repeated_short,
                alignments=alignments))
    return candidates


def audit_packet(p):
    assert p["schema"] == "ref2dex.randomized_intervention.v2"
    assert p["duration_levels"] == [4, 8, 16] and p["decision_region"] == "early-hold"
    assert torch.equal(p["delta"], residuals())
    for key, value in p.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    assert p["valid_steps"].all() and (p["pre_hold_steps"] >= 6).all()
    assert ((p["before"][:, 2]-p["rest_height"] >= .03) & (p["before"][:, 71] > .5)).all()
    assert torch.equal(p["history"][:, -1, 36:39], p["before"][:, :3])
    assert torch.equal(p["history"][:, -1, 86:139], p["before"][:, 13:66])
    assert torch.equal(p["base_action"], p["base_actions"][:, 0])
    active = torch.arange(32)[None] < p["duration"][:, None]
    delta = p["delta"][p["arm"]][:, None]
    expected = torch.where(active[:, :, None], (p["base_actions"]+delta).clamp(-1, 1), p["base_actions"])
    assert torch.equal(expected, p["actions"])
    action_diff, pd_diff = p["actions"]-p["base_actions"], p["pd_targets"]-p["pd_base_targets"]
    assert (action_diff[~active] == 0).all() and (pd_diff[~active] == 0).all()
    assert (action_diff[p["arm"] == 0] == 0).all() and (pd_diff[p["arm"] == 0] == 0).all()
    asset = ROOT / "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
    lower, upper = native_joint_limits(asset, "cpu")
    a, b = p["actions"].flatten(0, 1), p["base_actions"].flatten(0, 1)
    q = torch.zeros_like(a)  # current wrist q cancels in target differences.
    reconstructed = dexplore_action_to_native_targets(a, q, lower, upper)-dexplore_action_to_native_targets(b, q, lower, upper)
    pd_error = float((reconstructed.reshape_as(pd_diff)-pd_diff).abs().max())
    assert pd_error < 5e-7
    rows = []
    for duration in (4, 8, 16):
        for arm in range(7):
            keep = (p["duration"] == duration) & (p["arm"] == arm)
            ratio = None
            if arm and keep.any():
                d = p["delta"][arm]
                ratio = float((action_diff[keep, :duration]*d.sign()).sum()/(keep.sum()*duration*d.abs().sum()))
            rows.append(dict(duration=duration, arm=arm, count=int(keep.sum()), dose_ratio=ratio))
    return dict(status="PASS", trials=len(p["arm"]), complete_windows=int(p["valid_steps"].all(1).sum()),
        native_pd_max_error=pd_error, cells=rows,
        event_counts={key:int(value.sum()) for key,value in response_targets(p["trajectory"], p["rest_height"])[1].items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=219)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    manifest = dict(run_status="STARTED", command=sys.argv, input_sha256=sha(args.dataset),
        dataset=str(args.dataset.resolve()), seed=args.seed, device="cpu: file/dose/statistical analysis, no models",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={str(path.relative_to(ROOT)):sha(path) for path in (SCRIPT, ROOT / "src/task/cm-interaction-oracle/src/intervention.py", ROOT / "src/task/CmResidual/dexplore_cm_geometry.py")},
        collection_manifest_sha256=sha(args.dataset.parent / "manifest.json"), created_at=datetime.now(timezone.utc).isoformat())
    def save(name, value):
        (args.run_dir / name).write_text(json.dumps(value, indent=2)+"\n")
    save("manifest.json", manifest)
    try:
        p = torch.load(args.dataset, map_location="cpu", weights_only=False)
        audit = audit_packet(p)
        save("engineering_audit.json", audit)
        collection = json.loads((args.dataset.parent / "manifest.json").read_text())
        assert collection["run_status"] == "COMPLETED"
        command = collection["command"]
        env_count = int(command[command.index("--num_envs")+1])
        waves = p["episode_id"].numpy()//env_count
        half = waves >= collection["waves"]//2
        arms, durations = p["arm"].numpy(), p["duration"].numpy()
        labels = np.where(arms == 0, 0, np.searchsorted([4, 8, 16], durations)*6+arms)
        half_counts = [[int(((labels == cell) & (half == side)).sum()) for cell in range(19)] for side in (False, True)]
        support = all(row["count"] >= 30 for row in audit["cells"] if row["arm"])
        support &= int((arms == 0).sum()) >= 90 and min(min(rows[1:]) for rows in half_counts) >= 12
        dose = all(row["dose_ratio"] is not None and row["dose_ratio"] >= .90 for row in audit["cells"] if row["arm"])
        y, details = response_targets(p["trajectory"], p["rest_height"])
        interaction = physical_targets(p["before"], p["trajectory"][:, 15])[:, 12:]
        target = torch.cat((y, interaction), -1).numpy().astype(float)
        design, groups = current_design(p, waves)
        beta, tails, families, statistics = randomization(design, labels, target, groups, args.seed)
        half_fits = [residual_fit(design[half == side], labels[half == side], target[half == side]) for side in (False, True)]
        half_beta = [fit[0] for fit in half_fits]
        candidates = retention_candidates(beta, tails, half_beta)
        valid = support and dose and statistics["treatment_rank"] == 18 and all(fit[3] == 18 for fit in half_fits)
        rows = [dict(duration=k, arm=a, count=int(((durations == k) & (arms == a)).sum()),
            adjusted_arm_minus_zero=beta[j*6+a-1].tolist(),
            half_adjusted_arm_minus_zero=[value[j*6+a-1].tolist() for value in half_beta])
            for j,k in enumerate((4, 8, 16)) for a in range(1, 7)]
        # Common-time response curves and compensatory commands are descriptive.
        pair = (p["trajectory"][:, :, 71] > .5).float()
        object_dz = p["trajectory"][:, :, 2]-p["before"][:, None, 2]
        hand = p["trajectory"][:, :, 13:48].reshape(-1, 32, 5, 7)[:, :, :, :3].mean(2)
        hand -= p["before"][:, 13:48].reshape(-1, 5, 7)[:, :, :3].mean(1)[:, None]
        drift = p["base_actions"]-p["base_action"][:, None]
        i_curve = torch.cat((torch.log1p(p["trajectory"][:, :, 48:63].reshape(-1, 32, 5, 3).norm(dim=-1)),
            torch.log1p(p["trajectory"][:, :, 63:66].abs()), p["trajectory"][:, :, 66:72]), -1)
        assert torch.equal(i_curve[:, 15], interaction)
        curves = torch.cat((pair[:, :, None], object_dz[:, :, None], hand[:, :, [0, 2]], drift, i_curve), -1)
        curve_beta = residual_fit(design, labels, curves.flatten(1).numpy().astype(float))[0].reshape(18, 32, 36)
        curve_rows = []
        for j,k in enumerate((4, 8, 16)):
            for arm in range(1, 7):
                row = curve_beta[j*6+arm-1]
                d = p["delta"][arm].numpy()
                projected = (row[:, 4:22][:, d != 0]/d[d != 0]).mean(1)
                curve_rows.append(dict(duration=k, arm=arm, contact=row[:, 0].tolist(),
                    object_dz=row[:, 1].tolist(), hand_dx=row[:, 2].tolist(), hand_dz=row[:, 3].tolist(),
                    baseline_drift_along_residual=projected.tolist(), I14=row[:, 22:].tolist()))
        save("response_curves.json", dict(time_steps=list(range(1, 33)), adjusted_arm_minus_zero=curve_rows,
            limits="Post-treatment common-time descriptions; negative baseline drift is consistent with compensation, not proof of its causal role."))
        result = dict(status="PROMISING" if valid and candidates else ("UNPROMISING" if valid else "UNCLEAR"),
            support_pass=bool(support), dose_pass=bool(dose), candidates=candidates, families=families,
            statistics=statistics, half_treatment_ranks=[fit[3] for fit in half_fits],
            adjusted_cells=rows, half_counts=half_counts,
            audit=audit, elapsed_seconds=time.monotonic()-started, neural_models_executed=False,
            limits="Single-cohort duration/cumulative-dose/recovery-time response; no unique feedback-cancellation identification, certified contact, same-state selection or trained-policy utility.")
        save("result.json", result)
        torch.save(dict(target=target, details=details, I16=interaction, labels=labels,
            groups=groups, design=design, wave=waves, half=half, coefficient=beta,
            half_coefficient=half_beta, family_tails=tails), args.run_dir / "diagnostic.pt")
        print(json.dumps({key:value for key,value in result.items() if key not in ("adjusted_cells", "audit", "half_counts", "statistics")}, indent=2))
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save("manifest.json", manifest)


if __name__ == "__main__":
    main()
