#!/usr/bin/env python3
"""ref4 Gate A/B first; action-conditioned I training only if both pass."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from intervention import continuation_outcomes, physical_targets, residuals, CHUNK
sys.path.insert(0, str(SCRIPT.parent))
from probe_interventions import fit_model, standardized, split_stratified


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retention_ranking(prediction, target, ids, groups, arms):
    tolerances = (.03, .03, .5, .03, .03, .5)
    accuracies, support = [], []
    for channel, tolerance in enumerate(tolerances):
        scores, counts = [], []
        for group in sorted(set(groups[ids].tolist())):
            rows = ids[groups[ids] == group]
            i, j = np.triu_indices(len(rows), 1)
            left, right = rows[i], rows[j]
            difference = target[left, channel]-target[right, channel]
            keep = (arms[left] != arms[right]) & (np.abs(difference) > tolerance)
            left, right, difference = left[keep], right[keep], difference[keep]
            if len(left):
                product = (prediction[left, channel]-prediction[right, channel])*difference
                scores.append(float(((product > 0)+.5*(product == 0)).mean()))
                counts.append(dict(stratum=int(group), pairs=len(left)))
        accuracies.append(float(np.mean(scores)) if scores else None)
        support.append(counts)
    primary = [accuracies[i] for i in (3, 5) if accuracies[i] is not None]
    return dict(per_head=accuracies, support=support,
                primary=float(np.mean(primary)) if len(primary)==2 else None)


def adjusted_arm_test(p, interaction, outcome, details, seed):
    """Reviewed current-state residualization and within-block label permutation."""
    before = p["before"]
    phase = p["context"][:, 3].numpy()
    blocks = p["motion_id"].numpy()*8+np.minimum((phase*4).astype(int), 3)*2+(p["decision_tick"].numpy()<20).astype(int)
    unique = np.unique(blocks)
    block_design = (blocks[:, None] == unique[None]).astype(float)
    features = torch.cat((before[:, :7], p["history"][:, -1, :18],
        before[:, 13:48].reshape(-1, 5, 7)[:, :, :3].flatten(1),
        before[:, 48:63].reshape(-1, 5, 3).norm(dim=-1), before[:, 63:66],
        p["hand_root"][:, :3], p["context"][:, 3:4], p["decision_tick"][:, None].float()), -1).numpy().astype(float)
    scale = features.std(0)
    variable = scale > 1e-5
    controls = np.concatenate((block_design, (features[:, variable]-features[:, variable].mean(0))/scale[variable]), -1)
    u, singular, _ = np.linalg.svd(controls, full_matrices=False)
    q = u[:, singular > singular[0]*1e-10]
    target = np.concatenate((interaction.numpy(), details["short_contact_fraction"][:, None].numpy(),
                             outcome.numpy(), details["all32_failure"][:, None].float().numpy()), -1).astype(float)
    y_residual = target-q@(q.T@target)
    energy = (y_residual**2).sum(0)
    supported = energy > 1e-10
    arms = p["arm"].numpy()
    def statistic(labels):
        x = (labels[:, None] == np.arange(1, 7)[None]).astype(float)
        x_residual = x-q@(q.T@x)
        coefficient = np.linalg.pinv(x_residual, rcond=1e-10)@y_residual
        explained = ((x_residual@coefficient)**2).sum(0)
        return np.divide(explained, energy, out=np.zeros_like(energy), where=supported), coefficient
    observed, coefficient = statistic(arms)
    rng = np.random.default_rng(seed)
    null = np.zeros((1999, target.shape[1]))
    rows_per_block = [np.flatnonzero(blocks == b) for b in unique]
    for index in range(1999):
        permuted = arms.copy()
        for rows in rows_per_block:
            permuted[rows] = rng.permutation(arms[rows])
        null[index], _ = statistic(permuted)
    family_ids = dict(I14=np.arange(14), short_contact=np.array([14]), continuation=np.arange(15, 21), all32_failure=np.array([21]))
    adjusted_tail = np.ones(target.shape[1])
    families = {}
    for name, indices in family_ids.items():
        active = indices[supported[indices]]
        if not len(active):
            families[name] = dict(supported_axes=0, max_tail=None)
            continue
        null_max = null[:, active].max(-1)
        for i in active:
            adjusted_tail[i] = (1+(null_max>=observed[i]).sum())/2000
        families[name] = dict(supported_axes=len(active), max_tail=float((1+(null_max>=observed[active].max()).sum())/2000),
                              max_partial_explained=float(observed[active].max()))
    short_range = float(np.ptp(np.concatenate(([0.], coefficient[:, 14]))))
    short_positive = bool(supported[14] and adjusted_tail[14] <= .10 and short_range >= .10)
    aligned = []
    for axis in range(8, 13):  # existing object-surface distance axes only
        if adjusted_tail[axis] > .05:
            continue
        for arm_index in range(6):
            retained = coefficient[arm_index, 18] >= .05 and adjusted_tail[18] <= .10
            less_failure = coefficient[arm_index, 20] <= -.10 and adjusted_tail[20] <= .10
            if coefficient[arm_index, axis] < 0 and (retained or less_failure):
                aligned.append(dict(arm=arm_index+1, axis=axis, retention=bool(retained), failure=bool(less_failure)))
    return dict(status="PROMISING" if short_positive or aligned else "UNPROMISING",
        gate_pass=bool(short_positive or aligned), short_range=short_range,
        short_positive=short_positive, aligned=aligned, families=families,
        axis_partial_explained=observed.tolist(), axis_family_tail=adjusted_tail.tolist(),
        adjusted_arm_minus_zero=coefficient.tolist(), nuisance_rank=q.shape[1], blocks=len(unique), permutations=1999,
        limits="conditional exploratory randomization; coarse nuisance/carryover; no same-state comparisons or cross-family Validation")


def dropout_auc(score, labels):
    positive, negative = score[labels > .5], score[labels < .5]
    if not len(positive) or not len(negative):
        return None
    difference = positive[:, None]-negative[None]
    return float(((difference > 0)+.5*(difference == 0)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=215)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    collection_manifest = json.loads((args.dataset.parent / "manifest.json").read_text())
    manifest = dict(run_status="STARTED", command=sys.argv, seed=args.seed, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_sha256=sha(args.dataset), dataset=str(args.dataset.resolve()), collection_manifest_sha256=sha(args.dataset.parent / "manifest.json"),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            (SCRIPT, SCRIPT.parent / "probe_interventions.py", ROOT / "src/task/cm-interaction-oracle/src/intervention.py")},
        torch_version=torch.__version__, created_at=datetime.now(timezone.utc).isoformat(),
        protocol="ref4 early-hold; A randomized contrast + B GT I prognosis before conditional C")
    def save_manifest():
        (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    save_manifest()
    try:
        p = torch.load(args.dataset, map_location="cpu", weights_only=False)
        if p.get("decision_region") != "early-hold" or not torch.equal(p["delta"], residuals()):
            raise ValueError("wrong early-hold dose/region contract")
        for key, value in p.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("nonfinite "+key)
        outcome, details = continuation_outcomes(p["before"], p["trajectory"], p["rest_height"])
        interaction = physical_targets(p["before"], p["trajectory"][:, 7])[:, 12:]
        arm = p["arm"].numpy()
        counts = np.bincount(arm, minlength=7)
        delta = p["delta"][p["arm"]][:, None].expand(-1, CHUNK, -1)
        actual = p["actions"][:, :CHUNK]-p["base_actions"][:, :CHUNK]
        ratios = []
        for a in range(1, 7):
            keep = (p["arm"] == a)[:, None, None] & (delta.abs() > 0)
            ratios.append(float((actual[keep]*delta[keep].sign()).sum()/delta[keep].abs().sum()) if keep.any() else None)
        zero_error = float((p["actions"][arm == 0]-p["base_actions"][arm == 0]).abs().max()) if (arm == 0).any() else None
        after_error = float((p["actions"][:, CHUNK:]-p["base_actions"][:, CHUNK:]).abs().max()) if len(arm) else None
        audit = dict(trials=len(arm), counts=counts.tolist(), dose_by_nonzero_arm=ratios, zero_action_error=zero_error,
            post_chunk_error=after_error, at_risk=int(details["risk"].sum()), complete_windows=int(p["valid_steps"].all(-1).sum()),
            minimum_pre_hold_steps=int(p["pre_hold_steps"].min()) if len(arm) else None,
            outcome_std=outcome.std(0, unbiased=False).tolist(),
            event_counts={key:int(details[key].sum()) for key in
                ("early_failure", "all32_failure", "height_failure32", "proxy_loss_failure32", "late_height_failure", "late_proxy_loss_failure")})
        (args.run_dir / "data_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
        invalid = []
        if len(arm)<196 or counts.min()<20:
            invalid.append("insufficient trial/arm support")
        if not p["valid_steps"].all() or not details["risk"].all() or not (p["pre_hold_steps"]>=6).all():
            invalid.append("assigned window/early-hold timing violation")
        if zero_error not in (None, 0.) or after_error not in (None, 0.) or any(r is None or r < .9 for r in ratios):
            invalid.append("execution dose/zero baseline violation")
        if invalid:
            result = dict(status="UNCLEAR", stopping_gate=invalid, audit=audit, C_executed=False)
        else:
            gate_a = adjusted_arm_test(p, interaction, outcome, details, args.seed)
            (args.run_dir / "randomized_arm_result.json").write_text(json.dumps(gate_a, indent=2)+"\n")
            command = collection_manifest["command"]
            env_count = int(command[command.index("--num_envs")+1])
            clusters = p["episode_id"].numpy()%env_count
            motion = p["motion_id"].numpy()
            for cluster in np.unique(clusters):
                if len(set(motion[clusters == cluster].tolist())) != 1:
                    raise ValueError("motion changed within environment cluster")
            train_np, test_np = split_stratified(motion, clusters, args.seed)
            if set(clusters[train_np]) & set(clusters[test_np]):
                raise ValueError("environment crossed train/test")
            device = torch.device("cuda:0")
            train, test = torch.as_tensor(train_np, device=device), torch.as_tensor(test_np, device=device)
            h_raw = torch.cat((p["history"].flatten(1), p["actor_obs"], p["context"], p["base_action"]), -1).to(device)
            h_normalized, h_mean, h_scale = standardized(h_raw, train)
            _, _, v = torch.linalg.svd(h_normalized[train], full_matrices=False)
            projection = v[:32].T
            h, projected_mean, projected_scale = standardized(h_normalized@projection, train)
            i_gpu, y_gpu = interaction.to(device), outcome.to(device)
            i_normalized, i_mean, i_scale = standardized(i_gpu, train)
            y_normalized, y_mean, y_scale = standardized(y_gpu, train)
            empty = torch.zeros(len(arm), 26, device=device)
            i_input = empty.clone()
            i_input[:, :14] = i_normalized
            group = motion*4+np.minimum((p["context"][:, 3].numpy()*4).astype(int), 3)
            y_np = outcome.numpy()
            models, predictions, metrics = {}, {}, {}
            def score(name, x):
                normalized_prediction, model = fit_model(x, y_normalized, train, args.seed)
                with torch.no_grad():
                    prediction = (normalized_prediction*y_scale+y_mean).cpu().numpy()
                mse = (normalized_prediction[test]-y_normalized[test]).square().mean()
                metrics[name] = dict(test_normalized_mse=float(mse),
                    train_normalized_mse=float((normalized_prediction[train]-y_normalized[train]).square().mean()),
                    test_MAE=np.abs(prediction[test_np]-y_np[test_np]).mean(0).tolist(),
                    failure32_auc=dropout_auc(prediction[test_np, 5], y_np[test_np, 5]),
                    ranking=retention_ranking(prediction, y_np, test_np, group, arm),
                    parameters=sum(value.numel() for value in model.parameters()))
                models[name], predictions[name] = model.state_dict(), prediction
                return model
            baseline_x = torch.cat((h, empty), -1)
            score("H", baseline_x)
            score("GT_I", torch.cat((h, i_input), -1))
            positive = int((outcome[test_np, 5]>.5).sum())
            negative = len(test_np)-positive
            rank_support = metrics["H"]["ranking"]["support"]
            support_per_head = [sum(row["pairs"]>=10 for row in rank_support[channel]) for channel in (3, 5)]
            support = positive>=10 and negative>=10 and min(support_per_head)>=3
            baseline_rank = metrics["H"]["ranking"]["primary"]
            gt_rank = metrics["GT_I"]["ranking"]["primary"]
            gt_rank_gain = gt_rank-baseline_rank if gt_rank is not None and baseline_rank is not None else None
            gt_error_gain = 1-metrics["GT_I"]["test_normalized_mse"]/metrics["H"]["test_normalized_mse"]
            gate_b_pass = bool(support and gt_rank_gain>=.03 and gt_error_gain>=.05)
            gate_b = dict(status="PROMISING" if gate_b_pass else ("UNPROMISING" if support else "UNCLEAR"),
                gate_pass=gate_b_pass, adequate_support=bool(support), test_failure_positive=positive,
                test_failure_negative=negative, supported_strata_per_primary_head=support_per_head,
                gt_rank_gain=gt_rank_gain, gt_error_gain=gt_error_gain)
            gate_c = dict(executed=False, reason="A and B both required before I predictor/direct/Cm training")
            extra = {}
            if gate_a["gate_pass"] and gate_b_pass:
                action, a_mean, a_scale = standardized(p["delta"][p["arm"]].to(device), train)
                action_slots = empty.clone()
                action_slots[:, :18] = action
                action_x = torch.cat((h, action_slots), -1)
                predicted_h, _ = fit_model(baseline_x, i_normalized, train, args.seed)
                predicted_action, i_model = fit_model(action_x, i_normalized, train, args.seed)
                perm = np.arange(len(arm))
                rng = np.random.default_rng(args.seed+1)
                for stratum in set(group[test_np].tolist()):
                    rows = test_np[group[test_np] == stratum]
                    perm[rows] = rng.permutation(rows)
                permuted_x = torch.cat((h, action_slots[torch.as_tensor(perm, device=device)]), -1)
                with torch.no_grad():
                    predicted_permuted = i_model(permuted_x)
                i_oof = torch.zeros_like(i_normalized)
                folds = np.full(len(arm), -1)
                for m in set(motion.tolist()):
                    envs = np.unique(clusters[train_np[motion[train_np] == m]])
                    rng.shuffle(envs)
                    for index, cluster in enumerate(envs):
                        folds[train_np[clusters[train_np] == cluster]] = index%3
                for fold in range(3):
                    fit_ids = torch.as_tensor(train_np[folds[train_np] != fold], device=device)
                    hold_ids = torch.as_tensor(train_np[folds[train_np] == fold], device=device)
                    target, mean, scale = standardized(i_gpu, fit_ids)
                    predicted, _ = fit_model(action_x, target, fit_ids, args.seed)
                    i_oof[hold_ids] = (predicted[hold_ids]*scale+mean-i_mean)/i_scale
                i_oof[test] = predicted_action[test]
                mediated_slots = empty.clone()
                mediated_slots[:, :14] = i_oof
                mediated_model = score("mediated", torch.cat((h, mediated_slots), -1))
                score("direct", action_x)
                permuted_slots = empty.clone()
                permuted_slots[:, :14] = predicted_permuted
                with torch.no_grad():
                    perm_pred = (mediated_model(torch.cat((h, permuted_slots), -1))*y_scale+y_mean).cpu().numpy()
                predictions["mediated_permuted"] = perm_pred
                perm_rank = retention_ranking(perm_pred, y_np, test_np, group, arm)
                errors = {name:float((value[test]-i_normalized[test]).square().mean())
                          for name, value in (("H", predicted_h), ("Ha", predicted_action), ("permuted", predicted_permuted))}
                cm_gain = 1-errors["Ha"]/errors["H"]
                cm_sensitivity = errors["permuted"]/errors["Ha"]-1
                med_rank = metrics["mediated"]["ranking"]["primary"]
                med_gain, unique_gain = med_rank-baseline_rank, med_rank-metrics["direct"]["ranking"]["primary"]
                rank_sensitivity = med_rank-perm_rank["primary"]
                flags = dict(I_gain=cm_gain>=.05, I_action_sensitivity=cm_sensitivity>=.05,
                    mediated_vs_H=med_gain>=.03, mediated_vs_direct=unique_gain>=.02, mediated_action_sensitivity=rank_sensitivity>=.02)
                gate_c = dict(executed=True, status="PROMISING" if all(flags.values()) else "UNPROMISING", gates=flags,
                    I_mse=errors, I_gain=cm_gain, I_action_sensitivity=cm_sensitivity,
                    mediated_vs_H=med_gain, mediated_vs_direct=unique_gain, mediated_action_sensitivity=rank_sensitivity,
                    permuted_ranking=perm_rank)
                extra = dict(i_predictions={"H":predicted_h.cpu(), "Ha":predicted_action.cpu(), "OOF":i_oof.cpu(), "permuted":predicted_permuted.cpu()},
                    i_model=i_model.state_dict(), a_mean=a_mean.cpu(), a_scale=a_scale.cpu(), perm=perm, folds=folds)
            status = "PROMISING" if gate_c.get("status") == "PROMISING" else "UNPROMISING"
            if not gate_b["adequate_support"]:
                status = "UNCLEAR"
            result = dict(status=status, audit=audit, A=gate_a, B=gate_b, C=gate_c, models=metrics,
                train_trials=len(train_np), test_trials=len(test_np), train_environments=len(set(clusters[train_np])),
                test_environments=len(set(clusters[test_np])), C_executed=gate_c["executed"])
            torch.save(dict(outcome=outcome, details=details, I=interaction, train_ids=train.cpu(), test_ids=test.cpu(),
                clusters=clusters, groups=group, predictions=predictions, models=models,
                h_mean=h_mean.cpu(), h_scale=h_scale.cpu(), projection=projection.cpu(),
                projected_mean=projected_mean.cpu(), projected_scale=projected_scale.cpu(),
                i_mean=i_mean.cpu(), i_scale=i_scale.cpu(), y_mean=y_mean.cpu(), y_scale=y_scale.cpu(),
                y_normalized=y_normalized.cpu(), **extra), args.run_dir / "diagnostic.pt")
        result.update(elapsed_seconds=time.monotonic()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        (args.run_dir / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()


if __name__ == "__main__":
    main()
