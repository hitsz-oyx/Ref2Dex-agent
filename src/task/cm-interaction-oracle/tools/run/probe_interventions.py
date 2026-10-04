#!/usr/bin/env python3
"""Frozen four-arm consequence/outcome diagnostic on randomized interventions."""
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
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from intervention import physical_targets, window_outcomes, residuals, CHUNK


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def standardized(x, ids):
    mean = x[ids].mean(0)
    scale = x[ids].std(0, unbiased=False).clamp_min(.001)
    return (x-mean)/scale, mean, scale


def fit_model(x, target, ids, seed, mask=None):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[-1], 64), nn.Tanh(),
                          nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, target.shape[-1])).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    if mask is None:
        mask = torch.ones_like(target)
    for _ in range(100):
        prediction = model(x[ids])
        loss = ((prediction-target[ids]).square()*mask[ids]).sum() / mask[ids].sum().clamp_min(1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(x).detach(), model


def split_stratified(motion, arm, seed):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for m in sorted(set(motion.tolist())):
        for a in range(7):
            group = np.flatnonzero((motion == m) & (arm == a))
            rng.shuffle(group)
            count = max(1, int(round(len(group)*.2))) if len(group) >= 2 else 0
            test.extend(group[:count].tolist())
            train.extend(group[count:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(test))


def ranking(prediction, target, ids, groups, arms):
    """Macro within observed strata, different randomized arms; not same-state."""
    tolerances = np.array([.001, .03, .03, .5, .001, .03, .03, .5])
    result, support = [], []
    for channel in range(8):
        strata = []
        pair_count = 0
        for group in sorted(set(groups[ids].tolist())):
            if channel in (3, 7) and group % 2 == 0:
                continue  # drop ranking is defined only among at-risk states
            rows = ids[groups[ids] == group]
            if len(rows) < 2:
                continue
            i, j = np.triu_indices(len(rows), 1)
            left, right = rows[i], rows[j]
            difference = target[left, channel]-target[right, channel]
            keep = (arms[left] != arms[right]) & (np.abs(difference) > tolerances[channel])
            left, right, difference = left[keep], right[keep], difference[keep]
            if len(left):
                product = (prediction[left, channel]-prediction[right, channel])*difference
                strata.append(float(((product > 0)+.5*(product == 0)).mean()))
                pair_count += len(left)
        result.append(float(np.mean(strata)) if strata else None)
        support.append(dict(pairs=pair_count, strata=len(strata)))
    # Drop is separately conditional and never determines aggregate score.
    active = [result[i] for i in (0, 1, 2, 4, 5, 6) if result[i] is not None]
    return dict(per_head=result, support=support, macro=float(np.mean(active)) if active else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    started = time.monotonic()
    manifest = dict(run_status="STARTED", dataset=str(args.dataset.resolve()), dataset_sha256=sha(args.dataset),
        command=sys.argv, seed=args.seed, torch_version=torch.__version__,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                     (SCRIPT, ROOT / "src/task/cm-interaction-oracle/src/intervention.py")},
        created_at=datetime.now(timezone.utc).isoformat(), protocol="PCA32, MLP64/32, tanh, AdamW.002/decay.001, 100 fullbatch epochs, OOF3")
    manifest_path = args.run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    try:
        p = torch.load(args.dataset, map_location="cpu", weights_only=False)
        if p["schema"] != "ref2dex.randomized_intervention.v1":
            raise ValueError("wrong input schema")
        if not torch.equal(p["delta"], residuals()):
            raise ValueError("arm dose drift")
        for key, value in p.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("nonfinite packet "+key)
        arm = p["arm"].numpy()
        assigned = p["delta"][p["arm"]]
        effective = p["actions"][:, :CHUNK] - p["base_actions"][:, :CHUNK]
        intended = assigned[:, None].expand_as(effective)
        nonzero = intended.abs() > 0
        dose_ratio = (effective[nonzero]*intended[nonzero].sign()).sum()/intended[nonzero].abs().sum()
        dose_by_arm = []
        for a in range(7):
            keep = (p["arm"] == a)[:, None, None] & nonzero
            dose_by_arm.append(float((effective[keep]*intended[keep].sign()).sum()/intended[keep].abs().sum()) if keep.any() else None)
        # Baseline mapping, untreated steps and arm counts are implementation gates.
        untreated = (p["actions"][arm == 0]-p["base_actions"][arm == 0]).abs().max() if (arm == 0).any() else torch.tensor(0.)
        post = (p["actions"][:, CHUNK:]-p["base_actions"][:, CHUNK:]).abs().max()
        counts = np.bincount(arm, minlength=7)
        effect = physical_targets(p["before"], p["trajectory"][:, 7])
        y, risk = window_outcomes(p["before"], p["trajectory"], p["rest_height"])
        audit = dict(trials=len(arm), arm_counts=counts.tolist(), complete_windows=int(p["valid_steps"].all(-1).sum()),
                     dose_ratio=float(dose_ratio), dose_by_arm=dose_by_arm,
                     zero_arm_max_deviation=float(untreated), after_chunk_max_deviation=float(post),
                     outcome_std=y.std(0, unbiased=False).tolist(), drop_risk_count=int(risk.sum()),
                     physical_target_hash=tensor_hash(effect), outcome_hash=tensor_hash(y))
        (args.run_dir / "data_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
        failures = []
        if len(arm) < 196 or counts.min() < 20:
            failures.append("insufficient randomized-arm support")
        if not p["valid_steps"].all():
            failures.append("assigned window truncated; no survivor filtering allowed")
        if dose_ratio < .9 or untreated > 1e-7 or post > 1e-7:
            failures.append("execution dose/zero control failure")
        if y[:, 0].std(unbiased=False) < .001 or max(float(y[:, 1].std(unbiased=False)), float(y[:, 2].std(unbiased=False))) < .05:
            failures.append("insufficient physical task outcome variation")
        if failures:
            result = dict(status="UNCLEAR", stopping_gate=failures, audit=audit)
        else:
            train_np, test_np = split_stratified(p["motion_id"].numpy(), arm, args.seed)
            train = torch.as_tensor(train_np, device=device)
            test = torch.as_tensor(test_np, device=device)
            h_raw = torch.cat((p["history"].flatten(1), p["actor_obs"], p["context"], p["base_action"]), -1).to(device)
            h_norm, h_mean, h_scale = standardized(h_raw, train)
            # Train-only PCA, exact SVD; no random truncated-SVD variability.
            _, _, vt = torch.linalg.svd(h_norm[train], full_matrices=False)
            projection = vt[:32].T
            h, projected_mean, projected_scale = standardized(h_norm @ projection, train)
            a = assigned.to(device)/.1
            zeros = torch.zeros(len(arm), 26, device=device)
            action_pad = zeros.clone()
            action_pad[:, :18] = a
            z = effect.to(device)
            z_norm, z_mean, z_scale = standardized(z, train)
            y_gpu = y.to(device)
            y_norm, y_mean, y_scale = standardized(y_gpu, train)
            mask = torch.ones_like(y_norm)
            # Drop loss only at risk; no-drop prelift rows are not negatives.
            mask[:, [3, 7]] = risk.to(device)[:, None].float()
            if int(risk[train_np].sum()) < 20 or int(risk[test_np].sum()) < 5:
                mask[:, [3, 7]] = 0
            zero_input = torch.cat((h, zeros), -1)
            action_input = torch.cat((h, action_pad), -1)
            cm_h, cm_h_model = fit_model(zero_input, z_norm, train, args.seed)
            cm_a, cm_a_model = fit_model(action_input, z_norm, train, args.seed)
            # Within observed motion/stage permutation, randomized intent only.
            phase = np.minimum((p["context"][:, 3].numpy()*4).astype(int), 3)
            groups = p["motion_id"].numpy()*8 + phase*2 + risk.numpy().astype(int)
            perm = np.arange(len(arm))
            rng = np.random.default_rng(args.seed+1)
            for group in set(groups[test_np].tolist()):
                indices = test_np[groups[test_np] == group]
                perm[indices] = rng.permutation(indices)
            permuted_input = torch.cat((h, action_pad[torch.as_tensor(perm, device=device)]), -1)
            with torch.no_grad():
                cm_perm = cm_a_model(permuted_input)
            cm_metrics = {}
            for name, pred in (("H", cm_h), ("Ha", cm_a), ("Ha_permuted", cm_perm)):
                error = (pred[test]-z_norm[test]).square()
                cm_metrics[name] = dict(mse=float(error.mean()), E_mse=float(error[:, :12].mean()), I_mse=float(error[:, 12:].mean()))
            cm_oof = torch.zeros_like(z_norm)
            fold_ids = np.full(len(arm), -1, dtype=int)
            for m in range(3):
                for a_id in range(7):
                    rows = train_np[(p["motion_id"].numpy()[train_np] == m) & (arm[train_np] == a_id)]
                    rng.shuffle(rows)
                    fold_ids[rows] = np.arange(len(rows)) % 3
            for fold in range(3):
                fit_ids = torch.as_tensor(train_np[fold_ids[train_np] != fold], device=device)
                hold_ids = torch.as_tensor(train_np[fold_ids[train_np] == fold], device=device)
                fold_target, fold_mean, fold_scale = standardized(z, fit_ids)
                pred, _ = fit_model(action_input, fold_target, fit_ids, args.seed)
                cm_oof[hold_ids] = (pred[hold_ids]*fold_scale + fold_mean-z_mean)/z_scale
            cm_oof[test] = cm_a[test]
            inputs = dict(H=zero_input, direct=action_input,
                          mediated=torch.cat((h, cm_oof), -1), GT=torch.cat((h, z_norm), -1))
            predictions, metrics, models = {}, {}, {}
            for name, x in inputs.items():
                pred_norm, model = fit_model(x, y_norm, train, args.seed, mask)
                pred = (pred_norm*y_scale+y_mean).cpu().numpy()
                predictions[name], models[name] = pred, model.state_dict()
                mse = (pred_norm[test]-y_norm[test]).square()*mask[test]
                mae = np.abs(pred[test_np]-y.numpy()[test_np]).mean(0).tolist()
                for channel in (3, 7):
                    eligible_risk = risk.numpy()[test_np]
                    mae[channel] = float(np.abs(pred[test_np[eligible_risk], channel]-y.numpy()[test_np[eligible_risk], channel]).mean()) if eligible_risk.any() else None
                metrics[name] = dict(normalized_mse=float(mse.sum()/mask[test].sum()),
                    test_MAE=mae,
                    ranking=ranking(pred, y.numpy(), test_np, groups, arm))
                if name in ("direct", "mediated"):
                    perm_x = permuted_input if name == "direct" else torch.cat((h, cm_perm), -1)
                    with torch.no_grad():
                        perm_pred = (model(perm_x)*y_scale+y_mean).cpu().numpy()
                    predictions[name+"_permuted"] = perm_pred
                    metrics[name]["permuted_ranking"] = ranking(perm_pred, y.numpy(), test_np, groups, arm)
            h_mse, ha_mse = cm_metrics["H"]["mse"], cm_metrics["Ha"]["mse"]
            cm_gain = 1-ha_mse/h_mse
            cm_perm_loss = cm_metrics["Ha_permuted"]["mse"]/ha_mse-1
            gt_gain = 1-metrics["GT"]["normalized_mse"]/metrics["H"]["normalized_mse"]
            h_rank = metrics["H"]["ranking"]["macro"]
            gt_rank_gain = metrics["GT"]["ranking"]["macro"]-h_rank
            mediated_rank_gain = metrics["mediated"]["ranking"]["macro"]-h_rank
            mediated_sensitivity = metrics["mediated"]["ranking"]["macro"]-metrics["mediated"]["permuted_ranking"]["macro"]
            gates = dict(cm_gain=cm_gain>=.05, cm_action_sensitivity=cm_perm_loss>=.05,
                         gt_task_error=gt_gain>=.05, gt_task_rank=gt_rank_gain>=.03,
                         mediated_task_rank=mediated_rank_gain>=.03,
                         mediated_action_sensitivity=mediated_sensitivity>=.02)
            result = dict(status="PROMISING" if all(gates.values()) else "UNPROMISING", audit=audit,
                          train_trials=len(train_np), test_trials=len(test_np), cm=cm_metrics, outcomes=metrics,
                          gate_values=dict(cm_gain=cm_gain, cm_perm_loss=cm_perm_loss, gt_error_gain=gt_gain,
                                           gt_rank_gain=gt_rank_gain, mediated_rank_gain=mediated_rank_gain,
                                           mediated_action_sensitivity=mediated_sensitivity), gates=gates,
                          unique_cm_rank_gain=metrics["mediated"]["ranking"]["macro"]-metrics["direct"]["ranking"]["macro"])
            torch.save(dict(train_ids=train.cpu(), test_ids=test.cpu(), groups=groups, perm=perm, fold_ids=fold_ids,
                            physical_target=effect, outcome=y, risk=risk, predictions=predictions,
                            cm_predictions={"H": cm_h.cpu(), "Ha": cm_a.cpu(), "Ha_permuted": cm_perm.cpu(), "OOF": cm_oof.cpu()},
                            z_norm=z_norm.cpu(), y_norm=y_norm.cpu(), y_mask=mask.cpu(),
                            h_mean=h_mean.cpu(), h_scale=h_scale.cpu(), projection=projection.cpu(),
                            projected_mean=projected_mean.cpu(), projected_scale=projected_scale.cpu(),
                            z_mean=z_mean.cpu(), z_scale=z_scale.cpu(), y_mean=y_mean.cpu(), y_scale=y_scale.cpu(),
                            models=models, cm_model=cm_a_model.state_dict()), args.run_dir / "diagnostic.pt")
        result["elapsed_seconds"] = time.monotonic()-started
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        (args.run_dir / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
