#!/usr/bin/env python3
"""CPU-only exploratory calibration repair screen for fixed r7 records.

The method choices are frozen before holdout scoring: five-fold cross-fitted
ridge predictions with residual-inflated intervals, plus a zero-contact/range
baseline. This is exploratory cross-conformal-style screening, not a formal
split-conformal certificate and not policy training.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, pathlib, time
from collections import Counter
import numpy as np
import torch

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
except RuntimeError:
    pass

NARM = 6
EXPERTS = ["balanced_e360", "cup_e340", "duck_e340", "mixed12_e300", "source_e260", "train5_e320"]
FIT_SHA = "aa60b82b35d95ea6278e8cf0c386d00b5fcf6ed5fc3ec31bc7a858c0f31dbf93"
HOLD_SHA = "b49e30045ba2370546cc0c6ea8949660fbcdc7984a584fe4d67113f59ddeef80"
ROUTER_SHA = "1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14"
NOMINAL = 0.90
THRESHOLD = 0.60
FOLDS = 5
RIDGE_LAMBDA = 1.0


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_payload(path: pathlib.Path):
    return torch.load(path, map_location="cpu")


def unpack(payload):
    r = payload["records"]
    n = int(r["env_id"].numel())
    ids = r["assignment"].long().numpy()
    obs = r["pre_action_observation"].float().numpy().astype(np.float64)
    actions = r["candidate_actions"].float().numpy().astype(np.float64)
    chosen = actions[np.arange(n), ids]
    onehot = np.eye(NARM, dtype=np.float64)[ids]
    x = np.concatenate([obs, chosen, onehot], axis=1)
    delta = r["target_delta_object_local_1"].float().numpy().astype(np.float64)
    contact = r["contact_mask_t_plus_1_to_t_plus_5"].float().mean(dim=1).numpy().astype(np.float64)
    return {"records": r, "n": n, "episode_id": list(r["episode_id"]), "assignment": ids,
            "router": r["router_teacher_candidate_id"].long().numpy(), "axis": r["object_lift_axis"].float().numpy().astype(np.float64),
            "actions": actions, "obs": obs, "x": x, "delta": delta, "contact": contact}


def all_candidate_x(d):
    n = d["n"]
    obs6 = np.repeat(d["obs"][:, None, :], NARM, axis=1)
    one = np.eye(NARM, dtype=np.float64)[None, :, :].repeat(n, axis=0)
    return np.concatenate([obs6, d["actions"], one], axis=2)


def fit_ridge(x: np.ndarray, y: np.ndarray):
    mean_x = x.mean(axis=0)
    scale_x = x.std(axis=0)
    scale_x[scale_x < 1e-12] = 1.0
    z = (x - mean_x) / scale_x
    mean_y = y.mean(axis=0)
    yc = y - mean_y
    gram = z @ z.T + RIDGE_LAMBDA * np.eye(len(x), dtype=np.float64)
    alpha = np.linalg.solve(gram, yc)
    beta = z.T @ alpha
    def predict(x_new):
        return ((x_new - mean_x) / scale_x) @ beta + mean_y
    return predict


def cross_fit(d):
    n = d["n"]
    fold = np.arange(n) % FOLDS
    oof_d = np.zeros((n, 3), dtype=np.float64)
    oof_c = np.zeros(n, dtype=np.float64)
    for k in range(FOLDS):
        te = fold == k
        tr = ~te
        pd = fit_ridge(d["x"][tr], d["delta"][tr])
        pc = fit_ridge(d["x"][tr], d["contact"][tr])
        oof_d[te] = pd(d["x"][te])
        oof_c[te] = pc(d["x"][te])
    return oof_d, oof_c, fold


def metric_rows(y_delta, y_contact, pred_delta, pred_contact, d_radius, c_radius, labels=None):
    q10 = np.clip(pred_contact - c_radius, 0.0, 1.0)
    lo = pred_delta - d_radius
    hi = pred_delta + d_radius
    out = {
        "rows": int(len(y_contact)),
        "delta_rmse": float(np.sqrt(np.mean((pred_delta - y_delta) ** 2))),
        "delta_mae": float(np.mean(np.abs(pred_delta - y_delta))),
        "contact_mean_mae": float(np.mean(np.abs(np.clip(pred_contact, 0, 1) - y_contact))),
        "contact_q10_lower_coverage": float(np.mean(y_contact >= q10)),
        "delta_interval_coordinate_coverage": float(np.mean((y_delta >= lo) & (y_delta <= hi))),
        "delta_interval_all_coordinate_row_coverage": float(np.mean(np.all((y_delta >= lo) & (y_delta <= hi), axis=1))),
        "contact_q10_mean": float(np.mean(q10)),
        "delta_radius": [float(x) for x in d_radius],
        "contact_radius": float(c_radius),
    }
    if labels is not None:
        out.update(labels)
    return out


def choose_labels(d, pred_delta_all, pred_contact_all, d_radius, c_radius, threshold):
    # Import the pre-declared deterministic teacher-label rule, without writing to it.
    import importlib.util
    contract_path = pathlib.Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-cm/src/task/CmResidual/scratch_teacher_arbitration_contract.py")
    if not contract_path.is_file():
        contract_path = pathlib.Path.cwd() / "src/task/CmResidual/scratch_teacher_arbitration_contract.py"
    spec = importlib.util.spec_from_file_location("cm_contract_screen", contract_path)
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    n = d["n"]
    labels = np.zeros(n, dtype=np.int64)
    fallback = np.zeros(n, dtype=bool)
    progress = np.zeros(n, dtype=np.float64)
    for i in range(n):
        cands = []
        for k in range(NARM):
            cands.append({"candidate_id": int(k), "delta_mean": pred_delta_all[i, k].tolist(),
                          "lift_axis": d["axis"][i].tolist(),
                          "contact_lcb": float(np.clip(pred_contact_all[i, k] - c_radius, 0, 1)),
                          "uncertainty_radius": float(np.max(d_radius))})
        valid = [c for c in cands if c["contact_lcb"] >= threshold]
        labels[i] = cm.select_teacher_candidate(cands, contact_lcb_threshold=threshold,
                                                 fallback_candidate=int(d["router"][i]))
        fallback[i] = not bool(valid)
        progress[i] = float(np.dot(pred_delta_all[i, labels[i]], d["axis"][i]))
    return labels, fallback, progress


def stability(d, pred_delta_all, pred_contact_all, d_radius, c_radius):
    base, base_fb, base_prog = choose_labels(d, pred_delta_all, pred_contact_all, d_radius, c_radius, THRESHOLD)
    out = {}
    for threshold in (0.55, 0.60, 0.65):
        labels, fb, prog = choose_labels(d, pred_delta_all, pred_contact_all, d_radius, c_radius, threshold)
        out[str(threshold)] = {
            "agreement_with_0.60": float(np.mean(labels == base)),
            "changed_vs_0.60": int(np.sum(labels != base)),
            "fallback_rate": float(np.mean(fb)),
            "selection_rate": float(np.mean(~fb)),
            "label_counts": {EXPERTS[k]: int(np.sum(labels == k)) for k in range(NARM)},
            "mean_predicted_certified_progress": float(np.mean(prog)),
        }
    return out, base, base_fb, base_prog


def label_hash(d, labels):
    obj = {"episode_id": d["episode_id"], "cm_on_candidate_id": labels.tolist(), "cm_off_candidate_id": d["router"].tolist()}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit(fit_path, hold_path):
    if sha256_file(fit_path) != FIT_SHA:
        raise RuntimeError("fit hash drift")
    if sha256_file(hold_path) != HOLD_SHA:
        raise RuntimeError("holdout hash drift")
    fit = unpack(load_payload(fit_path)); hold = unpack(load_payload(hold_path))
    for d, split in ((fit, "fit"), (hold, "holdout")):
        r = d["records"]
        if not np.isfinite(d["obs"]).all() or not np.isfinite(d["actions"]).all() or not np.isfinite(d["delta"]).all() or not np.isfinite(d["axis"]).all():
            raise RuntimeError(f"nonfinite {split}")
        norms = np.linalg.norm(d["axis"], axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-5:
            raise RuntimeError(f"axis unit failure {split}")
        if not np.allclose(d["delta"], r["object_pose_t_plus_1_object_local_frame"].float().numpy() - r["object_pose_t_object_local_frame"].float().numpy(), rtol=0, atol=1e-9):
            raise RuntimeError(f"pose delta mismatch {split}")
        if not torch.equal(r["contact_mask_t_plus_1_to_t_plus_5"], r["future_contact_mask"][:, :5]):
            raise RuntimeError(f"contact prefix mismatch {split}")
        if not np.allclose(r["assignment_propensity"].float().numpy(), 1/6, rtol=0, atol=1e-7):
            raise RuntimeError(f"propensity mismatch {split}")
        if len(set(d["episode_id"])) != d["n"]:
            raise RuntimeError(f"episode duplicates {split}")
        if set(fit["episode_id"]).intersection(hold["episode_id"]):
            raise RuntimeError("episode overlap")
        if set(r["router_teacher_source"]) != {"c1_observation_router"}:
            raise RuntimeError(f"router source mismatch {split}")
        if sorted(set(r["router_model_sha256"])) != [ROUTER_SHA]:
            raise RuntimeError(f"router hash mismatch {split}")
    return fit, hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl/outputs/P-20260928-six-expert-support-collection-r7-axis/fit/records.pt")
    ap.add_argument("--holdout", default="/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-rl/outputs/P-20260928-six-expert-support-collection-r7-axis-holdout/holdout/records.pt")
    ap.add_argument("--output", default="/tmp/CM_CALIBRATION_REPAIR_SCREEN_R1.json")
    args = ap.parse_args()
    started = time.time()
    fit_path, hold_path = pathlib.Path(args.fit), pathlib.Path(args.holdout)
    fit, hold = audit(fit_path, hold_path)
    # Freeze method choices after fit-only cross-fitting, before any holdout scoring.
    oof_d, oof_c, folds = cross_fit(fit)
    d_radius = np.quantile(np.abs(fit["delta"] - oof_d), NOMINAL, axis=0, method="linear")
    c_radius = float(np.quantile(np.abs(fit["contact"] - oof_c), NOMINAL, method="linear"))
    pd_fit = fit_ridge(fit["x"], fit["delta"])
    pc_fit = fit_ridge(fit["x"], fit["contact"])
    # Frozen final model is fit only; holdout is scored after all choices are fixed.
    fit_pred_d = pd_fit(fit["x"]); fit_pred_c = pc_fit(fit["x"])
    hold_pred_d = pd_fit(hold["x"]); hold_pred_c = pc_fit(hold["x"])
    fit_all = all_candidate_x(fit).reshape(fit["n"] * NARM, -1)
    hold_all = all_candidate_x(hold).reshape(hold["n"] * NARM, -1)
    fit_pred_da = pd_fit(fit_all).reshape(fit["n"], NARM, 3)
    hold_pred_da = pd_fit(hold_all).reshape(hold["n"], NARM, 3)
    fit_pred_ca = pc_fit(fit_all).reshape(fit["n"], NARM)
    hold_pred_ca = pc_fit(hold_all).reshape(hold["n"], NARM)
    conformal_st_fit, fit_labels, fit_fb, fit_prog = stability(fit, fit_pred_da, fit_pred_ca, d_radius, c_radius)
    conformal_st_hold, hold_labels, hold_fb, hold_prog = stability(hold, hold_pred_da, hold_pred_ca, d_radius, c_radius)
    crossfit_fit_metrics = metric_rows(fit["delta"], fit["contact"], oof_d, oof_c, d_radius, c_radius,
                                       {"folds": FOLDS, "fold_counts": [int(np.sum(folds == k)) for k in range(FOLDS)]})
    conformal_fit_metrics = metric_rows(fit["delta"], fit["contact"], fit_pred_d, fit_pred_c, d_radius, c_radius,
                                        {"label_fallback_rate": float(np.mean(fit_fb)), "selection_rate": float(np.mean(~fit_fb)), "label_sha256": label_hash(fit, fit_labels)})
    conformal_hold_metrics = metric_rows(hold["delta"], hold["contact"], hold_pred_d, hold_pred_c, d_radius, c_radius,
                                         {"label_fallback_rate": float(np.mean(hold_fb)), "selection_rate": float(np.mean(~hold_fb)), "label_sha256": label_hash(hold, hold_labels)})
    # Simple conservative baseline: no contact assertion (LCB=0), zero delta center with fit range.
    base_d_radius = np.max(np.abs(fit["delta"]), axis=0)
    base_c_radius = 1.0
    zero_fit_d = np.zeros_like(fit["delta"]); zero_hold_d = np.zeros_like(hold["delta"])
    zero_fit_c = np.zeros_like(fit["contact"]); zero_hold_c = np.zeros_like(hold["contact"])
    # Baseline labels use zero contact LCB, so every row uses frozen router fallback.
    base_fit_labels = fit["router"].copy(); base_hold_labels = hold["router"].copy()
    base_fit_fb = np.ones(fit["n"], dtype=bool); base_hold_fb = np.ones(hold["n"], dtype=bool)
    base_fit_metrics = metric_rows(fit["delta"], fit["contact"], zero_fit_d, zero_fit_c, base_d_radius, base_c_radius,
                                   {"label_fallback_rate": 1.0, "selection_rate": 0.0, "label_sha256": label_hash(fit, base_fit_labels)})
    base_hold_metrics = metric_rows(hold["delta"], hold["contact"], zero_hold_d, zero_hold_c, base_d_radius, base_c_radius,
                                    {"label_fallback_rate": 1.0, "selection_rate": 0.0, "label_sha256": label_hash(hold, base_hold_labels)})
    base_stability = {str(t): {"agreement_with_0.60": 1.0, "changed_vs_0.60": 0, "fallback_rate": 1.0, "selection_rate": 0.0,
                               "label_counts": {EXPERTS[k]: int(np.sum(base_hold_labels == k)) for k in range(NARM)}} for t in (0.55, 0.60, 0.65)}
    methods = {
        "crossfit_conformal_style": {
            "description": "5-fold fit-only cross-fitted ridge residual inflation; q90 absolute residual radii, exploratory cross-conformal style",
            "model": "centered ridge, lambda=1.0, input pre_action_observation + candidate_action + candidate_id_one_hot",
            "calibration_source": "fit OOF residuals only",
            "fit_oof": crossfit_fit_metrics,
            "fit_final": conformal_fit_metrics,
            "holdout": conformal_hold_metrics,
            "stability_fit": conformal_st_fit,
            "stability_holdout": conformal_st_hold,
            "radius_delta_q90": [float(x) for x in d_radius],
            "radius_contact_q90_abs": c_radius,
        },
        "conservative_zero_contact_range_delta": {
            "description": "constant contact lower bound 0 and zero delta center with fit max-absolute delta coordinate radius",
            "model": "no learned predictor",
            "calibration_source": "fit target range only",
            "fit": base_fit_metrics,
            "holdout": base_hold_metrics,
            "stability_holdout": base_stability,
            "radius_delta_fit_max_abs": [float(x) for x in base_d_radius],
            "radius_contact": 1.0,
        },
    }
    def gate(method):
        m = methods[method]
        h = m["holdout"]
        s = m["stability_holdout"]
        stab = min(s["0.55"]["agreement_with_0.60"], s["0.65"]["agreement_with_0.60"])
        checks = {
            "contact_q10_coverage_ge_0.90": h["contact_q10_lower_coverage"] >= 0.90,
            "delta_coordinate_coverage_ge_0.80": h["delta_interval_coordinate_coverage"] >= 0.80,
            "fallback_rate_le_0.50": h["label_fallback_rate"] <= 0.50,
            "threshold_stability_ge_0.90": stab >= 0.90,
        }
        return {"checks": checks, "min_stability_agreement": float(stab), "pass": bool(all(checks.values()))}
    gates = {name: gate(name) for name in methods}
    route_pass = any(v["pass"] for v in gates.values())
    report = {
        "schema": "ref2dex.cm_calibration_repair_screen_r1.v1",
        "experiment_id": "P-20260928-cm-calibration-repair-screen-r1",
        "conclusion": "PROMISING_CALIBRATION_REPAIR_SCREEN" if route_pass else "UNPROMISING_CALIBRATION_REPAIR_SCREEN",
        "classification_scope": "estimator/calibration screen only; no policy utility or Cm causal claim",
        "device": "cpu", "cpu_threads": 2, "gpu_used": 0,
        "input_files": {"fit_records": {"path": str(fit_path), "sha256": sha256_file(fit_path), "rows": fit["n"]},
                        "holdout_records": {"path": str(hold_path), "sha256": sha256_file(hold_path), "rows": hold["n"]}},
        "support_audit": {"fit_arm_counts": [int(np.sum(fit["assignment"] == k)) for k in range(NARM)],
                          "holdout_arm_counts": [int(np.sum(hold["assignment"] == k)) for k in range(NARM)],
                          "fit_holdout_episode_disjoint": True,
                          "axis_unit_max_abs_error": float(max(np.max(np.abs(np.linalg.norm(fit["axis"], axis=1) - 1)), np.max(np.abs(np.linalg.norm(hold["axis"], axis=1) - 1)))),
                          "router_teacher_source": "c1_observation_router", "router_model_sha256": ROUTER_SHA,
                          "pose_delta_exact": True, "contact_prefix_exact": True,
                          "input_fields": ["pre_action_observation", "candidate_action", "candidate_id_one_hot"],
                          "forbidden_fields_excluded": ["next_state", "future_contact", "final_held_lift", "terminal_success", "teacher_label"]},
        "frozen_method_choice": {"folds": FOLDS, "fold_assignment": "row_index_mod_5 on fit only", "nominal_coverage": NOMINAL,
                                 "threshold": THRESHOLD, "ridge_lambda": RIDGE_LAMBDA,
                                 "holdout_scored_after_method_freeze": True,
                                 "formal_distribution_free_certification": False,
                                 "reason": "prior holdout was inspected; this is exploratory cross-fitted screening"},
        "methods": methods, "gates": gates,
        "route_recommendation": "stop this calibration-repair estimator screen" if not route_pass else "retain estimator for a separately authorized follow-up; no policy claim",
        "resource_termination": {"gpu_used": 0, "isaac_gym": False, "collector": False, "new_transitions": False, "student_distillation": False,
                                 "ppo": False, "online": False, "cm_training": False, "processes_left": False,
                                 "wall_seconds": float(time.time() - started)},
        "labels": {"fit_crossfit_conformal": {"cm_on": fit_labels.tolist(), "fallback": fit_fb.tolist(), "label_sha256": label_hash(fit, fit_labels)},
                    "holdout_crossfit_conformal": {"cm_on": hold_labels.tolist(), "fallback": hold_fb.tolist(), "label_sha256": label_hash(hold, hold_labels)}},
        "notes": ["All labels are offline teacher choices generated from recorded candidate actions; no action was executed.",
                  "Cross-fitted residual inflation is exploratory and does not certify holdout coverage distribution-free.",
                  "A failed gate rejects this estimator/calibration repair screen, not the entire Cm research hypothesis."]
    }
    out = pathlib.Path(args.output); out.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "sha256": sha256_file(out), "conclusion": report["conclusion"], "gates": gates,
                      "holdout": {k: methods[k]["holdout"] for k in methods}}, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()

