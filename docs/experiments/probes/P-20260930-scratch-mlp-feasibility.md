---
schema: ref2dex.probe.v2
probe_id: P-20260930-scratch-mlp-feasibility
date: 2026-09-30
branch: agent/cm-scratch-mlp-policy-probe
git_commit: see run results.json
claim_id: C3
hypothesis_family: HF06
probe_index_in_family: 3
seed_pool: probe
status: COMPLETED
---

# Decision prerequisite: fixed scratch MLP feasibility

Question: does the originally designed nonlinear physical envelope clear the
existing coverage/selection gates, or is teacher arbitration still blocked?
This is the cheapest existing-data screen before paying for student
collection/training/physical evaluation. It does not answer Cm policy utility.

Protocol, input limits, and fixed gates are in
[decision memo](../../decisions/D-20260930-cm-policy-utility-next.md).
Only candidate action/identity and pre-action observation enter the model.
No holdout tuning or method selection; only one final checkpoint. Existing
holdout has been inspected, so a pass is exploratory, never formal Validation.

Budget: 2 CPU, 15 minutes, 100 MiB, 0 GPU. Stop on input hash drift,
canonical support rejection, split overlap, nonfinite values or budget expiry.
The fit calibration split is recorded by episode id. No student or physical
action is executed in this prerequisite.

Positive: prepare matched Cm-on/off/placebo student and physical held-lift
Probe on a fixed task; positive offline predictions alone are insufficient.
Negative: close this teacher-envelope implementation without a local sweep;
choose a different high-level mechanism. Core Cm claim stays OPEN.

Result: `UNPROMISING` for this teacher-envelope implementation, run_status
`COMPLETED`, code `9d5b423`. Canonical admission passed. 150 optimization /
38 calibration rows; wall 4.28 s, 0 GPU. Holdout contact coverage 0.88172
failed 0.90, delta coverage 0.92115 passed 0.80, fallback 0.43548 passed
0.50, threshold stability 0.89247 failed 0.90. Stop this implementation;
do not rescan. These are estimator results, not Cm policy utility evidence.

Artifacts: `outputs/P-20260930-scratch-mlp-feasibility/cpu_s278_r1/`.
