---
schema: ref2dex.probe.v2
probe_id: P-20260926-selective-causal-gate
date: 2026-09-26
branch: agent/cm-selective-causal-gate
git_commit: de5e83a52efa365a40ff866ea31a1daef7698b79
claim_id: C3
hypothesis_family: HF05
probe_index_in_family: 1
seed_pool: existing-probe-records
status: PLANNED
classification: Decision
---

# Probe: conservative selective causal Cm intervention

## Decision question

Can Cm improve policy utility when it is allowed to abstain? The policy learns
arm-specific outcomes from randomized contact-stage interventions, but applies
`-0.1` or `+0.1` wrist-z only when the fit-only uncertainty bound predicts a
safe contact-supported lift gain over the no-op arm. All other states keep the
source action. This is a new decision structure, separate from expert selection,
post-action credit prediction, and always-on action choice.

## Frozen substrate and provenance

All records use the self-trained `source_e260` airplane actor, recorded
`motion_id=0`, checkpoint SHA256
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`, and motion
manifest SHA256
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`.
The assignment is exactly three-arm randomized `{-1, 0, +1}` wrist-z with
propensity `1/3`; only rows with valid contact, complete five-step followup and
first-episode labels are retained.

| split | simulator seeds | assignment seeds | expected valid rows |
| --- | --- | --- | ---: |
| fit | 250, 251 | 20260925250, 20260925251 | 63, 58 |
| holdout | 252, 253 | 20260925252, 20260925253 | 63, 63 |

The exact transition and manifest hashes are fixed in the evaluator and must be
written to the run manifest and result JSON. No holdout label may affect fitting,
normalization, bootstrap policy construction, or thresholds.

## Frozen causal model and policy

* Features are only pre-action `q`, `dof_vel`, `object_state`, and `base_action`
  (67 dimensions). No `next_*`, followup, or final label is an input.
* Fit one ridge outcome model per arm with fit-only normalization and
  `ridge_lambda=25.0`. Targets are `held_lift`, `contact_fraction`, and the fixed
  physical composite
  `supported_lift_mm = final_max_contact_lift_mm * final_contact_fraction`.
* Fit exactly 128 within-arm bootstrap models using seed `2026092606`. The
  treatment-shuffled placebo permutes fit assignments once with seed
  `2026092605`; holdout assignments remain untouched.
* For each nonzero arm, compute the fit-only 10th-percentile bootstrap gain in
  supported lift over arm 0 and the 90th-percentile bootstrap contact change.
  Select the arm with the largest eligible lower bound only when gain is at least
  **5 mm** and contact change is at most **+0.02**. Otherwise select arm 0.
* Compare `always_base`, `cm_selective`, `cm_point_gate`, and
  `shuffled_selective`. The policy must intervene on 5–50% of heldout rows so a
  zero-intervention result cannot pass by abstention alone.

## Predeclared continuation gate

The gate requires all of the following on the single holdout split:

1. at least 15 valid rows per arm in fit and holdout, all finite;
2. `cm_selective` held-lift IPW/Hájek value is at least **+5 percentage points**
   over `always_base` and at least +5 points over `shuffled_selective`;
3. its supported-lift value is not below `always_base`, and contact fraction is
   no more than 2 percentage points below `always_base`;
4. intervention fraction is between 5% and 50%.

If any condition fails, mark HF05 `UNPROMISING`, freeze it, and do not change
seed, target, bootstrap count, threshold, or policy rule. If all pass, write a
new confirmation memo before proposing any matched online test; this card never
authorizes GPU collection, PPO, or online execution.

## Resource and stop contract

CPU only, two threads, one deterministic run, and at most 20 MB of output. Stop
on route/checkpoint/manifest drift, invalid assignment or first-episode boundary,
nonfinite data, missing arm coverage, or any CUDA/Isaac Gym attempt.

Implementation and tests are pinned at commit
`de5e83a52efa365a40ff866ea31a1daef7698b79`:

* evaluator SHA256:
  `43a97c5342cd381c1fe32b948c17ce2ebbf6bf197037da3ff8527120786c40ed`;
* test SHA256:
  `805ff133c735f747723e839dfc5845674777be25922ab9eec3ebf88092211a79`.

