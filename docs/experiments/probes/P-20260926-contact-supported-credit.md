---
schema: ref2dex.probe.v2
probe_id: P-20260926-contact-supported-credit
date: 2026-09-26
branch: agent/cm-contact-credit
git_commit: 326fed4b7a9dabd8c0e535341e8c46fba5259b3d
claim_id: C3
hypothesis_family: HF03
decision_changed_if_positive: "Design a new bounded critic or credit-assignment Probe with matched Cm-off control."
decision_changed_if_negative: "Freeze HF03 and require a different higher-level representation or supervision goal."
probe_index_in_family: 1
seed_pool: probe
status: UNPROMISING
classification: Decision
---

# Probe: contact-supported credit from executed handflow

## Decision question

HF02 tested whether an interaction history could select a candidate expert at
first contact. It failed its policy-value gate. This new family asks a
different, training-time question: after a randomized contact-stage action has
actually executed, does the immediately observed handflow add predictive
information about the later first-episode held-lift outcome beyond the
pre-action state and action? A positive result would justify designing a new
critic/credit-assignment Probe; it would not itself establish Cm policy
utility.

This is a CPU-only retrospective audit of four already completed Cm-off
collections. It is independent of HF02 and does not reuse its six-expert
option assignment, route, horizon, or budget.

## Frozen substrate and seed ownership

All four runs use the same self-trained `source_e260` checkpoint
(`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`) and the
same airplane motion manifest (`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`).
The recorded rows contain only `motion_id=0`; therefore this card makes no
multi-motion or cross-object claim. The randomized intervention is one
unclipped `+/-0.1` wrist-z dose at a contacting state, followed by the frozen
source actor. The assignment probability is one half for each nonzero arm.

| split | simulator seed | assignment seed | valid rows | arm counts |
| --- | ---: | ---: | ---: | --- |
| fit | 246 | 20260925246 | 63 | -:31, +:32 |
| fit | 247 | 20260925247 | 60 | -:30, +:30 |
| holdout | 248 | 20260925248 | 60 | -:30, +:30 |
| holdout | 249 | 20260925249 | 62 | -:31, +:31 |

The exact transition and manifest hashes are recorded in the result index and
the audit run manifest. A row is accepted only when it is contacting, assigned
to a nonzero arm, remains in the first episode through the five-step followup,
has finite complete held-lift labels, and its executed action differs from the
base action only by the assigned wrist-z dose.

## Prespecified CPU comparison

Fit seeds 246/247 and evaluate once on seeds 248/249. Normalization is learned
from fit rows only; all models use ridge lambda 10.0 and the fixed shuffle seed
2026092603. The five inputs are:

1. `state_only`: pre-action `q`, `dof_vel`, and `object_state`;
2. `action_aware`: state plus base action and executed treatment delta;
3. `post_handflow`: action-aware input plus `next_q-q`,
   `next_object_state-object_state`, and immediate `next_contact`;
4. `action_shuffled`: action-aware input with the fit treatment delta
   permuted once;
5. `post_handflow_shuffled`: post-handflow treatment and post block jointly
   permuted once in the fit split.

Targets are binary first-episode `final_lift_success`, continuous
`final_max_contact_lift_m` in millimetres, and `final_contact_fraction`.
The primary continuation gate is fixed before inspecting the holdout result:
`post_handflow` must improve held-lift Brier score and max-contact-lift RMSE by
at least 5% relative to both `action_aware` and
`post_handflow_shuffled`, with at least 25 rows per arm in each split and all
estimates finite. No seed, horizon, ridge, target, or representation rescan is
allowed after this fit.

Resource contract: CPU only, two threads, <=20 MB output, no Isaac Gym,
collector, PPO, online Probe, or GPU process.

## Result

The row and finiteness gates passed (`123` fit rows, `122` holdout rows), but
the predictive gate failed:

| model | held-lift Brier | max-contact-lift RMSE (mm) |
| --- | ---: | ---: |
| `state_only` | 0.18699 | 284.633 |
| `action_aware` | 0.18676 | 293.118 |
| `post_handflow` | 0.19111 | 287.664 |
| `action_shuffled` | 0.18601 | 290.197 |
| `post_handflow_shuffled` | 0.18313 | 302.341 |

Relative to `action_aware`, post-handflow changes are **−2.33%** on held-lift
Brier and **+1.86%** improvement on continuous lift RMSE. Relative to the
post-handflow placebo they are **−4.36%** and **+4.85%**, respectively; neither
pair reaches the predeclared +5% joint gate. The post-action block therefore
does not provide a stable held-lift credit signal on this substrate.

Decision: **UNPROMISING**. Freeze HF03 after this one bounded audit. Do not
collect a new physical seed, start a critic/PPO integration, or tune the same
features and thresholds. A future Cm route must change the higher-level
representation or supervision hypothesis and receive a new goal/experiment
ID.

## Reproducibility artifacts

* evaluator and CPU audit code: `326fed4b7a9dabd8c0e535341e8c46fba5259b3d`;
  script SHA256 `f71c09edc4868eed8e269a99692c2758af7f1ef09abf7ec47773c37e47cfe42f`;
* CPU tests: `src/task/CmResidual/tests/test_contact_supported_credit_audit.py`,
  SHA256 `9c86b79bdd146ecc720afb87a112640a1fbaecdcd3cef16ba078df9d014b20f5`;
* run manifest:
  `outputs/CmResidual/agent_contact_supported_credit_audit_20260926_r3/run_manifest.json`,
  SHA256 `93c92ea2c5d425efe6fa0045b7afb45c2acff4bc67fc52bd39cd09ce9972b888`;
* CPU report:
  `outputs/CmResidual/agent_contact_supported_credit_audit_20260926_r3/report.json`,
  SHA256 `265a06272f460ff2ec48513496c42b990e9ff9c9c87dd5ed97fd70e8d89a52e3`;
* fitted CPU model artifact SHA256
  `3838f29ca827e42736ca7e68b6fb05c4c07ae889227b420f18bdc18ca6e9ebb5`.
