---
schema: ref2dex.probe.v2
probe_id: P-20260926-temporal-expert-credit
date: 2026-09-26
branch: agent/cm-temporal
git_commit: PLANNED
claim_id: C3
hypothesis_family: HF02
decision_changed_if_positive: Freeze a temporal Cm option head and run a matched online Cm-on/off probe on the same self-trained expert portfolio.
decision_changed_if_negative: Freeze HF02 and review a higher-level Cm representation or credit-allocation route; do not tune the same option family.
probe_index_in_family: 2
seed_pool: probe
status: IMPLEMENTED_PENDING_ONLINE
---

# Probe: temporal expert-option credit

## Question

Does a recent interaction history make the value of a frozen self-trained
expert option predictable at first hand-object contact, beyond the current
state and beyond a history-only or action-shuffled control?

This stays inside the fixed self-trained airplane task distribution. Apple,
cross-object transfer, and the previously stopped wrist-z action family are not
part of this Probe.

## Hypothesis

**H1:** A 10-step history of contact, relative object motion, joint velocity,
and executed actions contains information about which candidate expert should
be continued for the next 10 steps. A temporal-Cm head using that history and
the candidate expert action should predict held-lift and contact-supported lift
better than history-only and action-shuffled controls.

**Alternative:** History only estimates contact state, or the candidate expert
does not carry stable conditional value. In that case temporal-Cm will not
improve held-lift policy value on a held-out simulator seed.

## Decision

If H1 is supported, freeze the head and run a fresh matched online Cm-on/off
Probe using the same six self-trained experts and observation-driven baseline.
If H1 is not supported, mark this HF02 slot `UNPROMISING` or `UNCLEAR` and do
not repeat it by changing only seed, horizon, regularization, or threshold.

## Minimal protocol

1. Use the existing six self-trained expert checkpoints and the three airplane
   motions in the self-trained split. At first valid hand-object contact,
   randomly assign one of the fixed experts as a 10-step candidate option;
   the control continues the object-route expert. Continue the episode to its
   normal first-episode endpoint.
2. Save the pre-option 10-step history, candidate and base actions, assignment
   propensity, 20-step contact-supported lift, contact fraction, and complete
   held-lift label. Enforce first-episode boundaries and exact action execution.
3. Fit on seed 254 and evaluate on seed 255. Compare `state_only`,
   `history_only`, `temporal_cm`, and `action_shuffled` with a multi-arm
   inverse-propensity policy-value estimate. Use the same predeclared feature
   normalization for all arms.
4. Only if the offline head beats both controls by the predeclared minimum
   policy-value margin and does not reduce contact-supported lift, collect the
   frozen online Cm-on/off Probe. Otherwise stop before online integration.

## Budget

GPU: one idle GPU, sequentially, no more than two 64-environment collections.  
wall time: 30 minutes per collection; CPU analysis <=20 minutes.  
storage: <=200 MB of transition and report artifacts.

## Stop condition

Stop immediately for checkpoint/config/input drift, GPU occupancy, missing
first-contact history, incomplete episode labels, invalid assignment/action
alignment, nonfinite tensors, or a clearly failed necessary condition such as
fewer than 30 valid candidate rows per arm in the fit split.

## Result

Status: `IMPLEMENTED_PENDING_ONLINE`

The frozen temporal checkpoint is
`outputs/CmResidual/agent_cm_history_value_probe_20260925_v2/history_action.pt`
(SHA256
`3f4a4dec8d71334460d66070fdd1ea98450f479a2228b63716e3d7cffc5d5100`).
Its in-distribution held-out offline gate was positive: supported-lift
high-minus-low was `33.70 mm` with bootstrap 95% CI `[28.78, 38.67]`, and the
shuffled-action separation and contact-direction controls passed. The separate
cross-object transfer audit failed its physical ranking gate, so this card
keeps the online test on the same specialist substrate and makes no
generalization claim.

The online agent and bootstrap are implemented and CPU-smoke tested. The
matched GPU continuation is pending a compliant idle card; all currently
occupied cards belong to unrelated SDF jobs and are not touched.

## Decision update

Pending the two-seed offline comparison. This Probe cannot establish a formal
Cm causal claim; it only decides whether the temporal expert-option route is
worth an online matched test.

## Artifacts

Planned output root:
`outputs/CmResidual/agent_temporal_expert_credit_20260926/`
