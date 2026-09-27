# Cm distinct-mechanism design review — 2026-09-27

```text
STATUS=NO_DISTINCT_DESIGN
CLASSIFICATION=READ_ONLY_DECISION_DESIGN
EXPERIMENT_OR_CARD=NONE
GPU=0
NEW_DATA=0
```

## Scope and frozen contract

This review uses only the canonical main documents at
`3767988eb34a62b23e2c9fc36b106e5f18c85470` (the current `MISSION.md`,
`STATE.md`, `CAMPAIGN.md`, and [Cm freeze Option A](../decisions/D-20260927-cm-freeze-option-a.md)).
The accepted C1 route is treated as a fixed self-trained substrate; it is not
new Cm evidence. No code, registry, configuration, result, or existing
experiment was changed.

## Decision

No pre-declarable causal mechanism is currently identifiable that is both
policy-relevant and genuinely different from the frozen HF01–HF05 families,
the 3D/H10 train-time auxiliary route, and the previously reviewed phase
handoff idea. C3 therefore remains `OPEN` but frozen. A change of target,
seed, horizon, metric, or threshold would not satisfy the distinct-mechanism
requirement.

The unresolved gap is not another feature or window. It is a missing causal
policy decision that is outside the already tested classes (initial option
choice, temporal/post-action credit, or selective intervention). Existing
records do not select such a decision.

## What the evidence already bounds

- **Initial option/action choice is not a new opening.** HF01 did not meet its
  joint Brier and route gate; the earlier local-residual and action-boost
  records also show that an active Cm action selector did not improve the
  matched policy. See [HF01](../experiments/probes/P-20260925-cm-option-value.md),
  [local residual](../experiments/probes/P-20260925-cm-route-local-residual.md),
  and [action boost](../experiments/probes/P-20260924-cm-online-action-boost.md).
- **History-conditioned option value is already HF02.** Its temporal model
  lost to the action-shuffled placebo on the frozen holdout gate. A phase
  label, another history length, or an expert handoff is the same policy
  family unless a different causal decision is supplied. See
  [HF02](../experiments/probes/P-20260926-temporal-expert-credit.md).
- **Using information after an action is already covered.** HF03 and HF04
  tested post-action handflow and a short trajectory token and both failed
  their predeclared credit gates. See [HF03](../experiments/probes/P-20260926-contact-supported-credit.md)
  and [HF04](../experiments/probes/P-20260926-trajectory-credit.md).
- **Selective intervention is already HF05.** A gate that abstains unless a
  predicted intervention is safe did not reach nontrivial coverage or policy
  gain. See [HF05](../experiments/probes/P-20260926-selective-causal-gate.md).
- **Training-time auxiliary representation is not a distinct supported route.**
  The old 3D and H10 matched probes were below their stability/upgrade gate;
  [the representation review](CM_REPRESENTATION_ROUTE_REVIEW_20260926.md)
  explicitly found no unique target to preregister.
- **A model-based sequence planner is not presently selectable.** The old
  two-step structured model predicted a physical wrist-x interaction, but the
  task-aligned wrist-z sequence gate was `UNCLEAR`, and neither result is
  policy utility. Reusing the same pre-action effect to choose a dose or a
  short sequence would be a reparameterization of HF01/HF05, not a new
  mechanism. See [two-step structured model](../experiments/probes/P-20260924-cm-two-step-structured-model.md),
  [two-step z decision](../experiments/probes/P-20260924-cm-two-step-z-effect.md),
  and [Mission's matched causal requirement](../MISSION.md).

The C1 partition only established that many failures occur after the initial
route agrees; it did not identify a new intervention variable or causal target.
It therefore cannot promote the previously reviewed phase-handoff sketch into
a mechanism.

## One nearest-candidate audit (rejected as non-distinct)

For completeness, the closest conceivable candidate is:

> **Pre-action transition-control Cm:** predict the signed next-step or
> short-sequence object-relative displacement together with a contact-stability
> margin, then change the magnitude of the current base action (or choose a
> two-step dose).

This candidate has the required ingredients on paper:

- **Physical prediction object:** object-relative displacement/velocity and
  contact retention, computed from pre-action state, base action, and a
  candidate dose.
- **Policy decision:** continuous dose or short-sequence selection, rather
  than selecting an expert identity.
- **Leakage guard:** only pre-action `q`, joint velocity, object state, base
  action, and candidate dose may enter the predictor. `next_*`, post-action
  contact/force, future labels, and later expert choices are forbidden. A
  split must be by episode, with the action assignment fixed before outcomes.
- **Controls:** matched Cm-off executes the unchanged base action;
  action-shuffled uses a fixed candidate permutation; a parameter-matched
  generic transition/placebo head receives the same pre-action fields but a
  non-Cm target. A future positive result would still be a Probe, not a
  Validation or a Cm utility claim.
- **Cheapest falsification test (not authorized):** replay only the already
  randomized pre-action transitions, fit the same-capacity state-only and
  action-shuffled controls, and check the existing predeclared physical
  interaction/sign gate on a held-out split. Any post-action field, failed
  sign/interaction gate, or no improvement over both controls stops the route
  immediately; no seed, horizon, metric, or threshold search follows.

The audit rejects this candidate as a new family. Its physical target and
changed action are exactly the path exercised by the earlier CATE/effect-rank,
local-residual, two-step, and HF05 intervention routes; switching from expert
identity to a dose or from one to two steps changes notation, not the causal
mechanism. Predicting contact retention instead would be HF03/HF04, and making
it a continue/switch decision is the already reviewed phase-handoff/HF02
family. Thus there is no honest predeclared design to register from current
evidence.

## Route options and the actual missing decision

### Option A — keep the freeze (recommended)

- Cost now: `GPU=0`, new data `0`, no process, no code or registry change.
- Keep C1 frozen as the task-bound substrate; leave HF01–HF05 and C3 unchanged.
- Preserve the negative evidence and Research Debt. This is the only route
  supported by the current Decision Checkpoint.

### Option B — open a genuinely new high-level question (not authorized)

The next action would first be a **user Decision Checkpoint**, not a collector
or fit: specify a policy decision and physical target that cannot be mapped to
initial option choice, temporal/post-action credit, selective intervention, or
train-time auxiliary representation. Only then could a new family, budget,
matched Cm-off/action-shuffled/placebo controls, and a falsification gate be
written. No defensible GPU/time/storage budget can be assigned before that
mechanism exists; assigning a generic Probe budget would merely reopen the
frozen search.

The minimum user decision is therefore whether to change the high-level
research question/claim or accept that Cm policy utility remains unproven and
frozen. It is not a choice among another seed, feature, horizon, or threshold.

## Stop and handoff

No Probe, Validation, fit, collection, logging patch, PPO, online run, C1
rerun, HF06 registration, or Research Debt repayment should start from this
memo. If a later checkpoint cannot name a distinct policy decision and its
pre-action physical prediction before looking at results, stop immediately
with `NO_DISTINCT_DESIGN` again.

```text
BRANCH=agent/cm-selective-causal-gate
HEAD=0d4abd721dbb88224cf9bea1308c81084e889d5a
MAIN_REFERENCE=3767988eb34a62b23e2c9fc36b106e5f18c85470
STATUS=NO_DISTINCT_DESIGN
RESOURCES=CPU-only read-only synthesis; GPU=0; new evidence=0; no long process
NEXT=Option A; require a new user-authorized high-level goal and Decision Checkpoint before any Cm continuation
```
