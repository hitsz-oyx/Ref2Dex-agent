---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-actor-curation-ab
experiment_id: P-20261013-goal-point-wam-actor-curation-ab
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 4ca2baeb
claim_id: C3
hypothesis_family: HF-goal-point-wam-curation
probe_index_in_family: 2
seed_pool: probe
seeds: [299]
decision_changed_if_positive: qualify actor-specific weighting on robot contact panels before any larger training
decision_changed_if_negative: stop this decoder-only weighting recipe and prioritize target semantics and reference-derived interaction goals
status: UNCLEAR
run_id: actor-curation-ab-20261013-r1
---

# Matched decoder-only actor weighting Probe

Result: IN PROGRESS.
Decision: execute the user-approved short comparison and audit a legitimate
reference-goal source separately; no joint intervention in the two arms.

## Motivation and fixed design

Decision Probe for goal-conditioned hand prediction upstream of C3. The
previous full TRAIN census found14.70% expected mixed exposure in static-goal,
far-moving hand windows, mostly OakInk2 background anchors. Test whether
reducing their actor supervision improves useful interaction prediction.
The cheapest intervention freezes all encoders/action/world and adapts only
the trajectory decoder. This does NOT test encoder learning or joint world
training; world labels/interface remain intact, world parameters/buffers are
unchanged, and its loss is constant with respect to the only trained module.

- Both arms start from original local step10000:
  `outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt`.
- Frozen main-three dataset/stats, existing source/category sampler. Fixed
  TRAIN pool9216 with seed299; fixed original VAL1536, no TEST. Repeated
  draws from this bounded pool are a Probe limitation, not new demonstrations.
- A: all actor weights1. B: weight.1 ONLY for OakInk2 g<2mm, object peak<5mm,
  hand peak>=5mm, target remains>100mm. Existing audited classifier preserves
  contact/approach before far classification. All other weights1.
- Per-valid-point Huber using original scale.05m, weighted numerator divided
  by weighted valid-point count. A is exactly the original actor objective;
  B is normalized so removing weight does not merely shrink the whole loss.
- Cache frozen scene/goal/observed-hand contexts once in GPU RAM. Identical
  context and batch indices for both arms; identical per-step RNG stream.
  No cached prediction or future labels enters the observed context.
- Fresh AdamW LR1e-5, wd.01, gradient clip1, BF16, batch96, exactly1000 updates
  per arm. Report0/250/500/1000; use final1000 for decisions, no best-checkpoint
  selection, LR sweep or budget extension.
- Primary signal: mean first8 EPE across six source x (manipulation,
  static-goal-near) VAL panels, B at least5% below A. Guard: no panel degrades
  more than5%+.1mm; retain actual-goal advantage over static goal in all three
  manipulation all24 panels. Report all24/all-source, far-hand and geometric
  hold/departure strata separately, including regressions and small n.
- Positive means PROMISING only; qualify downstream robot panels before
  scale-up. Otherwise stop this fixed recipe as UNCLEAR or locally UNPROMISING;
  no conclusion against all curation or core Cm. No closed-loop claim.

## Resources and stops

One idle local GPU2, <=30min total, <=1GB new disk output; caches RAM only.
Global GPU use<=6 and outputs+tmp<=400GB; other users' GPUs are untouched.
Stop on input drift, nonfinite values, OOM, freeze/matching failures, foreign
GPU conflict, budget, or two consecutive evaluations with B primary>1.25*A.
Initial smoke uses the same executable in an engineering-only small run.
Code committed before launch. Follow long-running-tasks monitoring for GPU,
memory and ETA; waits remain bounded by session communication requirements.

## Reference-goal scope authorized by the user

User permits some reference demonstration. Inspect the existing planned
human/object reference independently and export a small, explicitly planned
wrist/palm/approach-goal packet only if semantics and coordinates are sound.
Do not feed measured future expert/learner states or future robot commands.
This reference audit does not change either A/B arm, and is not evidence of
robot feasibility or successful reference-conditioned control.

## Limitations / future evidence

One seed, fixed bounded TRAIN pool, decoder-only adaptation, no robot training
or new rollout. The validation panel was already used diagnostically, so this
is development evidence, not held-out formal Validation. The prior corpus
coverage gap and robot R own-state errors are separate blockers; neither is
resolved by better native EPE. Reference goal source and time alignment must
be explicitly admitted before training an augmented actor.

Related: [full census](P-20261013-goal-point-wam-ref3-curation-audit.md),
[goal contract](../../research/2026-10-13-object-and-hand-goal-contract.md).
