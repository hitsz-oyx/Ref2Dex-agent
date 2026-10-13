---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-actor-curation-ab
experiment_id: P-20261013-goal-point-wam-actor-curation-ab
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: a2867e8a
claim_id: C3
hypothesis_family: HF-goal-point-wam-curation
probe_index_in_family: 2
seed_pool: probe
seeds: [299]
decision_changed_if_positive: qualify actor-specific weighting on robot contact panels before any larger training
decision_changed_if_negative: stop this decoder-only weighting recipe and prioritize target semantics and reference-derived interaction goals
status: UNPROMISING
run_id: actor-curation-ab-20261013-r1
---

# Matched decoder-only actor weighting Probe

Result: UNPROMISING for this fixed decoder-only recipe: final B improves the
six-panel first8 metric only0.0893% over A, below the predeclared5% gate.
Decision: retain the original model and both Probe artifacts; stop weighting
scale-up. Planned human wrist goals are exported and geometrically checked,
but not admitted as robot execution targets. Prioritize reference conditioning
with own-state/R coverage rather than more native decoder weighting.

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

## Completed result and attribution

Engineering smoke4 updates completed6.72s. Scientific run1000 updates per arm
completed187.56s GPU2, code `a2867e8a`. Frozen context preparation88.97s TRAIN
and13.61s VAL; actual neural training/periodic evaluation about80s. Memory
about4.63GB; observed utilization up to86%. Two arms saw identical96,000
draws from9216 TRAIN examples,1358 suspect pool occurrences;14.7208% of
actual draws were downweighted in B. World/scene/goal/observed-hand encoder
state and buffers compared to parent: max difference0. No TEST or rollout.

| Primary metric: mean of six source x interaction first8 EPE panels | mm |
|---|---:|
| Original step10000 | 14.74733 |
| A: ordinary continuation | 14.67368 |
| B: actor downweight.1 | 14.66057 |

B-A is -0.01311mm (-0.0893%). At250/500 the two arms were also close;
final1000 is used as fixed in advance. The primary5% gate fails, while all
six source/interaction guards and all three manipulation-goal-utility guards
pass. A small positive sign alone does not justify scale-up.

| VAL first8 EPE mm | A | B |
|---|---:|---:|
| OakInk2 manipulation | 3.99969 | 3.96959 |
| OakInk2 static-goal near | 4.92290 | 4.92230 |
| GRAB manipulation | 14.54194 | 14.54080 |
| GRAB static-goal near | 30.41082 | 30.39303 |
| ARCTIC manipulation | 15.61143 | 15.59907 |
| ARCTIC static-goal near | 18.55530 | 18.53865 |

OakInk2 far-moving first8 slightly worsens6.30828->6.35581mm, all24
22.06055->22.21331mm. Its small departure stratum(n10) also worsens
7.04117->7.20620mm first8. All remaining source/phase metrics, including
GRAB departure n114 and ARCTIC departure n24, are retained in `result.json`.
The observed changes are small; no statistical significance is asserted.

Independent saved-array audit passes: direct threshold reconstruction exactly
matches all suspect flags; shared target/mask arrays are identical; metric
recomputation max difference<0.000049mm. Both decoders actually changed from
parent (max parameter differences.00393/.00357), and A/B differ by up to.00200.
Thus the null practical result is not explained by accidentally identical
weights or a disconnected optimizer. Two focused weighting tests pass,
including equivalence to the original objective for unit weights, invariant
overall weight scaling and zero-weight gradient exclusion.

Scope: this cannot refute curation from scratch or shared-encoder adaptation.
It supports stopping the particular cheap decoder-only remedy. Per the
predeclared positive-only downstream gate, no robot promotion or extra rollout
is warranted. Existing R own-state recovery limitations remain separate.

Artifacts: `outputs/goal-point-wam/actor-curation-ab-20261013-r1/` contains
launch manifest, fixed inputs/draws, evaluation history, both decoder-only
checkpoints, both prediction/target arrays, result and `independent_audit.json`.
The initial producer left manifest status RUNNING despite completed result;
status-only reconciliation follows completed result, and future producer is
fixed. No numeric result or checkpoint was changed.

## Reference-goal admission and useful historical evidence

User explicitly permits some reference demonstration. The existing planned
reference `outputs/trajectory-policy/competent-reference-20261011-r1/` has
543 frames, human16 positions/orientations and object poses. Independent raw
field checks confirm right wrist=human point0=raw51:54; human/object rotation
max differences<4e-7. Exported +8/+16/+24 planned wrist position/orientation,
object pose, active-hand identity, validity and desired-object-relative wrist
geometry. Relative reconstruction error<5.4e-7m and shared-world-transform
equivariance<1.3e-7m. The live transform must use measured current object pose,
not assume the planned current pose is the robot's actual pose.

Output `outputs/goal-point-wam/planned-wrist-goal-20261013-r1/`, exporter commit
`73bbee4e`, engineering-only / training_allowed=false. It contains no measured
future state, future robot q/action or teacher commands. Human wrist is not
Inspire hand_base_link: comparison to nominal teacher geometry gives median
point separation77.62mm (different landmarks, not a tracking-error or
feasibility verdict). Do not copy this coordinate directly as a robot command.
Reference timing is fixed-frame with terminal clamping; event alignment is
not implemented or claimed solved.

The existing reference-conditioned DAgger validation was re-read and both
saved s400/s401 result summaries checked: each has joint hold/place8/8, median
held376/405 and zero clipping, at the same fixed initial condition/reference.
That route uses rich H328/G408 and visited-state PD correction, a different
actor/R setup; it is not evidence that wrist-only conditioning alone suffices.
It supplies a qualified reference-conditioned baseline to compare against,
not a reason to attribute all prior failure to missing goals.

Raw TRAIN geometry examples are exported to
`outputs/goal-point-wam/curation-examples-20261013-r1/`. Four examples are the
median hand-peak member of their geometric stratum, with source/sequence/
anchor/tick recorded. They illustrate background-target/other-active-object,
idle-away, GRAB departure and ARCTIC near-at-both-ends. Visual inspection
agrees with those geometric labels; semantic human intent remains unannotated.
No examples were used to select training/validation rows or thresholds.
