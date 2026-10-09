---
schema: ref2dex.probe.v2
probe_id: P-20261009-old-utility-evaluator
experiment_id: P-20261009-old-utility-evaluator
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 76e03db
claim_id: C3
hypothesis_family: HF-old-utility-evaluator
probe_index_in_family: 1
seed_pool: probe
seeds: [261, 262, 263, 288, 289, 290, 292]
decision_changed_if_positive: quantify frozen PointWorld value gap and screen actual learned GT-selector execution
decision_changed_if_negative: audit source fit and horizon or interaction coverage without redefining old Y
status: UNCLEAR
run_id: old-utility-evaluator-20261009-r1
---

# Can GT and frozen PointWorld geometry recover the old teacher's action ranking?

Result: pending.
Decision: user ref4_4 freezes old U32 and authorizes matched C0/C1/C2 evaluator
training and held-panel ranking. Stop the Y-definition detour. The previous
temporal-label Probe remains separate; its model/label are not reused here.

## Motivation and Decision Note

Mission C3 needs useful action-conditioned consequences. This Decision Probe
distinguishes whether GT geometry24 improves old-U32 candidate ranking over
H+A, and how much a fixed PointWorld endpoint preserves. It does not prove
formal horizon insufficiency, representation insufficiency or RL benefit.
No further teacher-label qualification is required by the current user route.

Cheapest source is the192 preserved full dose-intervention trajectories: source
261train,262val,263ordinary-test,64episodes each. Historical intervention files
from the deleted baseline workspace are not available. These new episodes use
the official actor only as a supervision generator. The final screening panel
uses the reconstructed self-trained e260 actor, preserved ref13 whole-world
contract,25current-only s3 anchors at tick71 and seven known residual choices.
The entire panel is excluded from fitting, statistics and val selection. Its
labels have been exposed by earlier experiments; this is a frozen screening
cohort, not a sealed Validation holdout.

User clarification authorizes **offline oracle C2**: condition PointWorld on
observed future hand points, replace only object future. This is not a native
control-plan-to-hand predictor and not a deployable planner. Do not silently
rename post-treatment hand trajectories as a decision-known proposal.

## Frozen contracts

- Teacher exactly inherited `short_y` with32complete future height/force-pair
  states, scalar `Y7+.25Y3-Y6`. No coefficients or persistence cutoffs change.
- Future model inputs only24steps, geometry45: relative object effect12 plus
  measured hand11x3 in each future object frame. Future contact enters labels
  only, never C1/C2 or PW inputs. All arms have the SAME raw native H1442 and
  decision-known requested residual plan24x18. Executed feedback actions are
  not exposed as A.
- Ordinary data starts at tick>=3 for measured four-state PW history. Stride8
  eligible known-plan windows plus perturbation onset; complete32future only.
  All active windows retained, then seed289 fills to16windows per episode
  without inspecting Y. No episode outcome/recovery labels enter this teacher.
- Native contact is hand-net-force>.1 and object-net-force>.1, identical to
  historical force-pair semantics. Rest is the fixed full-reference frame0
  object height; confirm against recorded initial height.
- C0/C1/C2 identical128wide/two-layer/one-scalar backbone and initial seed288.
  MSE on frozen U, AdamW3e-4, batch128,1200updates each, same batch schedule.
  Half active/half inactive plan samples, episode-uniform then window-uniform.
  H and future statistics use train only; future std floor.001. A uses fixed
  native residual unit.5 for all arms, avoiding dilution-driven tiny std.
- Select each checkpoint only by val episode MSE. Freeze all three before
  opening ordinary source263 and ref13 panels. No auxiliary Y-head or tuning.
- PW freezes main-three best46000; input object surface512 and actual native
  hand points with native interleaved semantic order mapped to wrist/MCP/tips.
  Object history uses one fixed current-object frame, right hand only with
  absent left masked. Native collision mesh scale1, four past/current states,
  measured future hands24. No GT future objects or outcome flags in forward.
  Both C1/C2 retain actual hands; C2 interactions are recomputed relative to
  PW object transforms. C2 is independently fit with PW futures.

## Metrics and gates fixed before fitting

Report ordinary MAE/RMSE/Spearman and held same-H strict pair accuracy (old
epsilon.02), baseline-first top1 agreement, tie-set agreement, per-anchor and
mean/median regret. Test has78strict pairs on5informative anchors; all25anchors
and tie support remain visible. Predictions within1e-7 are half-credit ties.
Inference panel batches are multiples of seven.

Future shuffle within each same-H panel deranges seven candidate futures with
seed290, keeping H/A fixed. Exploratory GT screen: C1 pair accuracy>=.70,
gain over C0>=.03, shuffle drop>=.03, mean regret no worse than C0. PW retention
screen additionally C2>=.70, C1-C2<=.05, and C2 regret<=C1+.05. These screen
thresholds do not establish formal support or unique learned Cm contribution.
Failure is UNCLEAR for a method; source/transfer fit must be inspected first.

## Resources and stopping

Fresh task-owned outputs: old-utility-data-20261009-r1,
old-utility-pw-20261009-r1, old-utility-evaluator-20261009-r1.
One idle GPU2 sequentially. CPU raw-label/provenance preparation<=120s;
PW inference<=900s, batch<=16; matched evaluator fit<=600s,1200updates each;
outputs<=512MiB excluding read-only source checkpoints. Monitor utilization,
memory and elapsed/ETA; stop on input drift, missing complete32labels,
nonfinite tensors, clock/frame/order mismatch, foreign GPU use or deadline.
No PW/actor retraining, remote push, new branch or external writes.

If the GT screen passes, a separate bounded Decision Note may authorize actual
one-shot selector Z90 paths within this same Probe; no fork mosaics count as
rolling outcomes. C2's observed-hand limitation remains. New same-H data,
full-task suffixes, PointWorld adaptation/execution bridge and RL follow only
after these comparisons resolve the useful next step.

## Limitations / future evidence

### Engineering repeatability repair before evaluator fitting

The first PW inference run stops before generating usable futures: CUDA atomic
`mean_groups(index_add_)` differs between identical native-geometry batches.
Measured batch8 AMP translation max171.5micrometres and rotation-entry max.000300;
RNG reset does not fix it, eval shuffle is off and input tensors are unchanged.
FP32 alone also leaves variation. A bounded1/8batch AMP/FP32 feedback loop
isolates unordered group accumulation. Process-local sorted CSR arithmetic
mean removes differences exactly in all four cases, preserving frozen weights,
coordinate definitions, feature/parameter shapes and untouched PW source files.
Group/empty-group test passes. Use this deterministic evaluation backend for
r2; preserve failed r1, original/stable diagnostics and their numerical data.
No teacher, actor, model objective or PW training changes. This is a Blocker
repair, not a negative result for the research method.

Human-to-robot PW domain shift, official-to-e260 actor/backend/plan-shape shift,
only five informative test anchors, shared PhysX world and ties limit this
Probe. Local U32 and Z90 do not equal full controlled placement success.
Teacher reproduction is not final MissionB. Actual mixed execution and formal
matched training-policy Validation remain needed. Do not infer that C1 failure
uniquely identifies horizon/contact insufficiency or that C2 degradation
uniquely identifies numerical PW accuracy; source fit and transfer may confound.
