---
schema: ref2dex.probe.v2
probe_id: P-20261009-old-utility-evaluator
experiment_id: P-20261009-old-utility-evaluator
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: edccfe1
claim_id: C3
hypothesis_family: HF-old-utility-evaluator
probe_index_in_family: 1
seed_pool: probe
seeds: [261, 262, 263, 265, 288, 289, 290, 292]
decision_changed_if_positive: quantify frozen PointWorld value gap and screen actual learned GT-selector execution
decision_changed_if_negative: audit source fit and horizon or interaction coverage without redefining old Y
status: PROMISING
run_id: old-utility-evaluator-20261009-r1
---

# Can GT and frozen PointWorld geometry recover the old teacher's action ranking?

Result: GT ranking screen PROMISING; C1.75641 versus C0.60256, C2.69231.
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

### Conditional actual selector Decision Note

The frozen fit completes1200updates each, val-selected steps C0=800/C1=1000/
C2=1200. On78strict pairs, C1 improves15.38pp and same-H future-shuffle loses
24.36pp; mean regret improves .07083→.05667. GT screen passes. C2 loses6.41pp
against C1 and misses the fixed .70/5pp retention screen; retain the model gap.
The informative support is only5anchors. A larger fit or claim is not warranted.

Root action: execute the already frozen first-query choices, each as one real
contiguous90step mixed path, to distinguish ranking-only signal from bounded
grasp rescue/harm. No new choice rule, extra fitting or model selection. Use
the existing pinned cold whole-world runner, recovered actor, query71,
96solver envs/25selected s3 rows and first8residual/then frozen feedback. C0,
C1 and observed-hand-oracle C2 all get the same budget. No new rolling queries.

Budget: one idleGPU2, <=240s per owned fork, <=720s total; outputs<=256MiB
additional. Exact historical prefix/geometry checks and full Z90/no-terminal
coverage must pass. Stop on mismatch, process/resource conflict or incomplete
continuation. Positive execution supports a new independent control Probe;
no benefit keeps the ranking signal but defers planner/bridge/RL. C2 cannot
be described as prospective or deployable. No external authorization needed
within current user/mission/campaign boundaries.

Actual artifacts are declared under the existing replay root
`outputs/cm-interaction-oracle/ref13-progress-recovery-20261009-r3/`
in fresh `evaluator-c0-one-shot`, `evaluator-c1-one-shot`,
`evaluator-c2-one-shot` folders/status/logs. New choices and the summary stay
in the consequence-evaluator run; existing replay data/manifests are preserved.

The final frozen audit uses <=60s GPU inference to replay saved panel scores
and source training fit. A post-freeze C1-with-PW-future swap holds its weights
fixed to diagnose the future/fit gap. It does not change any selector, gate or
actual execution; report it as a diagnostic, not a fourth tuned comparison.

## Completed results and next decision

Preparation commit70f40dd completes in19.67s,1024ordinary windows per split
plus175held candidates. Active ordinary windows128/1024per split. PW r2 uses
commit edccfe1 and frozen best46000,39.38s,peak allocated254.76MiB; all3247rows
complete. Same geometry and actual-hand oracle contracts apply to both arms.
Matched evaluator commit edccfe1 completes1200updates each in31.50s; reported
GPU memory485MiB. Val selects C0step800/C1step1000/C2step1200 before test.

| Metric on frozen25x7panel | C0 H+A | C1 GT24 | C2 PW24 oracle |
| --- | ---: | ---: | ---: |
| Strict pair accuracy (78pairs/5informative anchors) | .60256 | .75641 | .69231 |
| Same-H future-shuffle pair accuracy | — | .51282 | .56410 |
| Exact baseline-first teacher top1 agreement | .00 | .12 | .24 |
| Teacher tied-maximum set agreement | .84 | .88 | .92 |
| Mean regret | .07083 | .05667 | .06333 |
| Informative-anchor mean regret | .35417 | .28333 | .31667 |
| Informative Spearman | .24779 | .56862 | .44260 |
| Ordinary source263 test RMSE | .28352 | .25495 | .28293 |
| Ordinary source263 test Spearman | .80882 | .83604 | .79768 |

All median regrets are0 because20anchors are teacher-tied. Exact top1 is
especially sensitive to baseline-first teacher ties; report tie-set agreement
and regret alongside it. GT ranking screen passes (+15.38pp vsC0,24.36pp
shuffle loss). PW retention screen does not (69.23%,6.41pp belowC1). This
is PROMISING for GT-geometry teacher ranking, not formal Gate1 or Cm benefit.

| Actual one-shot Z90 | Count/25 | Rescue vsbaseline | Harm vsbaseline |
| --- | ---: | ---: | ---: |
| Frozen actor baseline | 20 | — | — |
| C0 | 19 | 0 | 1 |
| C1 GT oracle | 20 | 0 | 0 |
| C2 observed-hand PW oracle | 21 | 1 | 0 |

Three actual mixed paths complete in39.70/39.60/40.95s. All prefix contracts,
raw model-H exactness, object/hand geometry prefixes, frozen requested plans,
zero clipping and complete90nonterminal samples pass. Saved evaluator scores
replay exactly (all maxerrors0), audit1.93s. C2 gain is4pp with descriptive
paired bootstrap CI[0,12]pp; C1 vsbaseline0; C0−4pp CI[−12,0]. Outcome support
is single exposed shared-world cohort; actual control benefit remains UNCLEAR.
These are one-shot paths, not the old teacher's three-replan23/25result.

Post-freeze diagnostic: the **same C1 weights** given PW futures achieve
.73077pair accuracy and .05958mean regret, versus .69231/.06333 for separately
PW-trained C2. This diagnostic was not used to choose a checkpoint or execute
a fourth policy. It prevents attributing the entire6.41pp gap uniquely to PW
accuracy. Source-train RMSE is .03527/.02758/.02192 versus ordinary test
.28352/.25495/.28293; source-to-held generalization remains a concern despite
the clear held candidate GT signal. PW candidate h24 translation/rotation
error is32.76mm/.409rad; source ordinary errors are larger.

Root decision: keep **U32 fully frozen** and preserve the useful GT evaluator
and PW artifacts. Do not restart Y definition, automatically extend epochs,
declare PW inadequate, or advance to deployable planning/RL from these data.
The next cheapest discriminating comparison is an independently frozen test
using one common C1 evaluator with GT vsPW futures, keeping oracle-hand and
source/actor boundaries explicit. New independent candidate/cohort support is
needed before upgrading the post-freeze swap signal or choosing a planner.
Rolling control remains future evidence: the lack of one-shot C1 rescue does
not close a route whose old teacher evidence came from repeated replanning.

Artifacts:
- `outputs/consequence-evaluator/old-utility-data-20261009-r1/`
- `outputs/consequence-evaluator/old-utility-pw-20261009-r1/` (failed engineering,
  retained diagnostics) and `old-utility-pw-20261009-r2/` (completed)
- `outputs/consequence-evaluator/old-utility-evaluator-20261009-r1/`: frozen
  models, manifest/result, panel scores, fixed plans and `control-result.json`.
- Three actual paths/status/logs at the replay root declared above.

Five focused unit/interface tests pass; all runs remain below budgets.
Total new evidence output is below512MiB; GPU2 is released. No external writes,
PW/actor fitting, new branch, remote push or checkpoint overwrite occurred.

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
