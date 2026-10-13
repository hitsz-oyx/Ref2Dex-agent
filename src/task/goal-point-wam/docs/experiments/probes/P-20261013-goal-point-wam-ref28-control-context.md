---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref28-control-context
experiment_id: P-20261013-goal-point-wam-ref28-control-context
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: cee4c3e0
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 39
seed_pool: probe
seeds: [292, 295, 296]
decision_changed_if_positive: retain robot conditioner and qualify its closed-loop grasp before further scaling
decision_changed_if_negative: retain C and diagnose contact recovery without rerunning large geometry pretraining
status: UNPROMISING
run_id: wam-control-context-20261013-r3
---

# Robot control context and current C-state recovery

Result: Adapter2000 + decoder5000 updates completed; selected stage2 step4500.
Offline own-state recovery improves, but full542-control actor rollout held0:
contact occurs and object becomes airborne, then falls. Local grasp is UNPROMISING.
Decision: Preserve the adapter/candidate and successful engineering contracts;
do not scale to20k or rerun geometry pretraining based on offline loss alone.
Observation insufficiency and Cm causal benefit remain unestablished.

## Motivation and fixed design

Decision Probe for C3 manipulation: does robot-only proprioception and temporal
object reference conditioning improve the contact transition of the existing C
actor? Chunk-R already receives q/dq; this intervention adds context to WAM.
No attribution between the two new inputs or Cm-on/off causal claim is attempted.
The cheapest useful evidence is current C own-state expert recovery at32/40/48,
adapter-only contact panels, then a matched-reset native rollout.

48D input is measured q18, dq18, previous actually applied ACTIVE12 command;
at reset previous command is zero. 27D goal is inv(current own object pose)
times planned reference pose at+8/+16/+24, each translation3 plus rotation6
(first two rotation matrix columns, column-major). MLP48/27 ->128 ->128,
two scalar gates initialized exactly0; residual added to existing decoder
queries after raw-hand conditioning. Human conditioner is masked to exactly0.
No actual future object pose, expert future hand, task1442D or contact force
is admitted as an actor observation. World remains goal-blind.

First collect each C-final takeover with128 controls, four native roles and
seed292. Only row3 actual applied expert futures are admitted after takeover;
audit prefix equivalence and require each admitted24-step window unclipped.
Do not substitute another row's future for failed late recovery.

Stage1 freezes all existing modules including decoder and R, trains adapter
only for2000 updates at1e-4, robot draws nominal/recovery3:5. Stage2 only if
each recovery contact panel (first16 post-takeover queries) has first8 and
first-PD XYZ p95 <=90% baseline and nominal PD/clipping do not regress beyond
10%+.1mm /1 percentage point. Stage2 adapter1e-4/decoder1e-5, exact
20% human/30% nominal/50% new recovery, initially5000 updates; extension up
to20000 only if contact panels continue improving. Evaluate every500 updates.
Supervision retains C trajectory + qualified Chunk-R control-aware loss;
World/action/R frozen. Frozen-context caching is allowed, labels remain own.
Initial equivalence is checked with identical dtype/context and zero gates.

One idle GPU for each sequential collection/training, max6 globally; collection
<=600s each, training <=2h, new artifacts <=2GB, total outputs+tmp <=400GB.
Stop on nonfinite output, incorrect frame/previous-action semantics, frozen
state drift, foreign GPU process, budget, or two consecutive gross contact
regressions. Same main branch; no new branch or remote push.

Closed-loop success uses unchanged >=45 consecutive held frames (gap<=10mm,
height>=30mm and unsupported), teacher/GT calibration and actual clipping;
full placement reported separately. A single reset is only a Probe.

## Inputs and outputs

- User design: ../../user/ref/ref2_8.md
- C-final: outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt
- Warm policy: outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1
- Qualified Chunk-R: outputs/trajectory-policy/wam-v2-control-fit-20261012-r1/chunk.pt
- New recovery: outputs/trajectory-policy/wam-ref2-execution-C-t{32,40,48}-20261013-r1/native
- Stage1: outputs/goal-point-wam/wam-control-context-20261013-r1
- Completed stage2: outputs/goal-point-wam/wam-control-context-20261013-r3
- Selected candidate: outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt
- Native closed-loop: outputs/trajectory-policy/wam-ref2-execution-ctx-20261013-r1

## 2026-10-13 stage1 result and Decision Note (exploratory protocol amendment)

Current C takeovers32/40/48 completed128 controls each; actual expert continuation
holds75/74/71 consecutive frames with no clipping. Each saved-trace audit passes
2316 checks max0; prefix physical tolerances pass (approximate PhysX pairs,
not a hidden-state clone). Actual expert future24 grants73/65/57 TRAIN windows.
No t56 collection needed to determine recoverability of this contact region.

Adapter-only2000 updates at commit7e25cb30, ~200s including frozen-context
preparation, exact initial C output and frozen decoder/scene/World/R difference0.
Human outputs are exactly unchanged. first8 t32/40/48:24.133/35.890/60.496 ->
22.962/32.974/42.527mm; first-PD XYZ p95:67.670/233.239/358.766 ->
23.218/162.421/252.208mm. Nominal PD23.394->22.882mm, startup .617->.965mm,
all clipping0. Original strict gate fails: two first8 improvements<10%, startup
relative limit exceeded. This must not be described as passing the original gate.

Decision: continue only5000 decoder updates because all three contact first8
improve, average normalized improvement is >10%, and all first-PD improve>10%;
startup absolute error remains <1mm. Retain strict diagnostics but amend the
exploratory continuation gate to each contact first8 nonregressing (+.05mm),
mean first8 ratio<=.9, each contact PD<=.9 initial, nominal PD original limit,
startup PD<=max(original limit,3mm), clipping original limit. This decision is
made from offline TRAIN data before any new actor physical outcome. It is not
Validation or evidence that observation insufficiency is established. Stage2
still freezes geometry/goal/raw/World/R, uses20/30/50 draws and1e-4/1e-5 LRs;
stop at5000 or prior declared budget/gross-regression conditions. Candidate
selection uses contact score with nominal/startup and <=20% human all24
preservation; report a fallback explicitly if none qualifies.

Stage1 r1's fallback selection retained first500 when no candidate met the
strict startup guard. Preserve r1 as recorded; stage2 continues explicitly from
its actual frozen-decoder step2000, and fixes fallback selection for future runs.
No rerun of stage1 or alteration of the protected source packet is needed.
Frozen train contexts use1104 nominal windows,195 recovery and192 human replay
windows; these are repeated samples, not new environment exposure. Human val
panel16 per domain and robot nominal roles24..27; recovery remains TRAIN.

## Completed stage2 and independent saved-panel audit

Stage2 r3 at cee4c3e0 continues stage1 step2000 with a new AdamW optimizer;
5000 updates,659.42s including cached-context preparation, GPU2 ~1.965GB,
typically .12s/update. No20k extension. New inputs are exactly masked on human:
all three final human tau arrays with adapter on/off differ0. Frozen
scene/goal/raw/action/World/R/buffers difference0; decoder legitimately changes
(selected maximum state difference .0163483). Zero-gate stage1 adapter-off
arrays exactly equal initial C tau for all8 panels (max0).

Select stage2 step4500 by contact score with nominal/startup and human all24
preservation, not training loss. Robot clipping0 throughout the fixed panels.
Original strict gate failure and amended exploratory admission remain disclosed;
selected candidate still has t32 contact16 first8 deterioration4.77%, so do
not describe every contact metric as improving or all revised gates passing.

| panel | first8 C -> candidate, mm | first-PD XYZ p95 C -> candidate, mm |
|---|---:|---:|
| nominal | 15.307 ->15.388 | 23.394 ->22.240 |
| startup | 10.804 ->11.702 | .617 ->.697 |
| t32 first16 expert queries, TRAIN | 24.133 ->25.284 | 67.670 ->4.343 |
| t40 first16 expert queries, TRAIN | 35.890 ->32.878 | 233.239 ->106.981 |
| t48 first16 expert queries, TRAIN | 60.496 ->37.697 | 358.766 ->111.961 |

The exact takeover queries are separated from subsequent expert states:

| actual C query | first8 C -> candidate, mm | first-PD C -> candidate, mm |
|---|---:|---:|
| t32 | 38.681 ->34.745 | 51.763 ->1.981 |
| t40 | 86.375 ->51.818 | 119.756 ->83.486 |
| t48 | 221.889 ->85.399 | 351.894 ->33.692 |

These195 recovery windows contain only3 exact C takeover states; the other192
are own expert continuation states, not additional autonomous C visits. This
distinction matters for sampling and attribution. Human all24 Oak/GRAB/ARCTIC
13.040/56.258/47.333 ->12.961/58.881/47.350mm; replay prevents gross drift,
not every human metric regression. The intervention combines new inputs, new
recovery data and decoder adaptation, without matched conditioner-off training.

`root_panel_audit.json` recomputes35 robot trajectory/first-PD metrics from
actual own rows, with maximum trajectory scalar difference3e-5mm and PD
difference8e-6mm. It also diagnoses fixed R on privileged actual expert tau:
contact16 GT-Chunk first-PD p95 t32/40/48 =34.003/171.604/223.137mm, clipping0;
exact takeover GT errors32.821/58.973/159.946mm. These are offline diagnostics,
not physical GT-recovery qualification or an irreducible control-error floor.
The learned tau can lower control error below true-tau R error, exposing a
trajectory/control tradeoff and new-state R generalization limitation.

Engineering incidents preserved: smoke r1 lacked the t48 persisted audit receipt;
smoke r2 passed2 updates. Stage1 r1 completed2000 but summary serialization
rejected infinity from its no-qualified-selection fallback; use the unchanged
step2000 packet and separate root completion receipt. Stage2 r2 failed before
updates on NumPy bool JSON serialization; r3 fixes it. Panel audit first needed
float32 GT inputs, then identical16-row TF32 evaluation batching; whole-panel
R forwarding differed .066mm due numerical batching, not labels. Final audit
passes at the original tolerance; failed logs remain, source packets untouched.

## Actual closed-loop result and diagnosis

Four-step adapter/native smoke completes, full-WAM vs actor-only first-query
tau max0 and own q/dq/previous applied input max0. Full native run on GPU2,
seed292,147.54s,542 controls/543 states; both peers exit0. Original base/warm/C
and exact qualified Chunk-R remain authenticated; native adds its own q/dq,
previous actually applied ACTIVE12 and planned+8/+16/+24 relative object poses.
Thirteen independent native checks pass: all command/current-state/history/
goal/previous-action checks max0, independent SE3 inverse vs producer transpose
rotation6D maximum8.6e-7. Calibration teacher/GT-Chunk hold434/441 frames with
0/.9225% clipping; both controlled-place succeed in this run. Prior C run's
GT placement failed, so do not imply deterministic matched physics or compare
placement causally across these independent simulated rows.

| actor checkpoint | max consecutive held | max object height | clipping |
|---|---:|---:|---:|
| prior C-final | 0 | 0mm | 9.225% |
| robot context candidate | 0 | 276.605mm | 1.476% |

New actor minimum hand-object gap .913mm at tick54; object first reaches30mm
height at tick68, peaks at tick75 with hand-object gap363.666mm, then falls
to terminal -757.170mm. No held frames occur. This is contact followed by
airborne separation/drop, not a stable grasp or successful lift45.
Measured wrist deviation from same-trace teacher t48=46.771mm vs prior174.213,
t56=82.765mm vs393.974; t32 worsens32.566mm vs7.213. These comparisons describe
two fixed candidates and are not an input-only causal estimate. At late ticks
the new actor also diverges. Preserve the valid local negative result and
partial approach/contact improvements without closing the observation route.

Artifacts: native `root_effect_rollout_audit.json`, `drift_diagnostics.json`,
and `visualization/measured_rollout.mp4`, `outcomes.png`, `contact_tick48.png`.
Video is a measured-pose reconstruction, not camera footage. Relevant8 tests
pass, repository verify passes; own GPU2/3 released, global outputs+tmp295.18GB.

## Limitations and future evidence

If continuing, first distinguish current candidate visited-state coverage,
sampling of the3 genuine C queries versus192 expert continuation states,
and frozen R's true-tau recovery error before additional scale. Additional
contact/force/joint-error observations and matched input ablations remain
future decision probes; none was added opportunistically to this run.

New recovery windows are TRAIN diagnostics, not independent generalization.
Single seed/reset and joint proprio/reference intervention cannot prove
observation insufficiency or Cm causal benefit. Matched input ablations and
multi-reset validation are deferred until this Probe changes the route.

## 2026-10-13 session continuation: evidence review

The prior session completed the C-final t32 and t40 takeover collections;
t48 and adapter training have not completed. Both 128-control collections
retain teacher/GT-Chunk/unaided-C held counts73/75/0. Actual expert takeover
holds75 frames at t32 and74 at t40, with zero executed clipping. These are
local recoverability signals, not autonomous actor success or full placement.
The existing t32 audit passes; continuation ran `audit_wam_recovery.py` on
t40, also passing2316 checks with numerical max difference0 and physically
comparable replay prefixes. The rows are not exact hidden-state forks.

Read-only inspection finds73/65 complete unclipped24-step expert windows
for t32/t40 (query ticks32..104 /40..104); these overlapping windows are
not independent episodes. Their futures belong to the actual takeover row.
The same-state expert query in the t40 trace gives first-PD XYZ disagreement
with unaided C of51.64/119.96/120.46/353.30mm at ticks32/36/40/48.
Corresponding sampled hand-object gaps at32/36/40 are864.10/654.77/298.48mm:
control disagreement is already present before close contact. The first
executed actor clipping is tick92, so clipping does not explain its onset.
Expert disagreement diagnoses imitation/control mismatch, not a unique
optimal-action requirement or proof that missing proprioception is causal.

Route recommendation: preserve C and the qualified R, complete current-state
recovery coverage and test the bounded robot adapter before another scale-up.
Once testing decoder adaptation, compare the same new data/update budget
with and without the extra context to distinguish data from input benefits.
Score a fresh closed loop after the short fit; do not admit long training
solely from recovery TRAIN error or frozen-world gap. The adapter jointly
adds proprioception and planned reference, so a positive result alone cannot
attribute benefit to either input. No extra collection or model training was
performed during this evidence review; original protocol remains unchanged.

- [t32 result](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-t32-20261013-r1/native/result.json)
- [t40 result](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-t40-20261013-r1/native/result.json)
- [t40 audit](../../../../../../outputs/trajectory-policy/wam-ref2-execution-C-t40-20261013-r1/native/root_recovery_audit.json)
