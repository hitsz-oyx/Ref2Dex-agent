---
schema: ref2dex.probe.v2
probe_id: P-20261008-consequence-pair-coverage
experiment_id: P-20261008-consequence-pair-coverage
date: 2026-10-08
task: consequence-evaluator
branch: consequence-evaluator
git_commit: e62dd08
claim_id: C3
hypothesis_family: HF-consequence-oracle-headroom
probe_index_in_family: 2
seed_pool: probe
seeds: [296, 297, 298, 299]
decision_changed_if_positive: use the unchanged local-state preference contract for a fresh matched evaluator fit
decision_changed_if_negative: keep evaluator fitting paused and redesign collection around explicitly repeated current states
status: UNCLEAR
run_id: pair-coverage-20261008-r1
---

# Can repeated waves recover strict current-state preference coverage?

Result: The historical three-wave label report was `READY` under the then-current
nonempty-split rule, with only 2/1/5 train/val/test pairs. Under the current
contract it is observational-only: pair matching also requires history `H`, and
fit requires at least 8/4/4 distinct episode pairs. Physical geometry remains
valid, but the sample is not fit-ready.
Decision: Stop before fitting and keep the route gate closed. The user has
explicitly retained ref2 continuous rollout without forks; twin is engineering
diagnosis only. Diagnose continuous phase coverage without relaxing state
thresholds or adding generic waves.

## Purpose and decision

The first six-route collection produced 0/0/2 train/val/test local preference
pairs under the predeclared state contract, despite a valid geometry audit.
This Probe tests the cheapest remaining explanation: two waves did not repeat
the same motion/phase/current state often enough. It keeps the contract fixed:
24-step decision-known residual plans, object translation <=3 cm, object z <=1
cm, rotation <=15 degrees and measured 11-point hand RMS <=2 cm.

The result changes only whether a fresh evaluator fit is eligible. It cannot
change the global Cm claim, promote a weak specialist, or justify relaxing a
state threshold after seeing the labels.

## Protocol and resource boundary

Use the frozen route and corrected native motions from the first collection.
Collect three waves with 24 environments for each of train/val/test, using
seeds 296/297/298, amplitude 0.08, the same 1200-step cap and one GPU at a
time. This produces at most 72 episodes per split and preserves seed-group
isolation. Label with the existing abstaining rule on CPU.

The required gate is nonempty train and val pairs under the unchanged rule,
plus valid native geometry diagnostics. If either split is empty, stop before
any model fit and record the raw artifact for future targeted annotation. If
both are nonempty, freeze the source/label hashes and write a separate fit
Decision Note; no fitting is authorized by this card alone.

## Outputs

Collectors write to `outputs/consequence-evaluator/continuous-20261008-pair-{train,val,test}-r1/`.
Labels write to `outputs/consequence-evaluator/labeled-continuous-20261008-pair-all-r1/`.

## Result and stopping Decision Note (2026-10-08)

All three collectors completed 72 episodes each. Their manifests are
`COMPLETED`, use commit `e62dd08dcc601282e84c0c49bc860e29125f6bc3`, and stay
within the 900 s / 2 GiB collector budget. The CPU labeler accepted all 216
episodes and reports `READY`; train/val/test pairs are 2/1/5 under the
unchanged contract. The geometry audit has 149,841 valid frames, 49,388
native proxy frames, 45,984 proxy-and-near frames, and only one proxy-far
frame. Repeated waves therefore repair the empty-split gate without exposing a
native geometry failure.

The fixed-contract diagnostic is
`outputs/consequence-evaluator/pair-coverage-diagnostic-20261008-pair-r1/coverage.json`.
It reproduces 2/1/5 pairs at 2 cm hand RMS; diagnostic limits of 3/5/8 cm
produce 2/1/5, 3/1/5 and 4/1/5. The extra waves increase coverage, but the
substantive sample remains one validation pair and two training pairs.

Question: is this enough evidence to fit the matched evaluator?

Root decision: stop before fitting. `train_matched.py` would resample the same
two training comparisons for every update and select on one validation pair;
that is an engineering execution check, not a decision-useful oracle-headroom
Probe. The raw, label and diagnostic artifacts are retained. Do not relax the
2 cm state contract or claim E0/Eoracle ranking information from this run.

The next useful experiment is a new targeted collector that creates explicitly
paired current states (same reset/history and two residual branches), followed
by a fresh fit Decision Note. Generic waves and automatic specialist epochs
are closed for this route until that protocol is specified.

## Post-run contract audit (2026-10-08)

The implementation now uses rule
`geometry-corroborated-H-matched-state-residual-events-v3`. A local preference
must match `H` within relative RMS `0.25`, in addition to the existing object
and hand geometry thresholds. Labeling, window preparation, `Windows`, and
`train_matched.py` count unique unordered episode pairs and enforce
`train=8`, `val=4`, `test=4` whenever `pair_coverage_required=true`. The
historical label artifact remains preserved for audit but cannot pass these
fit gates. Its route also records `all_experts_operationally_qualified=false`
and `training_allowed=false`, so no new production collection or fit is
authorized from it.

`consequence_evaluator.twin` is a strict Isaac-free v1 contract for a future
collector. It checks the canonical native state inventory, Python/NumPy/Torch
RNG provenance, fresh-simulator prefix replay metadata, rigid object poses and
branch starting anchors. A native capture adapter now emits this snapshot from
an initialized task/controller boundary and rejects incomplete prefix traces;
it still does not run Isaac Gym or create twin branches. The native replay
collector and its branch manifest remain unfinished.

## Native replay engineering check (2026-10-08)

The user requested review and repair of the new twin implementation. Before
changing the continuous observational collection protocol, run the supplied
`tools/audit/native_twin_probe.py` against the qualified self-trained airplane
endpoint. This is a Blocker engineering check: do three fresh native processes
reproduce an eight-control-step prefix exactly, and does the repeated positive
residual branch reproduce its measured trajectory? Use one idle GPU, one
environment, at most 900 seconds and 1 GiB in a fresh owned output. Reject
state/RNG/input drift; do not relax exact identity after observing the result.
The result decides whether the native adapter needs repair before a separate
collection design. It creates no training dataset, changes no ref2 contract,
and does not authorize additional specialist training or evaluator fitting.

### Review and repairs

Standards review found stale qualification and execution status in the Task
README; these now match the actual endpoints and raw collections. Spec review
found that native repeat acceptance omitted hand futures and termination, and
that the production qualification gate also prevented read-only relabeling.
Repeat checks now cover residual plans, executed controls, object poses, hand
keypoints, done and the complete canonical native future trace. The explicit
labeler `--audit-only` path preserves source/route hashes and always emits
`training_allowed=false`; production gates remain closed. Twin collection
remains a scope addition separate from ref2 continuous observational data.

Real native execution exposed two additional implementation defects. Run r1
failed serializing PhysX enums; r2 exposed recursive vector dtype metadata.
The serializer now handles enum values, dtype metadata and named structured
fields, with a recursion bound. Run r3 reproduced the physical traces exactly
but failed branch provenance after payload deserialization: whole-tree pickle
hashing depends on shared string/dtype object identities. Commit `a7d5146`
replaces it with framed content hashing; identity survives save/load without
weakening state equality. Original failure logs and artifacts remain intact.

### Measured results and limits

`outputs/consequence-evaluator/label-history-audit-20261008-r1/` rechecks the
216 existing episodes at commit `08c3ac3`, without GPU/model computation.
It completes in 40.83 s with 14,117 windows, window pairs 1/1/2 and unique
episode pairs **1/1/1** (train/val/test). It is
`INSUFFICIENT_PREFERENCES` and never trainable; H matching therefore does not
repair the observational coverage blocker.

`outputs/consequence-evaluator/native-twin-engineering-20261008-r4/` executes
commit `a7d5146` on idle physical GPU 0, seed 17, one environment, eight
control-prefix ticks / sixteen physics frames, then two opposite 24-step
decision-known residual plans and a repeated positive branch. All three
prefix state/RNG hashes agree, measured replay error is zero, and every
repeat check passes. The pair contract reports requested-plan L2 0.75865 and
executed-control L2 1.14996. Result: **ENGINEERING_PASS**, not a scientific
Probe conclusion, qualification or preference label. The driver stays below
its 900 s / 1 GiB limit (67.44 s, 99.1 MiB); PyTorch peak allocation is about 141 MiB, while
observed total native-process GPU memory reaches about 16 GiB including
PhysX. GPU 0 is released on completion.

CPU regressions: 124 Task tests pass, including serialization identity,
interaction/done repeat failures, physical enum/dtype handling and audit-only
fit rejection; `tools/verify.py --changed` also passes against `e0861bd`.
Next decision remains a separately specified collection protocol with enough
independent paired states and train progress anchors. No twin-to-continuous
schema conversion, evaluator fitting or further generic waves occurred.

2026-10-08 user clarification: retain ref2 continuous rollout and no forks for
formal collection; twin remains engineering diagnosis only. Subsequent work
must address continuous observational coverage and may not switch these data
to twin supervision.

## Continuous phase-capacity audit and focused Decision Note

The CPU-only exhaustive diagnostic is
`outputs/consequence-evaluator/continuous-phase-capacity-20261008-r1/report.json`.
It pins the existing label manifest and inspects every selected, unambiguous
event window with the current fixed state rule. Physical matches allow 4/1/3
unique episode pairs, but adding H leaves exactly 1/1/1, equal to the labeler's
selected coverage. Thus the selector did not discard additional eligible
episode pairs. The audit finishes in 7.24 s; it is diagnostic-only and cannot
authorize a fit.

Question: does focused continuous allocation to a qualified airplane expert's
hold stage create comparable positive/negative windows, or is independent
rollout matching still too sparse? This serves the ref2 oracle-headroom input
blocker, not an online Cm claim. If coverage appears, specify fresh production
collection with its qualification scope intact; if it does not, inspect the
recorded phase reach/dose/state mismatches before spending more rollout budget.

Root decision: one bounded, **audit-only** continuous run on the restored
`s3_airplane_lift` motion, existing six-role route and unchanged owned weights.
Use idle GPU 0, seed 299 (Probe pool), 24 environments, two waves, amplitude
0.08, <=1200 steps/episode, <=900 s and <=2 GiB output. Within the actual
motion, half the episodes are clean and half receive one hold-triggered
24-step smooth residual; all run independently from reference frame zero
through termination with expert feedback, never fork/replay a branch. Keep
geometry, H, events and abstention thresholds fixed. Six actor hashes and
ancestry are still mandatory; the unqualified route and all audit outputs
remain `training_allowed=false`. No new expert training, model fit or
qualification bypass in production is authorized.

Fresh raw output:
`outputs/consequence-evaluator/continuous-hold-audit-20261008-r1/`.
Read-only labeling output:
`outputs/consequence-evaluator/labeled-hold-audit-20261008-r1/`.
Stop on input drift, foreign GPU use, budget exhaustion, partial episode or
failed native geometry. Training and formal data gates stay closed regardless
of this diagnostic's pair count; there is no new external permission boundary.

### Focused continuous run result

Commit `7389d05`, seed 299, GPU 0: the collector completes 48 independent
episodes in 115.86 s, 103.20 MiB. Allocation is 24 clean / 24 hold;
23 hold interventions actually trigger. All frozen sources still match;
the native process exits and GPU 0 is released. No fork was used.
Read-only labeling takes 7.32 s with 2881 selected windows and 11403 reliable
train progress-anchor frames. Event windows include 2040 maintained holds,
22 unrecovered drops in hold, 48 lift achievements and 30 grasp losses.
Geometry records 26016 valid frames, 19022 force-proxy frames, 18788
proxy-and-near frames and one proxy-far frame; force is still only a proxy.

`outputs/consequence-evaluator/hold-phase-capacity-20261008-r1/report.json`
pins all labeled episode/diagnostic and implementation hashes. Among 456
event-direction-compatible episode pairs, only one has a current physical
match under the fixed thresholds, and the same one also passes H. The labeler
selected that one train pair; val/test were intentionally not collected in
this single-split diagnostic. All outputs remain untrainable. This does not
establish that continuous rollouts are ineffective; the current event/window
sample cannot support the planned evaluator fit.

Next cheapest diagnosis: use the same immutable raw episodes, current event
rule and thresholds, but inspect every decision-known full-window start
instead of the labeler's eight-tick cadence. The audit tool's `--all-ticks`
option creates only a capacity report; it changes no labels, model inputs,
fit gate or sampling contract. Stop within 600 s / 20000 event windows.
This distinguishes cadence loss from genuinely sparse comparable states
before allocating any further GPU rollout.

### All-tick audit and current-state attribution

At commit `93dde07`,
`outputs/consequence-evaluator/hold-all-ticks-capacity-20261008-r1/report.json`
checks 17053 unambiguous event windows in 7.69 s. Event-compatible unique
episode pairs increase to 463, but physical and H-matched capacity remain
one train pair. Thus neither capped selection nor eight-tick cadence explains
the missing independent coverage. Val/test are absent by design in this pilot.

The read-only dose diagnostic
`outputs/consequence-evaluator/hold-dose-diagnostic-20261008-r1/report.json`
records 21/24 reliable clean successes; among 23 triggered interventions,
15 trigger-start windows maintain hold, seven have an unrecovered drop and
one abstains. There are 13 negative event windows during the residual plan
and 19 after it, so the intervention does create local negative events; the
blocker is not simply that no perturbation changes a local outcome.

`outputs/consequence-evaluator/hold-state-mismatch-20261008-r1/report.json`
inspects the closest positive selected window for each of 12 episodes with
hold/drop negatives, minimizing the largest normalized physical-threshold
violation. Eleven nearest comparisons exceed the 2 cm hand RMS limit and
ten exceed the 15-degree object rotation limit; only one meets all limits.
This explains why event diversity does not yield state-matched comparisons.
The diagnostic explicitly checks agreement with the actual geometry rule;
its distances are not labels or a proposal to relax thresholds.

Current decision: keep fit and production-route gates closed. Do not collect
additional val/test waves for this pilot or promote twin data. Further
continuous data need an explicit design that produces comparable current
states or separately grounded local annotations; generic waves, unqualified
specialist epochs and outcome-derived preference labels remain closed.
The broader oracle-headroom question is still `UNCLEAR`, not refuted by this
coverage-limited sample. All 130 Task tests pass; user/parallel-agent changes
remain untouched.
