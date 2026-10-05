---
schema: ref2dex.probe.v2
probe_id: P-20261005-rolling-oracle-control
experiment_id: P-20261005-rolling-oracle-control
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 73dd2f800c1fa8dec9fe5a64ca0a335bae336465
claim_id: C3
hypothesis_family: HF-rolling-oracle-control
probe_index_in_family: 1
seed_pool: probe
seeds: [263, 264, 268]
decision_changed_if_positive: qualify rolling short-Y utility for a separately frozen noise-tolerance experiment
decision_changed_if_negative: audit rolling execution and distinguish local utility failure from limited repeated-action authority
status: PROMISING
run_id: rolling-oracle-control-s263-s264
---

# Ref14_1: executed Rolling GT-Y Oracle Control

Result: PROMISING on the predeclared 32-anchor gate: baseline 23/32, actual
same-current-state rolling GT-Y 27/32, 4 rescued, 0 harmed, gain +12.5pp
(paired bootstrap 95% interval 3.125--25.0pp; lower bound 3.125pp).
All four synchronous groups completed with exact baseline-repeat and prefix
checks; the result remains a mechanism Probe on the exposed cohort.
Decision: Execute same-current-state rolling oracle before any new predictor, long-Y, execution forecasting or PPO.

## Decision Note / purpose

Mission C3 requires useful task information, rather than lower prediction loss.
Ref13 one-shot23/32→24/32 was UNPROMISING; ref13_1 showed delayed factual-path
signals but did not execute control. User ref14 defines the reverse dependency
chain; updated ref14_1 explicitly requests real same-current-state rollingGT-Y.
Question: does perfect unchanged short-Y improve bounded stable-graspZ when
reapplied every8steps? Positive prioritizes noise tolerance, negative pauses
this fixed controller after checking implementation/action authority.
No change to global claim and no external authorization boundary is crossed.
Cheapest valid method reuses the32 pre-treatment-frozen anchors and existing
cold initial states; PhysX warm state is not serialized, so replay full actual
prefix in a fresh simulator for every candidate and executed mixed action.

## Fixed cohort, actions, short-Y and Z

Four original synchronous groups: seed263/96env, clocks42/84, sizes7/13;
seed264/48env, clocks62/136, sizes9/3. Keep all32 rows,29s3/3s7.
This is an already exposed mechanism cohort, not an independent confirmation.
Before collecting alternatives, fresh baseline+repeat for every group must
reproduce old currentH, history, observations, actions/PD and90step height/pair
EXACTLY. Original current-only eligibility remains fixed; verify EVERY row
supports finaloffset88+32 future observations (motion/rollout remaining>121).
Failure aborts, no future-outcome row filtering or fallback subset.

Same frozen self-trained source_e260/checkpoint, airplane, canonical3motions,
30Hz, hybridreset. GPU PhysX,CPU tensor pipeline,singlethread, GPU6 actor.
Seven actions unchanged: baseline,thumbyaw±.2,middle±.2,grip+.2,wristz+.01m;
native feedback residual for8steps, then frozen actor feedback for24steps.
At EACH query, candidates use the identical actual prefix. First simulate
baseline. Fixed utility has a provable upper bound1.25 (Y7<=1,Y3<=1,Y6>=0).
When ALL group baseline utilities equal1.25, the baseline-first tie rule
mathematically certifies baseline choice, so omit the six dominated-or-tied
lookaheads and use that actually executed baseline first8step path. Record
only observed baselineY; omitted candidateYs remain unknown. Otherwise
physically simulate allseven candidates and execute the chosen mixed plan.
This exact branch-and-bound shortcut is frozen before main results; it
changes neither the selector nor cadence and uses no Z information. No future actual-flow input, learned model, E/I bottleneck or
execution predictor. Action-conditioned actual flow remains a later modeling
contract, not a required archive of all proposed physical trajectories here.

Y8 retains ref8/ref13/ref13_1 exact semantics, continuation steps9..16/32,
local contactloss history spanningstep8, current-risk mask, lateheightfailure
and heightheldfraction. Fixed U=Y7+.25Y3−Y6. Baseline-first then fixed armorder
exact ties, without new coefficients/masks. Replan at offsets0,8,...,88: full
32future steps at EVERY query, including beyond Z90. No padding/truncation.
Clipping at later queries is recorded, never used to drop candidate/anchor;
original headroom applies only at anchor0. Repeated residuals are new rolling
authority, disclosed separately from one-shotK8 authority.

Execute the per-env selected actions in a SEPARATE simulator run of the actual
mixed plan, not a mosaic of candidate8step states. Next query uses that run's
actual complete actionprefix, physical state, history and deterministic actor.
Intermediate execution run's24step speculative suffix is discarded when the
next query forks after8steps. Final Z uses one CONTIGUOUS executed mixed path's
first90steps, never selected branchZ bits. Last actionblock crosses90 but only
itsfirst2steps enter Z; speculative lookahead and replay work are counted as
extra oracle compute. Final shared-solver numerical co-treatment remains a
limitation; report selected-fork versus actual mixed first8height disagreement.

Z unchanged:45consecutive held steps (height>=3cm above reference rest AND
aggregate forcepair) bystep60, then no height<2cm or6consecutive pairloss through
step90:1.5s hold plus>=1s followup, bounded3s target. Native terminal/incomplete
window/nonfinite aborts entire experiment; no outcome-dependent exclusion.

## Engineering and decision checks

Cold root setter tolerance inherited2.5e−7 pose ONLY; velocity/DOF identity exact.
Full-world replay physical72,q/dq,root<=1e−4; shadow BASE actoraction<=1e−5;
assignedcurrentobs/history<=1e−4; prefixdone exact. Previous interventions'
applied actions are not compared against zero-residual shadow base actions.
Coldinitial,asset/controller/backend/model/RMS identities checked and actor
frozen. Every candidate currentH compared; finite all32lookaheads required.
Chosen delta must exactly enter recordednative actions and inherited PD mapping.
Engineering smoke: fresh oldbaseline/repeat, then mixed nonzero plan, then
fresh next-state zero repeats EXACT, including history/currentH/Y fields.

Primary paired32anchor result: baselineZ vs ACTUAL rollingZ; report rescued,
harmed, qualification/drop, selections/clipping, pergroup/motion and Yscores.
Fixed gate: adequate>=30anchors/>=2motions; PROMISING only gain>=5pp and
anchorbootstrap2000seed268 lower95>0; otherwise UNPROMISING for adequate cohort,
UNCLEAR for inadequate support. AnchorCI descriptive only; solver/group
uncertainty is not covered. No scientific conclusion beyond Probe labels.
Historical one-shot24/32 andfinitecandidate25/32 are comparisons, NOT rolling
upperbounds. Actual GT-Z rolling upper unknown; no R_Y gain-retention ratio.
Gate3 noise tolerance is deferred, not automatically run during this experiment.

## Completed result and next decision

The final immutable result is `outputs/cm-interaction-oracle/rolling-oracle-control-s263-s264/result.json`
at code commit `337d3a6633b5ea83f921b9b873d7b7da15d6f46c`. It reports
`status=PROMISING`, `adequate_support=true`, `baseline_count=23`,
`rolling_count=27`, `rescued=4`, and `harmed=0`. The raw result and all four
group records are retained; this does not convert the exposed mechanism cohort
into a Validation or prove trained-policy Cm utility.

Decision Note (2026-10-06): the positive fixed gate makes a separately frozen
offline Y ranking/noise tolerance Probe the cheapest next decision. It will use
only fully observed same-state candidate panels from the saved plans, with a
predeclared channel-noise grid, and will report ranking/regret coverage. It will
not infer closed-loop Z retention, alter Y/U, collect PhysX data, or train a
predictor. A fresh rolling intervention is required before any claim about
noisy-Y policy performance.

## Resources, provenance and stopping

Engineering cap600s plus main7200s wall,<=2simultaneous physical GPUs6/7,
<=4independent simulator workers concurrently,<=2workers/device, eachnative180s/shell240s.
Use first idleGPU with alltask-owned worker processes; no interference with
others. No model training/policytraining. Pure labels/statistics useCPU.
New artifact cap4GiB (main),0.5GiB(engineering), shared global300GB applies;
free disk preflight about66GB. Baseline cache/output symlink targets stay
immutable; unique new runfolders in authorizedai_ws only.
Smoke `outputs/cm-interaction-oracle/rolling-oracle-smoke-s263/`;
main `outputs/cm-interaction-oracle/rolling-oracle-control-s263-s264/`.
Commit code before native execution; frozenprotocol/inputsha/runmanifest and
progress perround retained. Stop on identity/replay/window/finiteness/choice
wiring drift, resource conflict or cap; preserve failures, do not reinterpret
implementation failure as negative method evidence. Root checks actual trace
and rawstatistics. Independent review only if implementation doubts arise or
negative would close this fixed route; no extra fitting/simulation budget.

Tools: `tools/run/run_rolling_gt_y.py`, extended
`tools/run/collect_oracle_y_candidates.py`, contracts `src/rolling_control.py`.

## Limitations / future evidence

Perfect short-Y uses real future simulator feedback continuation; repeated
replanning subsequently changes that continuation, an explicit local-utility
hypothesis. Real rolling intervention resolves saved-path stitching bias but
cannot prove real-time planner feasibility, predictive point-flow precision,
contact/friction authenticity or final trained-policyCm utility. Exposed
baseline-biased cohort,4solvergroups,onepolicy/object,forceproxy andboundedZ
limit interpretation. Samecode baselinerepeat/currentprefix controls do not
validate postfork solver independence. Independent cohort/formalValidation,
noise tolerance, representation comparison, desiredflow execution, learned
planner andmatchedCm-on/off policy training follow gates in ref14 later.

## Engineering result and main resource choice

Smoke at4b4769b completed163.16s:7anchor freshbaseline/repeat/old fields
EXACT,4 nonzero mixed actions executed, bothnext-state repeats EXACT.
No scientific control gain inferred. Measured one simulator uses about7.5GiB,
so fourworkers ononeGPU would not fit. Main uses two idle3090 GPUs6/7,
twoworkers/device, within global4GPU cap. Same tensorbackend/actor math and
fullworld/currentH replay remain required acrossphysical devices. Resource
choice is made before main, not a gate/target change. Main run uses a new
fixedcode checkpoint after the exact utility-upperbound shortcut.
