---
schema: ref2dex.probe.v2
probe_id: P-20261014-wam-residual-curriculum
experiment_id: P-20261014-wam-residual-curriculum
date: 2026-10-14
task: trajectory-policy
branch: main
git_commit: ea039ea3
claim_id: C3
hypothesis_family: HF-wam-residual-ppo
probe_index_in_family: 2
seed_pool: probe
seeds: [292, 296]
decision_changed_if_positive: continue the arm that improves startup acquisition or progressively bridge successful contact starts toward frame0
decision_changed_if_negative: do not equate insufficient evidence with convergence; inspect visited-state mismatch and residual authority before another long block
status: UNCLEAR
run_id: wam-ref2-execution-course-20261014-b2
---

# More updates versus contact-focused residual curriculum

## Motivation and decision

Serve global Objective A and this Task's autonomous grasp baseline. The user
questions whether 20 corrected PPO updates were insufficient and whether the
course spends too little time on contact-to-hold. This is a Decision Probe:
compare more updates under the existing reset distribution against a course
concentrated on pre-hold contact states. Keep the frozen WAM/World/old Chunk-R,
alpha=.15, reward weights, λ=1, optimizer, rollout horizon and model unchanged.
Do not claim convergence or close residual RL from the prior 20-update result.

## Existing evidence and source check

The r7 measured teacher prefix puts gap at phase32 about 860 mm, phase48
about 61 mm, phase50 about 14 mm, phase52 about 1–2 mm. Initial held is
false through phase54 in this prefix; first held appears at55/56. At62 it
is already held, about93–99 mm above task-start. Thus the old four-point
course (0/32/48/62) has no reset inside the actual 50–54 contact acquisition
interval. r7 normalized advantages on updates1/15/20 overwhelmingly favor
phase62; at update1 phases0/32/48 each have zero positive advantages,
while62 has83%. This is an observation about the collected batch and value
fit, not proof that global advantage normalization is wrong.

[Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300) motivates
gradually expanding successful start regions toward the original start
distribution. Here use that idea only to choose measured pre-hold starts;
do not implement the paper's Brownian-state sampling or claim equivalence.
The paper's original-distribution evaluation is particularly relevant:
success at a contact reset must not replace frame0 evaluation.

## Matched exploratory comparison

Common parent: `outputs/trajectory-policy/wam-ref2-execution-rppo-20261014-r7/final.pt`,
update20, with full actor/value/Adam/RNG and the same measured 4-env prefix.
Both arms restore the same training state and use the same phase-reset noise
stream and actor sampling stream. There is no teacher command inside training
or frame0 evaluation. B explicitly changes curriculum; it is a changed-course
continuation, not an exact continuation of the old distribution.

| arm | resets per update | continuation | run_id |
| --- | --- | --- | --- |
| A | 0/32/48/62 | +40 updates → total60 | wam-ref2-execution-course-20261014-a1 |
| B | 0/48/50/52 | +40 updates → total60 | wam-ref2-execution-course-20261014-b1 |

Each arm has 4 environments, 96-control training rollouts with actual
post-control bootstrap, and full542 frame0 evaluation at total40 and60.
Periodic evaluation preserves all training RNG states. A zero-residual
diagnostic covers the old four phases and any new curriculum phases; B must
also establish50/52 current context correctness. A and B share checkpoint
state/prefix, not identical post-intervention rollouts. Distinct but same-model
physical GPUs and GPU physics variability remain a Probe limitation.

New metric: per-env maximum consecutive held and `new_hold45`, which excludes
episodes already held at reset. Report time to contact/held, phase-specific
returns, raw/normalized advantages and positive fractions, value EV,
residual RMS/std and saturation. Held fraction alone is not acquisition.

## Decision Note and resource boundary

Choose this paired continuation rather than jumping to200 updates: each arm
adds15,360 physical training env-controls and provides intermediate full
evaluation. GPU4/6 if free; one native simulator and one WAM server per GPU,
expected18.4 GiB per GPU and about15–20 minutes concurrently. A has5060 RPCs,
B5156 RPCs, under the existing query cap; total outputs expected<200 MiB.
Hard deadline4 hours per arm. Shared project GPU/storage limits still apply.
Abort on occupied/foreign GPU, nonfinite tensors, context/execution mismatch,
unexpected early terminal or a zero-residual phase contact gate failure.

Decision screen at60: frame0 hold45 on>=2/4 copies is a PROMISING startup
signal for that arm; prefer the arm with materially better acquisition if
only one passes. If B consistently produces new_hold45 from48/50 and A
does not improve startup, prioritize a staged backward bridge of those states
toward frame0. If both frame0 remain0 and contact success is weak or unstable,
do not spend another block solely on more updates; diagnose residual authority
and actual visited-state coverage. If learning curves are still materially
improving, a further bounded +40 update continuation may be chosen and recorded
here. No core-claim change or new external authorization is required.

## Limitations / future evidence

One paired seed and one reference; not a formal curriculum superiority claim,
not a convergence study, and not Cm benefit. Fresh teacher physics replay is
only used to initialize the native environment; the saved parent measured
prefix is the reset source in both arms. Hidden PhysX contact caches are not
serialized. A common-state multi-seed validation, longer budget if justified,
joint-wise feasible action parameterization, actual frame0 visited-state
curriculum, and baseline basin screen remain deferred unless this Probe makes
them the next decision. CPU is used only for tiny contracts and file/label
statistics; model inference and training use GPU.

## Offline coverage / exploration checks

`outputs/trajectory-policy/wam-ref2-execution-course-20261014-summary/state_coverage.json`
compares r7's full frame0 evaluation to its measured teacher prefix. At the
first tight contact (gap<=10 mm), the closest teacher hand pose over phases40–55
still has28.6–60.0 mm hand-point RMS difference and .211–.326 rad active-finger
RMS difference. The actual first-contact clocks are48/49/49/67, so matching only
demonstration phase is also misleading. This suggests a teacher-start versus
actual visited-state coverage gap; it does not prove that gap causes failure.

Native controller scales make the initial Gaussian exploration physically
uneven: alpha=.15 and std=exp(-2) imply20.3 mm wrist translation std per axis,
about.064 rad wrist orientation std, .0162 rad for four finger coordinates,
.0117 rad thumb yaw and .00558 rad thumb pitch. Thus a shared normalized alpha
does not give equal physical exploration. No scale/std/reward changes were
introduced into this A/B; authority diagnosis is a possible next decision.

Zero-residual phase48 held fractions differ between the two native launches
(A .406 vs B .760), despite the identical saved prefix. Its WAM query batch
companions differ under the expanded diagnostic set, and GPU/PhysX trajectories
are sensitive. The pair must not be used for a formal attribution of small
curriculum effects. Original-start outcomes and broad learning trends are the
decision signals, with independent audit still required after completion.

## Decision Note: extend B from60 to100

B's total40 frame0 evaluation has held0 on all copies. At total60 it still
has held0, but close-range object rise reaches64.1 mm, versus the parent r7's
1.27 mm, with all objects ultimately remaining on the table. Some support
persists, so this is levering/partial lift, not a stable grasp. B's last20
training trials have new_hold45 rates3/20 at48,9/20 at50 and12/20 at52.
These are continuing acquisition signals, not evidence of convergence.

Root chooses one further +40-update block on B, same contact course and all
other settings, to total100. A remains the completed total60 budget/control
comparison; do not present A60 versus B100 as a matched curriculum effect.
Parent is B1 final.pt at60, with full Adam/RNG and the same measured prefix.
Run `wam-ref2-execution-course-20261014-b2` on GPU6 after B1 exits and its
engineering audit passes. Evaluate full frame0 at80/100, one GPU, same4-hour
cap, expected15 minutes,5156 RPCs and<100 MiB output. Stop at100 or on the
existing safety/contract abort conditions. If strict startup hold still does
not appear, prioritize actual visited-contact state coverage/authority checks,
rather than another unbounded teacher-course block. No new external boundary.

## Completed results and decision

Code: `ea039ea3`; A1/B1 manifests record `59a31268` (same implementation
plus protocol), B2 records `8ef5e766` (same implementation plus audit tools).
A1/B1 share exactly the same saved parent measured prefix, model/value/Adam
and RNG checkpoint. A1 completed1090.5s on GPU4, B1 completed1021.6s on GPU6;
B2 completed973.4s on GPU6. Typical memory18834–18836 MiB; utilization17–34%.
All are complete and the processes exited. No further PPO block was started.

| arm / total updates | strict frame0 hold45 | close-range rise | last20 new_hold45 at48 /50 /52 |
| --- | --- | --- | --- |
| A original /60 | 0/4 | 3.55 mm | 2/20 / not sampled / not sampled |
| B contact /60 | 0/4 | 64.13 mm | 3/20 /9/20 /12/20 |
| B contact /100 | 0/4 | 23.14 mm | 0/20 /6/20 /9/20 |

Intermediate total40 and80 evaluations also have strict held0 on all copies.
At60 the object remained supported despite its center rising; this is partial
lift/levering, not a hold. The close-rise helper automatically labeled B60
PROMISING, but this does not satisfy the protocol's strict startup criterion.
Card-level verdict is UNCLEAR on budget/course efficacy; descriptively these
bounded fixed teacher courses did not solve startup. The60-to100 extension
does not show monotonic acquisition improvement. A last-batch value EV of
-40.42 reflects an overestimated failed rollout (return variance1.88), and B100
last EV is-.170; neither training reward nor critic fit establishes convergence.

Independent audits passed all three runs:43/44/44 reset captures and40 true
post-control bootstraps per run, value maximum discrepancies3.81e-6/7.63e-6/
7.63e-6. Full command/PD capture checks ran in the native loop. Parent prefix
array equality and comparison settings were independently checked. Related
contracts20 pass. Mean active-DOF saturation was .393%/.365%/.383% for
A1/B1/B2, so global clipping frequency alone is not the explanation.

Artifacts are under `outputs/trajectory-policy/wam-ref2-execution-course-20261014-{a1,b1,b2}/`.
`tools/audit/compare_residual_curricula.py` writes the paired60 comparison;
`plot_residual_curricula.py --b-continuation <b2>` plots the selected extension
with an explicit unequal-budget caption. Summary files in
`outputs/trajectory-policy/wam-ref2-execution-course-20261014-summary/` include
comparison.json, state_coverage.json, learning_curves_60.png and learning_curves.png.

Decision Note after the corrected short run, course/budget run and sparse
authority screen: do not equate20 updates with a bottleneck or these negative
startup observations with RL impossibility. More data did expose a learnable
near-contact region, but teacher-start performance has not transferred to
self-visited contact. Stop repeating this fixed teacher course now; prioritize
an actual visited-contact reset/geometry screen with properly restored H4,
commands and reward state, then choose physical per-DOF exploration only after
finding a usable correction direction. Cost of that next work is not spent
here. No core hypothesis/claim or resource boundary is changed.

The separate [frozen finger screen](P-20261014-wam-residual-basin.md) tested
explicit thumb± and finger-close corrections on the exact base; all four
copies remained strict held0. That is a limited negative observation about
those directions/window, not exclusion of a wrist/finger basin or evidence
that the Gaussian residual architecture is incapable.
