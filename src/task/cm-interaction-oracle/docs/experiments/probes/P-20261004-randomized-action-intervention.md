---
schema: ref2dex.probe.v2
probe_id: P-20261004-randomized-action-intervention
experiment_id: P-20261004-randomized-action-intervention
date: 2026-10-04
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: actual execution commit in manifest.json
claim_id: C3
hypothesis_family: HF-action-intervention
probe_index_in_family: 1
seed_pool: probe
seeds: [209, 210, 211]
decision_changed_if_positive: test a predicted-Cm physical candidate selector; only then consider matched policy training
decision_changed_if_negative: stop expansion of this residual and consequence contract after implementation review
status: RUNNING
run_id: randomized-intervention-s209
---

# Can randomized current actions control task-relevant consequences?

Result: Pending bounded physical intervention collection and held-out diagnostic.
Decision: Execute the ref3 control-information gate before any more RTG/value fitting.

## Motivation and Decision Note

Decision Probe for Mission C3: the missing link is controllable, predictable,
task-relevant consequences, not another E/I→RTG encoder. ref1/ref2 proxy
targets were unstable or unhelpful. Historical GT-consequence information
does not establish pre-action control information. Root chooses actual
random assignment in one fixed airplane task, one frozen self-trained
source_e260 actor, without state branching. No core hypothesis or final claim
is abandoned. Negative results close only this local dose/data/model contract;
insufficient action contrast/outcome support remains UNCLEAR.

Cheapest discriminator: one engineering wave followed by four bounded waves
of84 environments (336 native episodes); if fewer than196 usable interventions
but collection remains healthy, at most four additional waves within the
same total budget. Random draw uses seed209, native simulator210, model/split211.
Engineering smoke uses debug7/8 and a separate run. No seed or dose search.
Success leads to predicted-Cm candidate control; failure gets independent
engineering review, then stops or repairs only the invalid part. No external
authorization is needed within the existing Campaign bounds.

## Randomized physical contract

Seven equally likely arms: zero; wrist x±.01; wrist z±.01; finger synergy
±.1 on active native indices6/8/10/12/15. Native wrist units mean1cm PDtarget
offset at each feedback step; finger units map through half the joint range,
not .1rad. Four successive steps execute `clip(pi(current_obs)+delta)` then
return to the frozen deterministic actor. Future feedback actions are logged
for dose audit, never inference input. The planned operator is known at time0.

One intervention at the first eligible state per episode: ≥10 pre-action
history observations, >33 reference steps remaining, measured hand AND object
net force>.1N, a contact-body-center to sampled object surface distance<6cm,
and current action headroom for ALL seven candidates. This proximity rule is
a broad decision-region screen, not a certified hand-object contact detector.
Eligibility is fixed before assignment; no post-treatment quality filtering.
All arms share all actor observations and measured physical context.

Only synchronized full-batch resets between complete waves. An environment
that terminates is no longer recorded but is not reset until the barrier;
its post-terminal evolution is never used as outcome evidence. Assigned
trials remain in the packet even if a window is truncated; any truncated
32-step window blocks the fixed-window gate rather than silently excluding
the trial. Native early termination/curriculum disabled. Baseline Gaussian
sampling and episode noise removed. No counterfactual state-restoration claim.

## Physical labels and real task outcomes

Record time0 native obs1442, compact measured history10×139, context435,
current basea18, actual humanoid root13; assigned delta18, and32 native
post-step physical states/actions/baseactions/PD targets. Measured hand
body poses are used directly; no identity-root FK. All input hashes, source
checkpoint, actual code commit, model environment, timestamps enter manifest.

E12 at8 steps: object world translation/short rotation-vector change and
linear/angular velocity change. I14 at8: log1p five measured hand-force norms,
three object-force absolute components, five body-to-object surface distances,
and net-force pair proxy. Force/proximity I is not true paired-contact,
friction, or slip ground truth. Fixed representation; no I feature sweep.

Y16/Y32 each contains actual object Δz, net-force proxy retention fraction,
lift≥3cm AND proxy-contact held fraction, and conditional drop. Rest height
is fixed reference frame0. Drop is evaluated only when lifted≥3cm and contacted
at decision: subsequent height<2cm or six consecutive lost-contact steps.
Report that risk denominator separately; no-drop pre-lift states are not
drop-success examples. Full episode primary45-step held success and later
drop are additionally recorded, never substituted by30-step tracker events.
No bootstrap V, RTG, reward composition or future-action inference inputs.

## Offline diagnostic and decision gate

Freeze episode-level split, stratified by motion and randomized arm, seed211;
test is20%. All preprocessing fits train only. H includes full pre-action
actor observation, physical history and current reference context; all arms
receive identical H. Predictor target is E/I8; outcome target is Y16/32.
Compare H state baseline, H+planned action direct critic, predicted E/I
mediated scorer, and GT E/I8 scorer as privileged offline upper bound.
Predicted scorer training uses held-out/cross-fitted predictor outputs,
not in-sample Cm predictions; no GT labels at test inference. Fixed model
protocol and dimensional reduction recorded before fitting, after checking
input dimensions but before inspecting outcome contrasts. Primary compare
per-outcome errors and within-motion/stage cross-episode outcome ranking;
these are not same-state counterfactual ranks. No complex hand-made scalar.

Frozen model protocol: concatenate flattened history/native obs/context and
current base action into common H; train-only standardization (scale floor.001)
and exact SVD PCA32, then train-only projected standardization. Base action
is common to ALL controls; randomized intended delta18 is the action input.
Each conditional channel has26 slots: zero for H, delta/.1 padded with8 zeros
for direct/predictor, standardized E/I26 for scorer. Identical MLP64→32→output,
tanh, init211, AdamW lr.002/decay.001,100 fullbatch epochs, gradient clip2;
no architecture/epoch search. Physical targets26, task targets8, loss normalized
MSE; drop loss conditional on risk and disabled if train-risk<20/test-risk<5.
All other task heads enter the task error. Cm scorer uses three whole-episode
stratified OOF predictor fits, fold-specific physical target normalization;
test predictions come from one full-train predictor. Training input PCA is
shared and label-free; every scorer has identical capacity/init/protocol.

Ranking uses held-out different-arm episode pairs within motion × four
pre-action reference-phase bins × pre-action drop-risk, with no fake same-state
claim. True difference tolerances: dz1mm, retention/held.03, drop.5. Macro over
supported strata then six dz/retention/held heads; report each head and pair
support, drop separately at risk. Permute assigned intent among held-out states
within these same strata, leaving H/current basea unchanged. This sensitivity
check does not turn unsupported H/action pairs into physical counterfactuals.

Before fitting require ≥196 trials, ≥20/arm, all assigned windows32 complete,
finite inputs and ≥90% intended dose retained over all assigned nonzero
chunk elements. Outcome support requires variation in Δz and at least one
retention/held axis; drop may remain unsupported. If these fail, UNCLEAR,
with engineering review where implementation is suspected.

PROMISING requires action-conditioned consequence prediction improve≥5%
held-out normalized MSE over H, randomized action permutation worsen it≥5%,
GT short-consequence scorer improve≥5% task error and≥3pp task ranking over H,
and predicted mediated scorer improve≥3pp ranking over H with useful action
sensitivity (permutation drops task rank≥2pp). A matched direct critic is required; if direct is equally good,
record no unique Cm benefit and do not claim policy utility. Report E and I
separately, dose, randomized balance and per-arm actual outcomes. Fixed gates
are Probe decisions, not statistical Validation; no core-route refutation.

## Resources and artifacts

One currently idle GPU6 (logical cuda:0), ≤40min cumulative collection/model
execution, ≤4GB new artifacts, ≤8×84 native episodes plus21-env smoke.
Stop on deadline, input drift, nonfinite state, invalid native reset/dose,
OOM or resource conflict. No video/cache expansion. CPU only for statistics,
file hashes and contract tests; simulation/actor/model operations use GPU.

Task-local collector `src/task/cm-interaction-oracle/tools/run/collect_interventions.py`;
contract `src/task/cm-interaction-oracle/src/intervention.py`. Artifacts use
`outputs/cm-interaction-oracle/<run_id>/`, resolving via the existing output
symlink to authorized `Ref2Dex-agent-baseline/outputs/cm-interaction-oracle/`.
Only unique new runs; checkpoint, symlink and unrelated user edits preserved.

## Limitations / future evidence

One simulator/model seed, fixed actor and airplane references; proxy contacts,
nearby-state comparisons, limited at-risk drops, bounded seven-arm doses.
Statistical power for heterogeneous treatment effects may be inadequate.
Closed-loop selection and matched Cm-on/off policy training remain later
gates; a frozen selector gain does not complete the Mission. True friction/
paired-contact labels, varied perturbations and formal multi-seed validation
are deferred until this intervention changes the next research decision.

## Engineering smoke

`randomized-intervention-smoke-s7` failed before simulator creation because
CLI used `--use_gpu_pipeline` instead of SDK `--pipeline gpu`; directory and
failure manifest retained. No scientific evidence or interventions produced.
The corrected launcher at202a1b1 uses project-local Torch extension build/cache.
`randomized-intervention-smoke2-s7`:21 complete native episodes,20 assigned
trials,20 complete32-step windows,70s simulation. This establishes executable
collection and decision-region coverage only. Formal arm/outcome gates have
not been judged on this debug sample; no seed/dose selection performed.
