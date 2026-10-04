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
seeds: [209, 210, 211, 212]
decision_changed_if_positive: test a predicted-Cm physical candidate selector; only then consider matched policy training
decision_changed_if_negative: stop expansion of this residual and consequence contract after implementation review
status: UNPROMISING
run_id: randomized-intervention-s209
---

# Can randomized current actions control task-relevant consequences?

Result: 320 actual randomized interventions; action-conditioned E/I8 and mediated task ranking fail all seven fixed gates under environment holdout.
Decision: Stop this pre-lift E/I8-to-Y16/32 PCA/MLP expansion after independent review; preserve weak object-rotation response and do not refute core Cm.

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
Input permutation/fold RNG is the fixed derivative seed212.
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

Freeze grouped physical-environment split across waves, stratified by motion,
seed211; test is20% of environments (report arm balance, no outcome selection).
All preprocessing fits train only. H includes full pre-action
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
environment-grouped OOF predictor fits, fold-specific physical target normalization;
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
sensitivity (permutation drops task rank≥2pp), plus mediated rank≥2pp above
the matched direct critic. If direct is equally good, record no unique Cm
benefit and do not enter a Cm selector on these fits. Report E and I
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

Before formal fitting, root tightened the split to environment groups because
the four waves reuse native environments; all trials from an environment
remain in one partition/fold. This was done while collecting, before inspecting
formal physical/outcome contrasts. Input dimension choice and model protocol
remain fixed. Seven-arm assignment still occurs independently per episode.
The new transition MLP is trained on randomized E/I targets; it is NOT the
historical frozen point-flow checkpoint and its results cannot relabel that
model's previous evidence. Predicted-Cm benefit over direct is an explicit gate.

## Post-fit Decision diagnostic, before execution

All seven primary gates failed. Independent review confirms actual hand
perturbation but identifies limited pre-lift coverage, sparse outcome-pair
support and fitting/generalization limits. Root therefore chooses one cheap
existing-data diagnostic, not another model/encoder sweep: adjusted randomized
arm effects on E/I8, Y16/32 and recorded full-episode hold/drop. It distinguishes
physical intervention not reaching object/task from poor representation/fit.
If a signal survives randomized assignment controls, retain that control
question but do not rescue the failed learned gate; if absent, stop this
short-consequence/dose expansion without refuting Cm.

Fixed diagnostic: linear nuisance adjustment using CURRENT object pose7,
dof18, contact-body positions15, hand-force norms5, object-force3, hand-root
position3, normalized reference phase, decision tick, and motion×phase4×
warmup(tick<20) blocks. No future nuisance inputs. Fit centered arm6 contrasts
against zero via pseudoinverse after nuisance projection.1999 label permutations
within pre-treatment blocks, RNG211; report per-axis partial explained variance
and descriptive family-wise maxima separately for E/I, local outcomes and
full-episode outcomes. Report all arms/pairs, no best-arm selection or changed
primary model gates. Conditional exchangeability/carryover and coarse nuisance
adjustment remain limitations; exploratory permutation scores are not Validation.
CPU≤30s for statistical matrices, zero new simulator/model fitting; existing
scope≤40min/4GB unchanged. This is a Decision diagnostic because evidence of
real physical response determines whether to retain a predictor/control question.

## Completed physical intervention and learned result

Formal collection `randomized-intervention-s209`, execution06d293c; model
diagnostic `randomized-diagnostic-s211`, execution0eae290. Later audit/provenance
fixes do not rewrite either original manifest or their failed-gate result.
Source_e260 actor remains frozen/pinned; deterministic inference, zero episode
noise. Four synchronized waves,336 complete episodes,320 interventions,
320 complete32-step windows,16 episodes never eligible. Arm counts in declared
order44/54/50/37/51/42/42; each arm retains100% intended chunk dose within
floating precision. Zero and post-chunk actions/PDtarget differences exactly0.
Dataset SHA256 `6f164291193e220d13b3b81c99cfcf35d27211218b84f564b5067e9e4c84e741`.

Grouped train249 trials/66 environments, test71 trials/18 environments.
Every environment remains within one partition and one OOF fold. Test arm
counts9/13/9/10/8/12/10. Current decision lift maximum8.743mm above reference
rest: **all interventions were pre-lift, zero early-hold/drop-risk support**.
Drop losses disabled and errors/ranks unsupported, never interpreted as no-drop.
Outcome variability checks pass; no assigned trial removed after treatment.

| Consequence predictor | Test normalized E/I MSE | E12 MSE | I14 MSE |
| --- | ---: | ---: | ---: |
| H state | 0.574823 | 0.727212 | 0.444203 |
| H+assigned action | 0.584113 | 0.738725 | 0.451589 |
| H+permuted assigned action | 0.583560 | 0.737626 | 0.451504 |

Current action worsens E/I error1.62%; permutation improves it0.095%.
Neither E nor I has useful held-out action-prediction gain in this fit.
This is not a claim that physical action effects are absent.

| Task scorer | Test normalized task MSE | Macro outcome pair accuracy |
| --- | ---: | ---: |
| H | 0.376690 | 66.782% |
| Direct H+action | 0.376723 | 66.145% |
| Predicted E/I mediated | 0.372838 | 65.169% |
| GT E/I8 teacher | 0.364117 | 61.175% |

GT improves error3.34% but ranking−5.607pp. Predicted mediated error improves
1.02%, ranking−1.613pp vs H and−0.975pp vs direct; permuted mediated action
increases rank3.999pp. All seven predeclared gates fail. The negative branch
does not launch a selector, onlineGT oracle, or teacher/policy training.
Matched scorers each6,120 parameters, identical init seed and100-epoch protocol.
No threshold/epoch/width/seed sweep or GT-test inference in mediated scorer.

Useful localized signal is retained: GT contact-retention MAE16/32 drops from
0.07915/0.07805 to0.06232/0.06984; its retention ranks rise from66.31%/74.09%
to74.21%/88.78%. Other height/held heads worsen enough to fail aggregate gates.
Therefore the present failure does not establish that I cannot prognose task
outcomes. Held16 ranking has only117 cross-arm pairs in2 strata; supported
strata/head counts differ. Pair counts are not independent sample counts.

Saved-score audit exactly reproduces fixed targets, input timing, dose,
grouped split/folds, test normalized error and all rank supports. Cluster
bootstrap18 held-out environments gives descriptive relative-error-gain CI95:
direct[−6.71%,+6.28%], mediated[−6.90%,+10.81%], GT[−19.66%,+19.14%].
These are descriptive Probe uncertainties, not multi-seed Validation.

### Independent engineering review and root attribution

Read-only reviewer `ref5_engineering_review` independently verifies native PD
mapping, pre-action history/base/state equality, quaternion E8 reconstruction,
complete windows, all grouped partitions/OOF/permutations and raw-score ranks.
Smoke residual error≤7.45e−9, independent PDtarget error0, rotation reconstruction
≤2.98e−7. Formal step4 hand x/z raw ±arm displacement contrasts are≈13.3/13.8mm;
the hand physically receives the intervention. Heterogeneous-state raw means
are not controlled causal estimates. No new fatal implementation defect found.

Model/data limits matter. GT trainMSE0.092 vs H0.188 shows the model uses GT,
but test gains generalize weakly. Transition trainMSE H0.458/Ha0.449 versus
test0.575/0.584; no convergence/accuracy claim. OOF-train consequence MSE0.815
differs from full-fit test0.584, creating scorer-input noise distribution shift.
PCA retains89.12% standardized training energy but7.27% aggregate test energy;
one historical angular-velocity-transient row explains84.78% of test energy.
Typical test retained-energy median86.96%. That row contributes only≈1.5%
Cm loss and3%–5% task loss, so it does not explain the entire failure. No
outlier removal or changed primary test denominator used to rescue the gate.
The earlier broad concern about test PCA coverage was narrowed by this audit.

Original collector rglob missed symlinked motion directories in input hashes.
`input_provenance_supplement.json` explicitly records POST_RUN hashes and source
mtimes preceding execution for all three actual motion files; original
manifests remain intact. This is a provenance limitation, not a pre-run hash
capture. Subsequent collector fix follows these links for future launches;
no repeat simulation needed. Failed CLI smoke is retained separately.

### Existing-data randomized-arm Decision diagnostic

Recorded post-fit protocol executed once,1.12s CPU,1999 within-block label
permutations. Nuisance rank61,11 pre-treatment blocks, all320 trials; no
new model/simulator, no target/primary gate changes. Maximum adjusted partial
explained variance in E/I8 is7.79%, family-max permutation tail0.045, driven
by object y-axis rotation change. Adjusted wrist x+/x− contrasts against zero
are+0.0271/−0.0502rad. This is a **weak exploratory physical response**,
not validated causal mediation. Task16/32 family tail0.1275, full-episode
family0.4145. Coarse linear current-state adjustment, selection of the
diagnostic after fit, repeated-environment carryover and multiple families
prevent formal significance claims. No best arm is promoted into a policy.

Full episode summaries:105/336 reach45 consecutive held steps,76 later drop,
29 reach this criterion without a recorded subsequent drop. All105/76 belong
to assigned trials. Zero arm14/44 reach hold,13 later drop; raw arm variations
are exploratory and do not establish selector or learned-policy improvement.
Different sampling from previous Gaussian/episode-noise pools prevents using
these totals to relabel earlier baseline results.

### Closing Decision Note

- Question: is this randomized physical contract ready for Cm candidate control?
- Evidence: actual dose and physical hand response present; seven learned gates
  fail, GT has localized retention information, exploratory object-rotation
  response exists, but task-relevant predictability is not established.
  Every intervention is pre-lift; early-hold/drop remains untested.
- Root choice: stop this PCA/MLP E/I8→Y16/32 expansion. Preserve the randomized
  packet and weak rotation signal. Do not grow I/RTG encoders, tune advantage
  targets, launch an onlineGT oracle, or bypass the gate with raw best-arm means.
  Retain the physical control question; this contract cannot justify ending
  Cm overall or the Mission's learned-policy utility objective.
- Cost/next/stopping: formal simulation222s, smoke70s, diagnostic model4.86s,
  post-fit statistics1.12s; oneGPU6, new artifacts<13MB excluding bounded
  project-local compiler cache. All jobs ended, no extra four waves necessary.
  Any later route needs a new decision-relevant stage/horizon contract addressing
  controllable grasp retention, not a retry of this failed fit under new seeds.
- Boundary: Mission/claim/resource authorization unchanged. Probe UNPROMISING
  for one representation/data/fit contract; Cm policy utility remains OPEN.

## Verification and completion

All18 Task tests pass, including independent-environment holdout, score-tie
credit, conditional drop and physical rotation/action-unit contracts.
`tools/verify.py --changed` PASS, generated index current, owned whitespace
check PASS. Statistical audit matches immutable formal scores/targets; independent
review also verifies the post-fit randomized diagnostic (family correction is
within E/I26 only, not across three families). GPU6 idle and no owned experiment
process remains. New project-local Torch compiler cache3.4MB is bounded.
No remote push, pre-existing checkpoint/output replacement, or unrelated user
edit staged. Ref3's actual intervention and four-model gate are complete;
its positive-only selection stage was not activated. Global Mission remains open.
