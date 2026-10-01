# P-20261001-contact-consequence-ranking

HF09 slot2/3, Decision Probe. Active user Goal: consequence prediction and
held-state ranking with state-only/action-shuffled/always-base/best-fixed
controls, then direct short-segment replanning if supported.

The32state opportunity screen has2.224mm second-repeat candidate uplift versus
0.210mm base repeat noise, but11/32 valid full approximate pairs and no lifted
states. Scientific label UNCLEAR; no individual oracle/regret is established.
Switch measurement to the mechanism memo's randomized-intervention fallback,
without changing the failed paired gate or claiming it passed. Purpose:
distinguish learnable local action consequence from state/expert preference
and gather late-contact drop-risk support before any policy training.

## Frozen collection

Same airplane/three motions/six hash-pinned self-trained experts, base index4.
Seeded reference-frame resets; actual pre-action state and10step history.
At eligible first-episode contact (native force proxy true for3steps), draw
one candidate using a private uniform generator after observing state,
p=1/6. Cache current normalized action, execute it for2steps, own base8steps;
6step cooldown, maximum8 windows per first episode. No future action/label
as model input; each10step label belongs to its recorded actual state, not a
copied hot solver state or a different branch. All started windows must
complete before terminal; no selective dropping of partial windows.

Collect fixed simulator seeds331/332/333, assignment seeds7331/7332/7333,
96envs each, maximum650steps. Save candidate pool, assignment/propensity,
actual actions, future states/contact, explicit local lift/contact-loss/drop
labels and drop eligibility. Frozen expert/RMS hashes checked before/after.
This is Cm-off randomized behavior collection, not trained-actor comparison.

## Learning and measurement design

Partition all records using pre-action motion/start-frame groups, fixed
stratified seed9331, proportions60/20/20 fit/calibration/holdout; all windows
and duplicate initialization groups stay together across simulator seeds.
Save the split before fitting and never use holdout targets for normalization,
model/threshold selection or calibration. At least24 distinct episode groups
per split, three motions represented, at least15 factual rows per arm in fit
and holdout. Unsupported drop classes are explicitly unlearned, never zero risk.

Nonlinear history+numeric-action Cm predicts supported lift, contact fraction
and conditional drop probability. All predictors also receive the same current
base intent18D and pre-action motion/phase4D: the8step base continuation depends
on reference-policy intent, not physical state alone. No future realized action
or target enters this context. Same update budget for state-only six-slot
outcome heads (no numeric current candidate), action-shuffled Cm (independent
fit-row action corruption, labels retained), always-base and fit-only best
fixed expert. No expert-ID input in numeric-action Cm; physical feature
normalization and uncertainty thresholds use fit/calibration only.

Held-out factual error/calibration and randomized policy value are distinct.
Policy value uses recorded p=1/6 with episode/group uncertainty, reports
matched-row coverage and effective episode counts. Do not call unobserved
individual action labels true ranking/regret. Report state-dependent predicted
choices, selected-action estimated uplift, interventions/abstention, contact
loss and eligible drop risk. Model uncertainty may abstain to base; abstention
cannot be called successful coverage.

Minimum continuation screen for direct-loop engineering/Probe: held-state Cm
estimated supported-lift advantage>=0.5mm over each declared control, contact
fraction no worse by>.05, all-state drop no worse by>.05; intervention5–80%,
at least32 factual policy-matched windows from at least10 distinct episodes,
and finite/calibrated heads with documented support. This is a separate
learned-selector screen, not a relaxation of slot1's2mm candidate-opportunity
gate. If predictive/coverage support is inadequate, do not label Cm useful or
begin PPO; use the evidence to choose representation/candidate/data repair.

Single admitted idle GPU,2CPU threads, total collection/fitting<=60minutes,
new artifacts<=8GiB including engineering attempts. Native process<=300seconds.
Unique owned outputs; original HF08 and six-expert checkpoints remain read-only.
No actor/V/PPO optimizer updates. Stop on budget, input drift or label/action
contract failure. A positive local Probe still requires independent direct
replanning measurement and ultimately matched trained-policy validation.

## Fixed fitting implementation before execution

Three ensemble members per learned variant, history GRU32, fusion width96/64,
explicit nonlinear state×numeric-action interaction;1000 Adam updates/member,
lr.001, batch64, frame-group bootstrap seeds9561–9563. Current numeric-action
and history/base-context normalization use fit only. State-only has6 output
slots and the same history/base intent/phase, without current candidate values.
Action-shuffled uses independent per-fit-row nonzero cyclic offsets seed9431,
not a single global permutation; holdout candidates remain real.

Loss: normalized supported-lift smooth-L1, soft contact-fraction BCE,
eligible-only drop BCE; no V or actor optimizer. Calibrate factual errors only:
margin=max(.5mm,calibration median absolute lift error). Candidate gain minus
one ensemble standard deviation must exceed that margin; mean contact/drop
changes obey the fixed.05 guards, otherwise base. Ensemble effect dispersion
is a heuristic, not an identified/certified causal effect interval. Unsupported
drop classification always abstains in already lifted states. Support requires
fit>=20 lifted rows/3drops, calibration>=10 lifted rows/2drops; calibration
screen contact MAE<=.20 and held factual lift90%interval coverage>=.80.

Collection COMPLETED:1984 windows/275 distinct first episodes,193 lifted
eligible windows and28drop events; all action/label/input checks pass. Split
uses78 distinct motion/start-frame groups. Fit1163windows/158episodes,
calibration428/61, holdout393/56; lifted/drop support154/18,18/3,21/7.
This is descriptive label/support audit, before any model fitting or held
policy selection. All original input hashes unchanged; GPU collection279.90s,
15.9MB, no actor/Cm updates during collection. Full traces remain owned output.
