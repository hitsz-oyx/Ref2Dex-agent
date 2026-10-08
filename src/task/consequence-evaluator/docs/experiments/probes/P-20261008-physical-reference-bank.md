---
schema: ref2dex.probe.v2
probe_id: P-20261008-physical-reference-bank
experiment_id: P-20261008-physical-reference-bank
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 340b847
claim_id: C3
hypothesis_family: HF-consequence-physical-reference
probe_index_in_family: 1
seed_pool: probe
seeds: [230, 261, 281]
decision_changed_if_positive: retain physical references and prepare fresh same-state GT-value Gate1 after event checks
decision_changed_if_negative: stop using phase delta as primary GT and probe transition alignment from causal current phase
status: PROMISING
run_id: physical-bank-tcc-20261008-r1
---

# Does measured successful robot reference geometry fix local phase delta?

Result: All four clean tick176 windows become positive; clean negative windows
drop to zero and every frozen development check passes. No evaluator fit.
Decision: Retain physical reference bank for a same-state GT-value Gate1 Probe;
do not switch to transition alignment based on this positive development panel.

## Decision Note and purpose

This Decision Probe supports consequence-evaluator Gate0: define useful GT
candidate value before the full-chain Gate1, evaluator, PointWorld and execution
model. It asks whether original kinematic R versus physical execution is the
main obstacle to locally useful delta-progress. The prior TCC Probe reaches
clean finalP0.932..0.994 yet total backwards0.249..0.368, with all four clean
tick176 windows negative despite another lift. Training loss alone does not
establish value. User ref4_2 explicitly replaces the earlier original-R-only
choice with measured successful robot reference bank. This changes a local
supervision implementation, not the Mission's final policy utility claim.

Cheapest method: reuse exactly the previous eight source230weak-success nominal
train views, freeze a bank, train one1000update encoder and relabel the same
eight already-exposed source261development episodes. Do not tune on val/test.
If local delta remains unreliable, stop this GT definition and move to the
specified transition-alignment Probe; do not blindly lengthen training.

## Frozen protocol

Build bank from the first eight recorded clean train230episodes passing the
existing full-reference weak physical success check: e0/e1/e2/e3/e4/e5/e6/e8.
Qualification may inspect complete training demonstrations; no target-episode
outcome, actual future or validation/test sequence is consulted. This weak
qualification is stricter about intermediate loss than the user's final
recoverable episode success definition; it only selects conservative references,
does not redefine Gate1 outcomes. No new S/P/M targets. No ideal-FK substitution.
Fixed543state/30Hz geometry uses measured11hand points and object poses.

Same90D causal features,8frame context, group scaling recipe fitted solely to
the eight bank references, same MLP720/128/128/64/unit embeddings, seed281,
AdamW3e-4/weight_decay1e-4,256samples/view and1000updates. Random distinct
physical-bank pairs use upstream deterministic TCC regression_mse. Original
kinematic reference is excluded from this fit; old runs remain untouched.
Temperature0.1/max_step4/transition_scale1.5/epsilon0.01 remain unchanged.

Each bank member gets its own causal bounded history prior. Define P as the
fixed uniform mean of eight soft normalized reference indices. Equal lengths
make mean distribution explicit; store member P as well to expose disagreements.
No best-reference selection by full actual episode, no future-driven weights,
no monotonic projection. Y=P[t+24]-P[t] still sees only through t+24.
No second context over embeddings. Fixed64row batches preserve prefix numerics.

Use the same assigned four clean/four contact source261 episodes. They were
exposed in previous method development and are not untouched test data. Main
focus is all four clean tick176 Y: require >=−epsilon0.01. Retain existing
clean final>=0.85/backwards total<=0.2, bank-member0 self final>=0.95/MAE<=0.05,
static final<=0.05, prefix error<=1e-9 and mean-step bound checks. Inspect failed
e1 stagnation and nominal normal-place windows; improved endpoint coverage alone
does not approve value labels. Distance cutoff1 is uncalibrated diagnostic,
not confidence or an operational abstention certificate. No same-state pairs
exist in this offline panel; Gate1 is a separate fresh-replay experiment.

One currently idle GPU1: <=300s fitting, <=180s label replay, <=0.5GiB artifacts.
Check occupancy before each GPU job, monitor memory/utilization and ETA. CPU is
used only for file/hash/weak-qualification audit and tiny contract tests; no
network training on CPU. Stop on drift, deadline, occupied GPU or nonfinite.
Unique outputs/checkpoint, no overwrites and no best-checkpoint selection.

## Limitations and future evidence

Switching to a physical bank changes anchor geometry, reference-only scales,
training pair distribution and averaging together. This is a cheap route
selection Probe, not a matched causal attribution of a single component. A
positive result only motivates GT-value Gate1; it cannot show policy benefit.
Bank members are weakly qualified, not manually verified, and share one motion,
embodiment and generator. Multi-seed/generalization/calibrated confidence and
same-state candidate utility remain future evidence. Failed bank phase delta
does not refute TCC as a current-phase conditioner for transition alignment.

## Observed result and next decision

All three runs use code340b847: `physical-reference-bank-20261008-r1`,
`physical-bank-tcc-20261008-r1`, `physical-bank-progress-labels-20261008-r1`.
Bank construction is CPU file/weak-qualification work,1.379s. Measured geometry
retains the planned eight source230members; no fresh rollout or test inspection.
Encoder fixed1000updates completed in10.116s on GPU1; final100update mean loss
0.0000780253, peak allocated40,137,216bytes. Unique final checkpoint SHA256
`ce9ecaa8d1a8e0ee08e8e057fb0f19b21d4aebddad38562de377982035c1a0c6`.
Label replay8episodes/496windows completed in27.379s, peak allocated17,875,456bytes.
GPU1 was released. These are short alignment jobs, not long policy training.

| Clean episode | Original-R tick176 Y | Physical-bank tick176 Y | Old backwards total | New backwards total | Old/new negative windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| e0 | −0.025145 | +0.043161 | 0.367527 | 0.002428 | 5/0 |
| e3 | −0.023862 | +0.043681 | 0.272944 | 0.000000 | 6/0 |
| e6 | −0.023319 | +0.043790 | 0.249442 | 0.002245 | 6/0 |
| e7 | −0.023619 | +0.044150 | 0.281448 | 0.001499 | 8/0 |

Clean finalP0.993818..0.994876. All260clean K24windows have positiveY>epsilon,
including the normal placing suffix; this is observed on the fixed panel,
not a monotonic constraint. Self finalP0.992555/indexMAE0.002442, stationary
finalP0.001965, actual prefix distribution error0. Each member and aggregate
obey the4/542step bound. Report verifies the exact same eight IDs and frozen
raw-source manifest across old/new runs. The original-R/physical-execution
gap hypothesis is **PROMISING as a route decision**, not formally SUPPORTED.

Contact e1 physically falls below initial support−3cm at tick89, then remains
on the floor and endsP0.023337 instead of reaching the end. Its requested
intervention window is tick51→75: object height at75 is still+38.775cm, surface
gap1.197cm, and Y is+0.010590. The major fall occurs after the label horizon;
do not use later failure to rewrite this causal local value. Endpoint separation
already starts within the window, however, so this positive Y is **not proof
that the action is safe or better than a same-state baseline**. Embedding OOD
distance is not calibrated: very late failed states can have distance<1.
Other contact finals0.935656/0.994994/0.621422 retain varied recovery/stagnation.

Artifacts: each run's manifest and source hashes; label traces with per-member
progress and `train-progress-examples.png`; CPU-derived
`physical-bank-progress-audit-20261008-r1/report.json` with all gate checks and
tick176/failure-window numbers. Existing original-reference runs remain intact.
44 relevant tests and repository verification pass. Only the successful
physical-reference route is retained; no parameter sweep or appended fit.

The specific normal-lift/placing local regression defect is resolved in this
small exposed development panel. Next is fresh complete-prefix same-state
candidate selection with independent recoverable grasp/normal-place outcomes.
**Gate1 has not run**; labels remain `training_allowed=false`, no evaluator or
policy trained, no success-rate improvement claimed. A short-horizon local
phase value can miss delayed hazards; Gate1 is the decision test for whether
relative candidate ranking still helps. Switch to transition alignment if that
test or a new fixed semantic panel reveals persistent local-value errors.
