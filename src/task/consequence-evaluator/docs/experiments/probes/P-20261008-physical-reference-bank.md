---
schema: ref2dex.probe.v2
probe_id: P-20261008-physical-reference-bank
experiment_id: P-20261008-physical-reference-bank
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 1b24c5b
claim_id: C3
hypothesis_family: HF-consequence-physical-reference
probe_index_in_family: 1
seed_pool: probe
seeds: [230, 261, 281]
decision_changed_if_positive: retain physical references and prepare fresh same-state GT-value Gate1 after event checks
decision_changed_if_negative: stop using phase delta as primary GT and probe transition alignment from causal current phase
status: PLANNED
run_id: physical-bank-tcc-20261008-r1
---

# Does measured successful robot reference geometry fix local phase delta?

Result: Pending bounded bank/encoder/label Probe; no evaluator fit.
Decision: Follow user ref4_2 using existing successful physical trajectories,
without tuning the original matcher parameters or collecting new data.

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
