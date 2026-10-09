---
schema: ref2dex.probe.v2
probe_id: P-20261009-weak-temporal-value
experiment_id: P-20261009-weak-temporal-value
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 0fdd3c5
claim_id: C3
hypothesis_family: HF-weak-temporal-value
probe_index_in_family: 1
seed_pool: probe
seeds: [261, 262, 263, 285, 286, 287]
decision_changed_if_positive: freeze the learned evaluator and screen GT-future candidate control with the self-trained actor
decision_changed_if_negative: audit weak-label and input semantics before any world-model substitution or larger training
status: UNCLEAR
run_id: weak-temporal-value-20261009-r1
---

# Can outcome/time supervision learn useful GT consequence value?

Result: pending.
Decision: user ref4_3 authorizes a bounded learned evaluator Probe, replacing
direct delta-progress as the value definition. The previous no-evaluator-fit
decision applies to that earlier progress pipeline, not this authorized route.

## Motivation / Decision Note

Mission C3 needs action-conditioned consequences to change policy decisions.
Question: can a learned Q(H,A,Z) extract task-relevant GT-future information
from success/failure and relative-time weak supervision, without a manually
weighted height/contact reward or fixed reference-phase value formula?

Reuse immutable192 full simulator episodes from the dose-calibrated outcome
data (official generator only, not final policy initialization). This is the
cheapest first test: no new sampling, no policy training, no PointWorld fitting.
Train data seed261, val262, test263 remain episode/group disjoint. All source
711 hashes passed preflight. Official-to-self-trained controller shift remains
a material limit for any later action test; this dataset does not prove transfer.

User clarified episode success: eventual regrasp/stable hold/controlled placing
counts as success. Positive-episode windows overlapping obvious loss/fall and
recovery until renewed stable45hold are marked uncertain and masked. Stable
placing/release is not loss. Final unrecovered failure episodes supply weak
negative labels, including their bad futures; this is not action imitation.
The physical cutoffs belong only to the previously fixed binary outcome and
uncertainty audit, never to an additive model reward or model input.

## Frozen model and labels

For complete K24 windows, target = sign(final episode outcome)*24/(T-t).
T/t and outcome are supervision/analysis metadata, never model inputs. No
S/P/M preference or stage/margin target enters this new schema. No TCC or
reference bank enters the model or defines its score.

H is raw native policy observation1442. A is the decision-known requested
24x18 residual plan under the recorded frozen feedback controller, preserving
ref2's interface; reactive future executed controls are excluded. Z is measured
24x45 object-relative rigid effect12 and object-frame11hand-point coordinates33.
This A is an explicit adaptation of Motus2's executable action-chunk interface.

Same128-wide two-layer transformer in HA and HAZ, identical initialization,
normalization (train only), balanced-episode/time sampler, AdamW3e-4, batch128,
maximum1200updates.41 signed bins, soft two-bin target, scalar expectation.
The categorical relative-progress supervision follows
[Motus2, §3.3.1](https://arxiv.org/html/2608.30237v1#S3.SS3.SSS1), whose
evaluator conditions on context/action/future. Our small geometry model and
requested-residual interface are adaptations, not reproduction of its model.

Checkpoint selection uses only val episode-averaged MSE. Test runs once after
both checkpoints are frozen. Report balanced sign accuracy, episode MSE and
same-time16frame-bucket cross-episode success/failure ranking. These are weak
label prognosis, not same-state causal candidate ranking. Future-shuffle uses
another test episode in the same time bucket, without selecting by outcome.

Decision screen: HAZ rank>=.65, rank gain over HA>=.03 and shuffle rank drop
>=.03. A negative screen is UNCLEAR about the underlying hypothesis. Audit
implementation and available future information before extending compute.

## Resource and stop conditions

One idle GPU2; CPU label/hash/statistical audit<=120s. Matched model training
<=600s, <=1200updates each, <=256batch and<=1GiB outputs. Monitor GPU memory,
utilization and elapsed/update ETA. No online control until inputs/labels and
offline usefulness are checked; conditional GT control gets a separately
recorded bounded budget. Stop on drift, nonfinite tensors, missing complete
windows, split leakage or resource conflict. No new branch, remote push,
external writes, checkpoint overwrite, reward engineering or actor training.

## Limitations / future evidence

Single-source groups, overlapping windows and cross-episode ranking cannot
establish formal Gate1 or Cm utility. Episode-final weak negatives can label
locally correct prefixes negatively; that noise is declared, not treated as
counterfactual ground truth. Positive recovery windows are explicitly masked.
Source dose/feedback controller/backend differ from the recovered e260 action
screen. Full-task controlled placing and historical Z90 are different outcomes;
Z90 alone cannot certify a model trained for final controlled completion.
Broader data, recovery auxiliary labels, full-episode matched control and
predicted-world replacement are deferred until this Probe changes a decision.
