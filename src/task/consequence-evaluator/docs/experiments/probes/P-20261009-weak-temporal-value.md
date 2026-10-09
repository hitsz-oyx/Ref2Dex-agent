---
schema: ref2dex.probe.v2
probe_id: P-20261009-weak-temporal-value
experiment_id: P-20261009-weak-temporal-value
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 5d13cf1
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

Result: initial offline screen UNCLEAR; completed matched 1200-step fit.
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

## Initial results and bounded audit

Training commit `5d13cf1`, runtime Torch2.4.1+cu121, physical GPU2, 26.66s,
peak reported memory561MiB. Val selected step400 for both arms; all1200updates
completed. Prepared labels preserve train41/23 and val38/26; test becomes37/27
after `s263_w1_e28_airplane` recovers and completes. Masked positive recovery
windows: train0, val31, test17. All711 source hashes remained frozen.

| Frozen test metric | HA | HAZ | HAZ future-shuffle |
| --- | ---: | ---: | ---: |
| Same-time cross-episode ranking | .88423 | .89666 | .84316 |
| Balanced outcome sign accuracy | .88247 | .88738 | .83588 |
| Episode-average MSE | .011396 | .011044 | .028155 |

Future gain1.24pp misses the frozen3pp screen, despite a5.35pp shuffle drop.
Do not promote this to Gate1 or expand training. A post-freeze semantic audit
checks time strata, active-plan coverage and positive recovery support. It may
also score the already cached first seven same-H GT forks (25anchors) with
frozen checkpoints: one idle GPU2, <=60s, no simulation, no model selection,
no extra fitting or online execution. The cached forks have only32-step future
coverage and a different self-trained actor/backend; their scores have no
full-task outcome ground truth and cannot override the failed offline screen.
This audit decides whether the next useful information needs better data rather
than more epochs. Existing raw results/checkpoints remain unchanged.

Artifacts: `outputs/consequence-evaluator/weak-temporal-labels-20261009-r1/`,
`outputs/consequence-evaluator/weak-temporal-value-20261009-r1/` and its `.log`.

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
