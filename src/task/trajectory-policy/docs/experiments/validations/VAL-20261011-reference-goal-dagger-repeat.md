---
schema: ref2dex.validation.v2
validation_id: VAL-20261011-reference-goal-dagger-repeat
experiment_id: VAL-20261011-reference-goal-dagger-repeat
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: 11c1dd9c
claim_id: C3
hypothesis_family: HF-trajectory-policy-reference-goal-dagger
frozen_method_commit: 3f07b895
development_seed_pool: validation.development
validation_seed_pool: validation.holdout
development_seeds: []
validation_seeds: [400, 401]
matched_control: same reference joint warm-start and fixed3000update offline joint continuation; no queried student-state PD data
status: SUPPORTED
run_id: reference-goal-dagger-validation-20261011-s400
checkpoint_sha256:
  tau_only: 4b646e964ec6d517859a2c9c710aa1e168e22569a8ebd536dab3ab583e8a6c64
  joint: 7c2d1fe98a27c7c3ba72ba44a217c631aea5e46b0c74384a0cc7c9d4f692677e
retargeter_sha256: 414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f
reference_sha256: 1f8d3f17a321738ae512a18b4df3eddc7618e9de7ed0fa6e327393103aada52a
---

# Frozen reference-conditioned PD DAgger: two holdout launches

## Decision Note / motivation

Decision experiment for Mission Objective A: is the positive single-reference
seed186 Probe reproducible enough to retain this autonomous baseline before
Cm/PointWorld work? User explicitly requests continued validation. Prior audited
DAgger8/8 placement, medianheld376, no clipping and reference position RMSE7.57cm,
versus offline continuation6/8, median92, RMSE75.69cm. Freeze both selected models
and run two independent native launches with predeclared holdout seeds400,401.
No retraining, model selection, new aggregation or reference change. Positive:
retain fixed models for the next baseline stage; inconsistent/negative: first
inspect control and launch variability, keep the positive Probe scoped to seed186.
Local repeatability claim only; no change to Mission/global claim or Campaign.

## Frozen method / comparison

Same audited `outputs/trajectory-policy/reference-goal-dagger-fit-20261011-r1/`,
same R and raw planned reference as the originating Probe. H328/G408 with
human/object pose anchors t+1,t+16 ->24x33 hand trajectory; per-control query;
frozen R receives live measured state. Expert intervention OFF, no expert action
or measured future object/q labels enter student inputs. Official teacher only
controls teacher rows; actual-hand oracle only enters GT-R rows.

Each seed uses24env/542controls/543states, fixed layout teacher0:4,GT-R4:8,
offline-control8:16,DAgger16:24, same source donors24:28. All arms share the same
native launch, assets/reference/controller/dt. Record initial-state differences;
do not assume environment rows are independent or perfectly paired physical twins.
This checks the two frozen models, not a new DAgger training-seed replication.
Original arms were both trained3000updates; selected offline3000/DAgger250 using
their predeclared objectives, including extra query validation for DAgger.
Extra labels and different selection criteria limit causal method attribution.

## Fixed metrics and verdict (before results)

Per-seed control calibration: teacher and GT-R each>=3/4 hold45 and controlled
placement. For EACH student arm apply the original gate without modification:
hold45>=6/8; placement>=6/8; clipping<1%; medianheld>=.8*GT-R median;
object reference position RMSE<=1.5*teacher; reference rotation mean geodesic
<=1.5*teacher+.1rad. Full-wave metrics include failures, no episode removal.

Primary claim: frozen DAgger passes all six readiness gates on BOTH seeds.
SUPPORTED iff both launches are calibrated/audited and both DAgger gates pass.
REFUTED iff both launches calibrated/audited and both DAgger gates fail.
INCONCLUSIVE for mixed passes, failed calibration, missing run or resource stop.
INVALID_IMPLEMENTATION for authenticated input/controller/feature/metric errors.
Single-run producer labels remain Probe-style observations; this card/aggregate
alone records the multi-run Validation verdict. Do not pool the discovery seed186
into the primary outcome.

Secondary descriptive comparison: relative advantage is repeated only if primary
SUPPORTED, DAgger RMSE<=.8*offline RMSE AND medianheld>=1.25*offline medianheld
on both seeds. Otherwise no verified advantage claim, even if baseline readiness
passes. Report placement/hold counts and each seed separately; no significance
claim from clustered16episodes or two seeds. Exact target pose, human-hand match,
full classical DAgger guarantees and Cm causal utility remain untested.

## Resources / stopping / provenance

One idle GPU, prefer3; inspect PID2024475 idle context again (<512MiB,<=5%util),
never modify other processes. Remaining shared competent-route20GPUmin/1GiB:
producer elapsed~729s excluding duplicated summary recovery; historical audits
and diagnostics included conservatively in a<=15.5GPUmin pre-run allowance.
This validation<=4.5GPUmin and<=190MiB new outputs. Each native launcher gets a
120s wall cap (TERM after115s, owned-process cleanup5s); each replay audit<=15s.
Stop on foreign PID/context>512MiB, nonfinite, early reset, drift or wall/storage
cap; preserve aborted records and do not replace a failing seed. Run BOTH seeds
when safe even after scientific failure; no adaptive seed selection or extra run.
GPU and log records in each run; current own outputs688.2MiB excluding symlinks.
Repo-owned tmp and outputs, main, no new branch/push or Campaign edits.

## Results / artifacts

run_status: COMPLETED
conclusion: SUPPORTED (only the two-launch frozen-model readiness claim)
repeated_descriptive_advantage: true

Both prespecified seeds completed without teacher interventions, resets or input
changes. Full GPU replay audits PASS, including actual PD exact equality and
feature/model/R maximum errors<1e-6. Independent CPU aggregation recomputed all
physical outcomes, reference metrics and six readiness gates. Both DAgger runs
passed every gate; offline continuation failed readiness in both. Seed186 discovery
measurements were not pooled into this Validation panel.

| Seed | Arm | Hold45 | Controlled placement | Median held frames | Clipping | Reference position RMSE | Reference rotation mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | Teacher | 4/4 | 4/4 | 436.5 | 0% | .07662m | .43487rad |
| 400 | GT-R | 4/4 | 4/4 | 436 | .0923% | .07808m | .46515rad |
| 400 | Offline continuation | 7/8 | 5/8 | 92 | .1845% | .80776m | 1.12013rad |
| 400 | PD DAgger | 8/8 | 8/8 | 376 | 0% | .07579m | .49841rad |
| 401 | Teacher | 4/4 | 4/4 | 438.5 | 0% | .07322m | .46135rad |
| 401 | GT-R | 4/4 | 4/4 | 436.5 | .4613% | .07837m | .44820rad |
| 401 | Offline continuation | 6/8 | 6/8 | 92 | .3921% | .76737m | .95382rad |
| 401 | PD DAgger | 8/8 | 8/8 | 405 | 0% | .07525m | .50784rad |

The two prescribed secondary descriptive checks also hold on each seed:
DAgger reference RMSE<=.8*offline and medianheld>=1.25*offline. Aggregate placement
counts16/16 versus11/16 are descriptive, not16independent random trials or a
population-level significance claim. The primary gate duration thresholds are
348.8/349.2frames and position thresholds .11492/.10983m respectively.

Critical scope check: measured q/object/hand/velocity at reset are EXACTLY the
same between the two seed launches AND between corresponding student rows.
These are reproducibility checks at one frame0 initial condition with native
contact/launch variation, not randomized initial-state or perturbation robustness.
Different outcome traces/hashes document real independent physics launches;
seeding alone did not produce a new initial-state distribution.

s400 producer11c1dd9c,63.88s,trajectory SHA
1873d30fc9cf3102e2089e794ba8e64dede44a0f203f7613ab7a805b87707889.
s401 producer6a228233,61.74s,trajectory SHA
0eba2b87e386e7d25e59d9d20f953cea4b04dd875e513320722e3931fd730eba.
All model/R/reference hashes match the predeclared values. Audit reports bind
measurement, checkpoint and manifest hashes; frozen pre-result protocol is
preserved as `protocol_frozen.md` in the summary. Later results appended here
are authenticated against the original producer Git blob when replaying.

Native GPU3 monitoring: own memory18.1GiB, total18.24GiB including inspected130MiB
idle context; active control utilization39-54%, each ETA updated inmonitor.jsonl.
125.62s producer GPU wall time plus two bounded<=15s replay audits, within4.5GPUmin.
New outputs156.8MiB, shared route total~845MiB<1GiB; no retraining/newcheckpoint.
All own GPU jobs ended. CPU used only for file/hash/statistical aggregation and
plotting; all repeated neural/simulation replay used GPU.35 Task tests PASS.

## Decision after Validation

Retain frozen PD DAgger as the fixed-reference/frame0 candidate baseline. This
removes the single-launch reproducibility blocker for same-task Cm/PointWorld
integration design; stop no-Cm aggregation/training sweeps. A formal Cm-on/off
training comparison still needs its own fixed protocol and evaluation, and these
measurements do not prove RL benefit. Initial-state perturbations, held-out motions
and replicated training seeds remain future evidence, not claims of this card.

Actual outputs:
`outputs/trajectory-policy/reference-goal-dagger-validation-20261011-s400/`,
`outputs/trajectory-policy/reference-goal-dagger-validation-20261011-s401/`;
each manifest/result/trajectory/audit/behavior curve. Summary and cross-seed
comparison at `outputs/trajectory-policy/reference-goal-dagger-validation-20261011-summary/`.

## Limitations / future evidence

One trained pair, same airplane reference, frame0 reset and two holdout launches.
No held-out motion, perturbation-recovery or training-seed generalization. Native
contact nondeterminism and environment-row effects remain possible. Broader tests
only after this repeat affects the next decision; do not block Cm work on an
unlimited no-Cm baseline sweep once readiness is adequate.
