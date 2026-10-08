---
schema: ref2dex.probe.v2
probe_id: P-20261008-value-policy-error-interventions
experiment_id: P-20261008-value-policy-error-interventions
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 8245f53
claim_id: C3
hypothesis_family: HF-consequence-value-interventions
probe_index_in_family: 2
seed_pool: probe
seeds: [236, 237, 238, 246]
decision_changed_if_positive: prepare a split-disjoint outcome dataset using the frozen learner-error holding sampler
decision_changed_if_negative: stop this action-error sampler and diagnose action authority or labeling before further collection
status: RUNNING
run_id: official-value-policy-error-train-20261008-r1
---

# Does same-H learner/expert control error supply outcome negatives?

Result: Pending same-H GPU replay check and bounded train collection.
Decision: Continue the authorized data-generation task with a measured policy
error sampler after the [placing sampler](P-20261008-value-place-interventions.md)
failed coverage (32clean successes,31/32perturbed successes).

## Decision and fixed protocol

Question: Does a frozen empirical learner/expert error bank at a geometrically
stable unsupported holding stage produce informative complete task failures,
while nominal controls remain successful? This is a Decision Probe: at least
24actual full triggers,24clean successes and4train negatives permit independent
Probe val/test collection. Otherwise stop expansion and audit the mechanism.

Only nominal train seed230 supplies bank inputs. Select the first32source
episodes with a complete24observation segment beginning at their first45frame
geometric held completion and ending before final placing. Compute actions from
the official frozen actor SHA8f6823db… and the current self-trained airplane
endpoint SHA8882fabd… on **the exact same raw H**, with each actor's own saved
RMS applied once. No learner/expert comparison on different simulated states.
The feed-forward projection must reproduce logged nominal expert controls with
max absolute error<=1e-5 or abort before sampling. Read only frozen checkpoints;
no actor fitting or checkpoint creation.

Build empirical24step error candidates from learner minus expert normalized
actions. Four-bin median/interpolation/sin-squared taper, zero endpoints,
gains0.25/0.5/1, residual cap0.2, translation cap0.05, zero Inspire-overwritten
distal channels.96candidates; uniformly pick one per episode. This borrows
DART's actual learner/expert error basis; it does not fit Gaussian covariance,
reproduce DART/CARE, or establish that candidate directions cause failure.

Use the same official actor, full airplane reference, task labels and geometry
as before.64train episodes at236 (32env/two waves, halfclean/halfhold). Hold
trigger requires45continuous measured near-hand unsupported elevated frames,
not5force-proxy frames. One immutable24step chunk and the same expert afterward,
no forks/resets. Pre-trigger unknown windows are excluded. If coverage passes,
collect64val at237 and64test at238 using the same bank, without tuning from
their outcomes. Pair seed246/stride8; minimum unique positive/negative4/2/2.
Probe seed groups are not formal Validation evidence. Old H-match headroom
schema and final self-trained-policy Mission remain unchanged.

## Resources and stops

Same-H inference uses one idle GPU, <=120s, batch768observations and <=1GiB bank.
GPU2collection: at most900s total across<=3runs, hard250s each, raw<=2GiB expected
total (existing2GiB per-run guard). CPU labels/statistical work<=120s each, <=1GiB
windows/audit. Commit before collection, freeze bank/source hashes, watch GPU
occupancy/memory and stop on drift, collision, incomplete/nonfinite data or
exhausted budget. No external writes/checkpoint/environment changes or push.

## Limitations and future evidence

This samples32nominal observations from one train seed/object/reference and
one weaker frozen learner; it is an empirical policy-error basis, not a fitted
general failure distribution. Source labels are weak geometry/support proxies,
not manually confirmed contact truth. Class coverage alone does not validate
an evaluator, prove oracle information gain or demonstrate causal policy utility.
Success classification, stage-balanced sampling and independent full-reference
manual spot checks remain necessary before formal model comparison.
