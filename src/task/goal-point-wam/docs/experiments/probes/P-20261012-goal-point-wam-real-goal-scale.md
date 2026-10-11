---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-real-goal-scale
experiment_id: P-20261012-goal-point-wam-real-goal-scale
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 1310c47ea9914bd95a9e861efc4e55c53406cf7c
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 9
seed_pool: probe
seeds: [247, 248, 249]
decision_changed_if_positive: retain scaled goal units and investigate current-hand spatial anchoring on real data
decision_changed_if_negative: stop units-only tuning and inspect real geometric conditioning before longer training
status: UNCLEAR
run_id: real-goal-scale-20261012
---

# Real-data goal coordinate scale comparison

Class: Decision. Serves trajectory representation for the eventual RL/Cm claim.
Conjecture: the analytic conditional-learning benefit of .05m input units
transfers to real goal-dependent trajectories. Full synthetic teacher gates
failed; this Probe tests a narrower positive component directly, without
claiming a passed engineering gate or physical validity of synthetic labels.

Result: UNCLEAR; matched units-only training gives no real-data trajectory improvement and negligible goal use.
Decision: Stop units-only and analytic tuning; prioritize current-hand spatial/history conditioning before longer training.

## Protocol fixed before execution

- Fresh seed247 WAM heads and strict PW step50000 scene transfer. NO analytic
  checkpoint, real fitted WAM parent or artificial teacher labels. Same measured
  OakInk2/GRAB/ARCTIC train corpus/stats, real object goal with no future hand goal.
- Endpoint point-set encoding, all other model/optimizer/loss settings identical.
  Only goal_coordinate_scale_m1 versus .05. Identical initial parameter hashes,
  normalization/data identities, train sample stream and96-window balanced val
  indices. Train seed248, val seed249. No test split or extra panel construction.
- Each arm600 stochastic updates, microbatch2, AdamW lr.001/decay.01,50 warmup
  cosine to10%lr, world_weight1, BF16, grad cap1. Existing val every100 updates.
  Decision uses FINAL600, not retrospectively chosen best checkpoint.
- Frozen final96-window val audit reuses identical scene per true/static-goal
  input. Compare true real goal versus replacing endpoint by current object
  start; no physically labeled counterfactual is claimed. Also report zero hand
  displacement baseline and endpoint error. Cached ablation changes trajectory
  inputs only; world branch remains goal blind.
- PROMISING screen requires >=2 same sources: scaled trajectory EPE at least5%
  below raw, scaled true-goal EPE at least5% below its static-goal ablation and
  zero-hand baseline. Scaled moving-world h24 EPE must not regress >5% versus
  raw on any source with moving objects. Otherwise UNCLEAR; report components.
- Check exact matched initial parameters/data/stats/validation indices and
  full600 completion before comparing. Record model train-mode flags/GPU samples.
  Stop on nonfinite, OOM, identity mismatch, competition or600s training/arm.
- One GPU2, sequential arms. This turn's shared cap <=20GPU-minutes,30minutes
  wall, <=5GiB new artifacts against300GB project cap (~258GB before turn).
  Existing diagnostic artifacts about1.2GiB; real checkpoints <=3.6GiB. No new
  branch, dataset download, robot/RL, video, or external mutation.

## Decision Note

Root selects units-only real-data comparison after near-unit analytic response
and little added benefit from paired displacement. This tests real supervised
use rather than extending the teacher budget. Failure closes units-only tuning
at this budget, not object-goal/WAM/Cm. Positive would justify current-hand
spatial conditioning; matched world-loss-off and RL remain deferred. Core claim
and resource boundaries unchanged, no additional authorization needed.

## Secondary geometric reference, fixed before frozen ablation

Before observing frozen goal-ablation metrics, add two no-training references:
linear interpolation of target object centroid translation applied to current
hand, and linear interpolation toward the current hand transformed by the
paired goal Kabsch rigid transform. Neither reads future hands; both use exactly
the intended object goal and current hand. These are crude information screens,
not physical controllers; all original gates remain unchanged. If they reduce
error while the network does not, prioritize explicit current-hand/object-goal
geometric conditioning over more units/epoch tuning. Contract test checks a
known90-degree rotation+translation and masked NaN payloads.

## Limitations / future evidence

Single seed, short training and96 val windows are a Probe, not Validation.
World generalization and real object goal may be ambiguous with respect to hand
motion. True/static-goal ablation measures information use but has no valid
alternative hand labels. No extrapolation to stable grasping or RL sample efficiency.

## Artifacts

- [Raw training results](../../../../../../outputs/goal-point-wam/real-goal-scale-20261012-raw/result.json)
- [Scaled training results](../../../../../../outputs/goal-point-wam/real-goal-scale-20261012-scaled/result.json)
- [Raw goal ablation](../../../../../../outputs/goal-point-wam/real-goal-scale-20261012-raw/object_goal_validation.json)
- [Scaled goal ablation](../../../../../../outputs/goal-point-wam/real-goal-scale-20261012-scaled/object_goal_validation.json)
- [Frozen evaluation](../../../tools/audit/evaluate_object_goal.py)

## Completed results and matched implementation review

Training code1310c47e; frozen audit code205f6be8. Both arms completed600
updates on GPU2 in302.31/301.40s. Exact initial parameters, data/stats/model/
trainer identity,96 validation indices and actual validation window identities
match. Config difference is ONLY goal_coordinate_scale_m1 versus .05. Sampler
precomputes1200 train indices with seed248 before updates, using the same data
and sampler code. Every logged training-mode flag is true; no artificial labels
or weights enter these runs. Strict checkpoint/stats/data/model reload succeeds.

| Source | Raw / scaled EPE (mm) | Zero-hand EPE (mm) | Translation / rigid-follow EPE (mm) |
| --- | ---: | ---: | ---: |
| OakInk2 | 35.59 / 35.67 | 34.80 | 35.87 / 36.25 |
| GRAB | 141.64 / 141.69 | 141.32 | 146.51 / 178.66 |
| ARCTIC | 71.86 / 71.88 | 71.63 | 68.70 / 69.12 |

Scaled changes versus raw are -.209%/-.030%/-.024% in trajectory improvement
(negative means worse). No source passes the5% improvement screen. Both models
are slightly worse than zero hand displacement. True versus static-goal inputs
change the raw predictions by2.73/3.10/3.16 micrometers on average and scaled
by4.48/7.24/6.27 micrometers. True-goal EPE improvements over static-goal
inputs remain below.0004% in scaled arm. All three conditional source gates
fail; moving-world non-regression gates pass (scaled changes+.412%/+.007%/-.064%).
Whole-forward goal-world isolation maxima1.83e-7/9.42e-7, both below1e-5.

The object-motion geometric references do not beat zero displacement by5% on
multiple sources; only ARCTIC improves modestly. They cannot be adopted as a
general hand policy, and these numbers do not prove that object goal is
inherently insufficient. The task can involve hands moving independently of
the target object and multiple possible paths to the same endpoint.

Overall **UNCLEAR**, a single-seed short real-data Probe. Scaling makes the
analytic teacher easier but does not rescue real supervised prediction here.
The conditional branch can respond under deliberately informative labels; no
missing goal input or disconnected gradient has been found. Current real-data
behavior is consistent with weak condition learning / a near-stationary predictor,
but this run does not isolate spatial encoding, optimization or supervision
ambiguity as the sole cause. Do not close the WAM/Cm route or assert RL utility.

## Final Decision Note and resource closure

Retain the optional diagnostic units and paired-geometry implementations, keep
historical default endpoint/meter configuration, and stop units-only/teacher
tuning. Next minimum decision should test direct current-hand spatial/history
conditioning against the same real-data control; check a hand-history forecast
baseline before spending on longer full training. World-loss-off is a subsequent
matched question, not inferred from these results. No new training is launched
after this bounded sequence. Mission/claim/permissions remain unchanged.

11 contract tests passed/1 optional GPU test skipped. Tiny CPU contract tests
avoid GPU startup overhead; all training and frozen model evaluation use GPU2.
Real training sampled utilization5--25%/6--37%, memory maximum1513MiB. The
small fixed microbatch is intentional for this matched Probe; no batch tuning
is mixed into the comparison. All owned jobs completed; run artifacts and
original weak results are preserved. Shared turn artifacts remain under5GiB
and project artifacts under300GB. Actual new run artifacts4.68GiB; current
outputs+tmp about263.23GB. GPU2 has no remaining compute process.

- [Matched checks, gates and telemetry](../../../../../../outputs/goal-point-wam/real-goal-scale-20261012-scaled/comparison.json)
