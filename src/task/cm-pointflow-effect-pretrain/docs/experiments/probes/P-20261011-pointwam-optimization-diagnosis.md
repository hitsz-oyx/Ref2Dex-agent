---
schema: ref2dex.probe.v2
probe_id: P-20261011-pointwam-optimization-diagnosis
experiment_id: P-20261011-pointwam-optimization-diagnosis
date: 2026-10-11
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: 3bae27c2629a7bb57e6de23e272e08e7fa583091
claim_id: C1
hypothesis_family: HF-pointwam-joint-learning
probe_index_in_family: 2
seed_pool: probe
seeds: [7]
decision_changed_if_positive: run one matched learner variant with the corrected optimizer scale and assess the frozen dev gate
decision_changed_if_negative: preserve the verified data contract and stop spending budget on this small substitute learner
status: UNPROMISING
---

# Optimization diagnosis for the bounded PointWAM learner

Class: Decision. The preceding joint-learning Probe reached only a 23.23%
scene fit-loss reduction at `1e-4`, while a single-clip diagnosis changed both
the clip population and learning rate and reached 91.01%. This card separates
those factors before changing the architecture or acquiring more data.

The cheapest red-capable loop is
`tools/audit/diagnose_pointwam_optimization.py`. It reuses the frozen
qualified manifest and fit-only normalization, checks all finite losses and
gradients, and never reads dev targets during training. The corrected run
`pointwam-optimization-diagnosis-20261011-r2` keeps the original continuous
RNG stream. A later `r3` adds pair-fixed point IDs to distinguish point
resampling noise while retaining two-clip batches.

Result: `UNPROMISING` for the optimizer-only variant; fit scene reduction is
41.8% and held-dev scene/hand gates fail despite exact replay.

Decision: retain the verified data contract and stop this small substitute
learner route until a new architecture or source-coverage decision is made.

## Ranked hypotheses and predictions

1. **Learning rate is too small.** With the same model, seed and six-clip
   target contract, `1e-3` should reduce fit loss much more than `1e-4`.
2. **Point resampling adds destructive gradient noise.** Fixed point IDs per
   clip should improve fit loss at the same learning rate and two-clip batch.
3. **The current architecture cannot represent the joint mapping.** Even a
   fixed six-clip fit at `1e-3` should fail the 50% scene and hand reductions.
4. **A data or mask contract bug dominates.** At least the single-clip fixed
   arm should fail to fit despite optimizer changes.

## Frozen diagnosis protocol

- Source: six fit clips from `pointwam-train-qualification-20261011-r1`;
  all pack hashes and the original fit-only contract are rechecked.
- Single-clip arms use the first fit clip and 512 fixed initial scene IDs.
- Multi-random arms preserve the original two-clip/1024-point sampling rule;
  only learning rate differs between `1e-4` and `1e-3`.
- Pair-fixed arms preserve two-clip batches and replace only point-ID
  resampling with one fixed 1024-point subset per clip.
- Full-fixed arms remove both clip and point sampling noise for an engineering
  upper-bound check. None of these arms reads dev labels.
- AdamW, weight decay `.01`, gradient cap `1`, seed `7`, and 600 updates are
  unchanged. The output is fit-only diagnosis, not a method or policy claim.

## Results

The corrected `r2` and `r3` runs completed on physical GPU7; peak allocation
was below 54 MiB and each run finished within 94 seconds. The key ratios are
final fit loss divided by its fixed-ID initial loss:

| Arm | Scene ratio | Hand ratio | Interpretation |
| --- | ---: | ---: | --- |
| single fixed, `1e-4` | 0.524 | 0.00023 | scene just misses the 50% gate |
| single fixed, `1e-3` | 0.090 | 0.000019 | single-clip fitting works |
| multi random, `1e-4` | 0.791 | 0.427 | too slow/noisy at the old scale |
| multi random, `1e-3` | 0.520 | 0.337 | near the scene gate; sampling remains noisy |
| pair fixed, `1e-4` | 0.791 | 0.450 | fixing IDs does not rescue the old scale |
| pair fixed, `1e-3` | 0.492 | 0.161 | small improvement, barely passes scene gate |
| full fixed, `1e-4` | 0.591 | 0.066 | batch/point averaging helps, old scale still slow |
| full fixed, `1e-3` | 0.380 | 0.010 | six-clip substitute can fit with stable batches |

The diagnosis supports hypotheses 1 and 2 partially: optimizer scale is the
dominant change, while point resampling contributes smaller noise. It does not
show a contract failure or prove that the architecture can generalize.

## Matched learning-rate result

The pre-registered matched variant `pointwam-joint-learning-lr1e-3-20261011-r1`
changed only AdamW's learning rate from `1e-4` to `1e-3`; it retained the
original two-random-fit-clip schedule, 1024 points, seed 247, 600 updates and
the original fit/dev gates. It completed on GPU7 in 19.2 seconds with 60.77 MiB
peak allocation, finite gradients and exact checkpoint replay.

| Metric | Initial | `1e-3` final | Gate |
| --- | ---: | ---: | --- |
| fit scene loss | 6.02475 | 3.50641 (41.8% reduction) | fail, needs 50% |
| fit hand loss | 1.00000 | 0.35183 (64.8% reduction) | pass |
| dev hand EPE | 94.112 mm static | 95.530 mm model | fail |
| dev moving-scene EPE | 32.652 mm static | 71.724 mm model | fail |

The model improved on 2/3 dev records for hand EPE, but the aggregate hand
gate and the moving-scene gate failed. Raising the learning rate therefore
overfits the tiny fit cohort and does not recover held-record scene prediction.
This optimizer-only recipe is `UNPROMISING`; it does not refute the PointWAM
data contract or the full paper architecture.

## Decision

Retain the qualified data and the verified scene/hand contract, but stop
spending this Probe's budget on learning-rate or point-sampling variants of the
195k-parameter substitute. The remaining gap is representation/capacity and
small-source generalization, while the faithful Mosaic3D/Recap-CLIP/Volt
encoders are not reproduced here. Any further investment should first have a
new explicit architecture or source-coverage decision; do not expand data or
claim PointWAM/policy benefit from these runs.

## Matched variant artifact

[learning-rate variant manifest](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-joint-learning-lr1e-3-20261011-r1/manifest.json)

## Closure

There is no further run authorized by this card. A future architecture/source
decision may reopen the route, but no policy or Cm claim follows from this
Probe.

Artifacts:

- [corrected r2 manifest](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-optimization-diagnosis-20261011-r2/manifest.json)
- [pair-fixed r3 manifest](../../../../../../outputs/cm-pointflow-effect-pretrain/pointwam-optimization-diagnosis-20261011-r3/manifest.json)
