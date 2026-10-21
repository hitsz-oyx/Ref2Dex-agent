---
schema: ref2dex.probe.v2
probe_id: P-20261021-wam-v6-geometry-consistency
experiment_id: P-20261021-wam-v6-geometry-consistency
date: 2026-10-21
task: ptv3-shared-wam-v6
branch: main
git_commit: b852f39cb016b2a04332c3ffd5a09037cc3ca7db
claim_id: C1
hypothesis_family: HF-v6-geometry-consistency
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain the geometric objective for fixed-sequence rollout comparison while monitoring physical-unit trajectory errors
decision_changed_if_negative: revisit geometry correspondence quality or gradient balance before expanding this method
status: RUNNING
run_id: geometry-consistency-20261021-r1
---

# Predicted hand/object contact geometry consistency

Decision Probe explicitly requested by the user: implement geometric consistency
and initialize from the current WAM checkpoint. Distinguish whether a direct
gradient to both coordinate readouts reduces relative hand/object contact error
without degrading held-out tau/object predictions. The cheapest discriminating
setup adds a deterministic loss branch to the existing v6 model, keeps the
expanded local dataset/goals/event inputs and uses the latest complete baseline
checkpoint. No new independently trainable contact decoder or future-hand input.

## Geometry and objective

The previous C head predicts labels from shared features, with no gradient to
the final hand_future/scene_future coordinate readouts. This branch computes
geometry directly from their denormalized outputs:

- Bind each observed778-point hand skin to its nearest observed joint21 bone,
  with a palm-normal-referenced local frame and projected bone coordinate.
  Predicted joints transport that skin differentiably. Attachments read only
  observed input; GT futures enter only the loss as targets.
- Extend the predicted32-point object flow to selected CAD surface queries with
  an observed-only affine pseudoinverse. This reproduces rigid GT flow for
  full-rank CAD samples; it is not a rigidity projection of arbitrary predictions.
- At frames1/4/8/12/16/20/24, select64 highest GT-proximity and64 uniformly spaced
  queries from2048. The original C supervision still covers all24×2048 outputs.
  GT positive queries (>0.25) choose a nearest skin vertex of the GT-labelled
  anatomical part. Compare the predicted object-minus-skin vector to the vector
  obtained from the same proxy under GT trajectories. This calibration makes
  exact GT predictions zero-loss despite approximate skinning; no forcing of
  joint centres onto object surfaces or zero contact-probability collapse.
- Positives use SmoothL1(relative-vector error/10mm), with no far-distance
  cutoff on predicted gaps, so a separated predicted hand receives gradients.
  Proxy GT gaps>50mm or unavailable hand/part geometry are masked and coverage
  is reported. Unknown geometry is never fabricated as a negative.
- GT-negative queries (<0.01) receive only local clearance penalties when the
  predicted nearest skin is too close. Margin=min(GT-proxy gap,33mm)-2mm;
  detached GT direction supplies a separating gradient at exact overlap.
  Distant noncontact hands are not pulled inward. This is not a full signed
  mesh collision, self-contact, force or stability guarantee.

Total loss is original tau/scene/C/part plus
geometry_weight × (geometry_contact +0.25 geometry_clearance). A four-example
TRAIN-only GPU preflight measures gradients at actual hand/object output
readouts. Choose weight=min(0.05,max(0.0001,0.5×base_gradient/geometry_gradient)),
then freeze it and ramp it over100 updates. Require positive proxy coverage>=80%,
finite nonzero direct gradients to both decoded outputs/readouts, and no direct
readout gradient from the original C-only loss. This calibrates scale, not an
efficacy result or an independent validation.

## Initialization, resources and budget

Keep the baseline running during implementation/tests. Then request owned STOP
on local-expanded-ddp-20261021-r1, wait for saved exit, and load its newest actual
final checkpoint. Copy model weights and all normalizer buffers exactly. Fresh
optimizer/RNG/count/warmup/selection for the new objective, explicitly recorded
as weights-only initialization; do not claim strict old-objective continuation.
Initialization step/path is written in the actual run manifest and switch receipt.

Reuse TRAIN207579/DEV3356 windows over1177 sequences; no input/cache mutation.
Goal endpoint and v6 CONTACT/RELEASE event conditioning retained. Three local
GPUs4/5/6 remain authorized by the user's previous multi-GPU instruction; do not
touch unrelated0–3/7 work. Python graspenv/Torch2.4.1+cu121, BF16/TF32,
8 spawn loaders/rank, NCCL_P2P_DISABLE=1. Preserve LR1e-5,50-update LR warmup.

Capacity test8 disposable updates at local96, falling back80/64 only if OOM or
reserved peak>21.5GiB. Then initialize8 real three-rank updates, audit checkpoint,
strict resume2, audit again, continue the same new-objective checkpoint. Three
parameter signatures and all568 optimizer states/RNG are checked. No capacity
updates are consumed by main training.

Fresh bounded run:10000 updates or14400 cumulative new-objective seconds,
whichever first; DEV before any update and every500, initial step20,32 fixed
samples/source, microbatch4. Keep original physical-unit hand/scene/C metrics
and add geometry relative EPE, predicted/GT proxy gaps, negative violations and
positive coverage. Compare physical metrics against this exact initialization;
new total objective is not numerically comparable to the old objective.
Per-stage8GiB checkpoint budget; transition/readiness/training combined~16GiB
peak reserved; stop at20GiB free, always exceeding user's15GiB hard floor.
Latest baseline and previous results remain untouched in their original run.

Outputs: outputs/ptv3-shared-wam-v6/geometry-consistency-20261021-r1/.
Code archive/logs: tmp/geometry-consistency-20261021-r1/.

## Engineering evidence

7 new geometry tests pass: rigid skin/flow equivariance, GT-zero and common-motion
invariance, far-contact gradients to both outputs, noncontact no-pull/overlap
separation, missing-geometry/NaN masking, gradients to actual SharedWAMV6
coordinate readouts, and explicit weights-only initialization/normalizer binding.
The final complete v6 suite passes30 tests in8.62s under graspenv. GPU preflight
and the actual three-rank save/recovery checks also pass, as recorded below.

Method limitations: skinned observed surface is an approximate future mesh,
and arbitrary predicted object flow is extended affinely. Paired-GT calibration
avoids a known proxy target bias but does not prove exact contact or prevent
every penetration. No matched no-geometry continuation control or physical
rollout comparison has yet run; no claim of solving drift or improving grasping.


## Actual launch and startup evidence

Frozen implementation b852f39cb016b2a04332c3ffd5a09037cc3ca7db, supervisor943417,
run geometry-consistency-20261021-r1. The old expanded run stopped normally at
4371; its complete1,186,352,333byte training/step004371.pt is the initialization,
not an earlier best checkpoint or remote23758. CPU audit passes finite model,
568 Adam states at4371,3 RNG payloads, world3/global288 and inherited normalizers.
All old model parameters/buffers are copied; optimizer/RNG/step/selection are
explicitly reset for this new objective. Later8→10 recovery is strict within
this new objective.

Actual current-checkpoint GPU preflight on four TRAIN examples passes. Original
C/part-only gradients to both coordinate readouts are None. New geometry gives
hand_readout16.60445/object_readout20.80343 and decoded hand5.33547/object1.60874
(nonzero finite gradient norms). Eligible positive coverage2897/2897=100%.
Base coordinate-readout norm1.220175 versus raw geometry26.617487 selects frozen
weight0.022920552754813268, yielding full-weight gradient ratio0.5; ramp100 updates.
These four TRAIN examples only calibrate the loss and check connectivity.

Actual batch96 capacity8 updates passes, allocated13.594GiB/reserved15.342GiB.
Keep three3090 GPUs4/5/6, local96/global288. Real readiness8 updates saves a
complete1,186,348,749byte checkpoint:568 optimizer states at8,3 RNG payloads,
identical three-rank parameter signatures and exact inherited normalizers.
Strict recovery8→10 passes the same audit (1,186,351,629bytes, all states10).
Sustained training resumes10 and has reached100 with finite loss0.75918,
grad_norm0.86805, nonzero geometry_contact0.55237/geometry_clearance0.07497 and
full weight0.02292055; allocated peak14.125GiB. Instantaneous selected GPU
utilization100/95/99%, physical memory21,013/20,299/19,001MiB; free197.71GiB.
No OOM, NaN, loader failure or unrelated process termination observed.

The fixed32-GRAB-example DEV panel at the exact4371 initialization has hand
EPE52.67626mm, geometry relative22.53791mm, predicted proxy gap23.53811mm,
GT-proxy gap4.22592mm, noncontact violation2.06307mm, positive coverage100%.
This is the correct before point for later geometry-run comparisons; the
historical remote23758/local-original49.9303mm panel starts from different weights.
DEV at20 is operational startup evidence, not a meaningful efficacy verdict.

Full trace: process.json, gpu-preflight/report.json, config-b96.json,
readiness/dev-step000000.json, per-stage result/checkpoint audits, training/
progress.json and train.jsonl under the declared output. Source/runtime metadata
are in each run-manifest; code archive and supervisor log are project-local tmp.
Root STOP requests a saved stop. Enforced budget10000 updates or cumulative4h,
20GiB free-space floor, bounded checkpoint retention; source checkpoint retained.
No free-feedback rollout or physical grasping evaluation yet.

Startup acceptance: startup-acceptance.json records the exact40→100 update
interval (240.56 samples/s), six recent GPU samples, full gradients/audits and
exact initialization DEV. Compute-only remaining ETA3.29h excludes periodic
DEV/save overhead; cumulative4h bound remains authoritative.
