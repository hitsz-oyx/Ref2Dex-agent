---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v6-event-continuation
experiment_id: P-20261019-wam-v6-event-continuation
date: 2026-10-19
task: ptv3-shared-wam-v6
branch: main
git_commit: d37e2320
claim_id: C1
hypothesis_family: HF-v6-event-training-readiness
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain v6 as a ready candidate for a separately chosen event-conditioned long run
decision_changed_if_negative: fix the isolated v6 migration or training path before any event-conditioned scale-up
status: COMPLETED
run_id: event-continuation-readiness-20261019-r1
---

# Four-GPU v6 event-goal continuation readiness

Classification: Blocker engineering Probe. The user asks to isolate the new
goal/event model in v6 and test whether it can sustain training, then explicitly
chooses to restore v5 afterward. Distinguish a valid migrated trainable event
path from startup-only success, unused condition parameters, worker/memory
instability or broken full-state recovery. The cheapest requested sustained
check is8 recovery +1000 full-batch updates +8 recovery, with existing scheduled
DEV/checkpoint paths and parameter signatures. This is not a scientific
conditioned-versus-unconditioned training efficacy experiment.

## Frozen protocol and budget

v6 is its own package/model/config/recipe/checkpoint identity. It copies the
v5 model implementation and event encoder while reusing shared geometry,
normalization, loss/metrics and immutable data primitives in the same archive.
v5 production files/configs remain unchanged. Preserve original H4/K24/joint21,
surface778/object2048/C2048,voxel5mm,768 width/12heads/12layers and TRAIN-only
stats. One CONTACT/RELEASE/NONE +relative30Hz time+valid condition;50,304 new
parameters, zero output residual, no new tokens/KNN. TRAIN dropout15% has a
separate deterministic seed/step/rank stream; base sampling remains exact.

Use all39526 sequences and inherited TRAIN3218740/DEV545459 windows with the
complete v2 event sidecar. Admit an explicit new v6 recipe; preserve candidate
cache manifests unchanged. Annotations represent nearest-surface proximity,
not force/stable grasp/intent certification; HOI4D midpoint annotation time
queries do not fill physical C/state labels. Keep unknown and known NONE
distinct and far events typed. Deployment conditions must be task plans.

Pause only the identified owned ac951370 v5 run through its controlled STOP;
retain its final complete checkpoint. Check effective batch/world/config/data
before explicit v5→v6 expansion. Keep old weights/moments/steps/RNG/elapsed and
all config values except software schema and declared event fields. Only new
event parameters start fresh. Ordinary resume stays strict. Reset selection
for the new input contract; protect objective-best and trajectory-best.

Four remote RTX4090 GPUs,local128/global512,bf16/TF32,8 spawn workers/rank,
timeout120s,LR1e-4 with original warmup history and80k/cumulative7day ceiling.
Remote graspenv Python3.8.20/Torch2.0.1+cu118. Reuse model/draw seed170 and
DEV171 from the parent; this is continuation, not independent replication.
New branch initialization seed170; no independent validation/TEST. DEV has8
sources (DexYCB has no DEV). In addition to conditional DEV, evaluate the same
samples with event_valid=false; no benefit claim from this diagnostic alone.

Remote root `/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/`.
Recipe `data/event-recipe-20261019-r1/`, controller output
`outputs/ptv3-shared-wam-v6/event-continuation-readiness-20261019-r1/`.
Each phase has a fresh directory; no checkpoint overwrite,20GiB per-phase
artifact ceiling,free disk>=20GiB. Intended total probe <=3600s; recovery phases
have900s launcher limits, sustained phase2400s. CPU checkpoint conversion300s.
Check GPUs released before each next run. Owned controller restores v5 in
finally; v6 updates never alter its baseline. The restore itself must produce
real updates on the original frozen ac951370 code/config/data/optimizer.

## Acceptance

Require all planned updates, finite losses/gradients, nonzero new-branch
gradient and changed output weights, actual full DEV and checkpoints, equal
four-rank parameter signatures, and strict v6 recovery from a published v6
checkpoint. Inspect real GPU utilization/memory/throughput and compare to the
v5 pre-pause records. Preserve all trial checkpoints and logs for rollback.
Finally verify v5 RUNNING beyond its original stop step. A failed v6 test does
not authorize replacing the baseline. Four-rank bitwise uninterrupted replay
and80k stability are not certified by this bounded check.

## Results

Local v5+v6 suite:48 tests passed in19.72s, including actual model forward
equality after expansion, unchanged inputs/sampling, spawn data loading,
strict recipe/batch rejection and uninterrupted-versus-resumed v6 entrypoint
equality. Scoped static/seed/link checks pass. Full repository gate remains
blocked by six existing trajectory-policy cards missing v2 metadata.

Frozen d37e2320 archived both v5 shared primitives and v6 under remote
`code-d37e2320/`. The independent v6 recipe is frozen/admitted with original
TRAIN3218740/DEV545459 and original stats. Controller and final checkpoint audit completed on2026-10-20. Actual phases:

| Phase | Updates | Final step | Loop seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Initial recovery | 8 | 15128 | 103.227 | 17.928 |
| Sustained | 1000 | 16128 | 1052.253 | 18.814 |
| Resume recovery | 8 | 16136 | 101.596 | 18.101 |

All phases covered nine TRAIN sources, wrote complete checkpoints and had equal
four-rank parameter signatures. Event branch gradients were nonzero and output
weights changed. Audit368feea9 verified564 original optimizer states at16136,
4 new states at1016,4 RNG payloads, strict effective batch128/global512/world4
binding, retained migration provenance and exact original horizon-normalizer
buffers. This is1016 new updates, not16136 event-training updates.

Sustained forward+loss258.096ms/backward596.526ms/optimizer29.566ms; steady
throughput about1.115 updates/s versus original v5 about1.118. Physical GPU
memory about25–30GiB and utilization predominantly90–100%. These are operational
observations with different overheads, not a matched performance experiment.

| Fixed DEV step | Objective | Hand EPE mm | Trajectory score mm | Unknown-event hand EPE mm |
| --- | ---: | ---: | ---: | ---: |
| 15129 | 1.290095 | 33.637805 | 17.250517 | 33.633622 |
| 16000 | 1.283868 | 33.644315 | 17.164916 | 33.738737 |
| 16129 | 1.286467 | 33.167013 | 17.007955 | 33.285051 |

Trajectory score is the predefined mean of short-horizon, root-relative and
contact-phase macro errors; it is not overall hand EPE. Latest conditional
versus unknown hand difference is−0.118mm. No matched unconditioned training arm
was run: continuation improvement cannot be attributed to event conditioning.
Full80k stability and robot success are not established.

At16128 rank0 local last_loss was26.098196; the last regular cross-rank log
at16120 was0.799613, resumed16129 was0.745433, final local16136 was0.752968.
Do not equate a local result field with a global logged loss. Exact rank0 draw
inspection found source4 sample[29384,0,35] with maximum future object
 displacement8.92525m and normalized scene-target mean-square2974.711; batch
mean24.011. This strongly suggests a target-outlier explanation but is not an
exact per-term model-forward attribution. The outlier is in inherited targets,
not introduced by the event overlay. Its upstream sequence should be traced
before a separate data-admission decision; no data was filtered during this run.

User-requested v5 restoration completed using original frozen ac951370 and its
own step015120.pt, optimizer/RNG/config/data/budget unchanged. Active run is
pretrain-nine-v5-width768-after-v6-probe-20261019-r3; it was verified RUNNING
beyond stop at15260, rate1.116 updates/s, global512, four GPUs100% utilized.
v6 is stopped; all trial checkpoints and selection candidates are retained.
No v6 updates were consumed by baseline v5.

Remote controller report: outputs/ptv3-shared-wam-v6/event-continuation-readiness-20261019-r1/report.json.
Remote checkpoint audit: outputs/ptv3-shared-wam-v6/event-continuation-audit-20261019-r1/summary.json.
Final v6 checkpoint: resume-recovery/step016136.pt under the controller directory.
Objective-best: sustained/step016000.pt; trajectory-best: resume-recovery/step016129.pt.
Small local report copies and target-outlier receipt:
outputs/ptv3-shared-wam-v6/event-continuation-readiness-20261019-r1/remote-audit/.
Decision: retain ready v6 checkpoints; keep v5 running pending a separate long-run choice.
