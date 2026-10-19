---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v6-event-continuation
experiment_id: P-20261019-wam-v6-event-continuation
date: 2026-10-19
task: ptv3-shared-wam-v6
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-v6-event-training-readiness
probe_index_in_family: 1
seed_pool: probe
seeds: [170,171]
decision_changed_if_positive: retain v6 as a ready candidate for a separately chosen event-conditioned long run
decision_changed_if_negative: fix the isolated v6 migration or training path before any event-conditioned scale-up
status: PLANNED
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

Pending local gate and real four-GPU execution.
