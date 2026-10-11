---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-three-domain-smoke
experiment_id: P-20261011-goal-point-wam-three-domain-smoke
date: 2026-10-11
task: goal-point-wam
branch: cm-pointflow-effect-pretrain
git_commit: 2a3a985f
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 1
seed_pool: probe
seeds: [247]
decision_changed_if_positive: start the bounded three-domain no-video Probe
decision_changed_if_negative: repair the scene/goal/world contract before training
status: COMPLETED
run_id: engineering-smoke-20261011-r2
---

# Goal-Point-WAM three-domain engineering smoke

This is an engineering smoke, not a scientific validation. It checks that the
approved ref1-inspired scene/action/world wiring can run on the existing
OakInk2/GRAB/ARCTIC main-three manifest without video inputs.

## Frozen question and boundary

The smoke distinguishes implementation readiness from a data or architecture
claim: can the scene-only PTv3-small encoder, endpoint-goal trajectory branch,
and executed-action world branch produce finite outputs and gradients under the
existing 30 Hz, history-4, horizon-24, dual-hand-11-point contract? It does not
measure held-out generalization, robot policy utility, or official PointWAM
reproduction.

## Run

- Output: `outputs/goal-point-wam/engineering-smoke-20261011-r2/`
- Steps: 4, one GPU, batch 1, fresh AdamW
- Data: frozen main-three manifest and train-only shared normalization
- Video/RGB: disabled
- GPU: local logical GPU 2 (`CUDA_VISIBLE_DEVICES=2`)

## Result

- `result.json` reports `COMPLETED` at step 4.
- Forward, finite loss, backward, clipping and checkpoint round-trip passed.
- Peak allocated memory was approximately 0.66 GB for the batch-1 smoke and
  reserved memory approximately 1.05 GB; this is not a full Probe estimate.
- Recorded utilization ranged from 1% to 85% during the short run.
- Goal replacement changed world translation/rotation by at most
  `5.82e-11`, satisfying the goal-blind world audit.
- The fixed validation panel was finite; the values are diagnostic only and
  are not a generalization claim.

The first attempt (`engineering-smoke-20261011-r1`) exposed a bfloat16 loss
backward incompatibility and was retained as a failed implementation record.
The trajectory head now casts its physical-loss input to float32; r2 passed.
