---
schema: ref2dex.probe.v2
probe_id: P-20261015-ptv3-shared-v2-robot-finetune
experiment_id: P-20261015-ptv3-shared-v2-robot-finetune
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 75edfd80
claim_id: C1
hypothesis_family: HF-v2-remote-robot-adapter
probe_index_in_family: 1
seed_pool: probe
seeds: [151]
decision_changed_if_positive: continue v2 remote-WAM fine-tuning with the explicit PD-target action adapter
decision_changed_if_negative: stop this adapter and diagnose data/model/wiring before spending a long run
status: PLANNED
run_id: ptv3-shared-v2-robot-finetune-20261015-r1
---

# V2 remote-WAM robot fine-tune

This Probe adds a v2 Chunk-R action adapter on top of the remote v2 shared WAM
checkpoint and trains on the frozen DExplore RL660 corpus. The action loss uses
the feedback-compensated `q_pd` target encoded as
`wrist-geometry-continuous-lift.v1`. Measured `q_real` remains the current-state
input/diagnostic. Geometric `q_ref` is loaded from the read-only source into the
sample contract for provenance and future audits, but is not silently used as
the teacher action.

## Decision Note

The reference-q replay and branch-aligned follow-up both failed direct
execution gates while preserving the distinction between q_ref, q_pd, and
q_real. The remote v2 checkpoint has only the pretraining forecaster, so the
minimal compatible route is a new v2 adapter under this Task: load the remote
v2 model strictly, attach a fresh 15-D continuous-lift Chunk-R head, and keep
the three q fields explicit. A v1 weight reshape is prohibited.

## Protocol

Before scientific training, run the four-step engineering smoke on GPU 4/6 and
require finite loss, nonzero gradients through v2 shared/geometry and R, exact
checkpoint load, and no foreign GPU process. Then run the bounded two-GPU Probe
on GPUs 4 and 6 with seed 151, frozen RL660 split, batch 32 per GPU, 12,000
updates, validation every 500 updates, and an 8 GB owned-artifact budget. The
parent must be a non-engineering v2 pretraining checkpoint with matching model
schema, config architecture, and horizon statistics.

## Data contract and interpretation

Each robot window carries `q_ref_next` from geometric
`interaction_hand_inspire.pt[:,373:391]`, `q_pd` from RL660 `pd_targets`,
`q_real` at command time, and `q_real_next` after command execution. Only the
existing physical action target encoded from q_pd contributes to the action
loss. A successful smoke proves wiring only; a short Probe can be PROMISING,
UNPROMISING, or UNCLEAR and cannot establish a grasping claim.

## Limitations / stop conditions

This is offline supervised fine-tuning; it does not execute a learned policy
in Isaac or on hardware. Stop on nonfinite values, checkpoint/schema mismatch,
foreign GPU PIDs, no progress for 30 minutes, less than 20 GiB free disk, or
an adapter audit failure. If the remote SSH tunnel is unavailable, do not
substitute the local v2 checkpoint while claiming a remote-WAM result; use it
only for engineering smoke and record the parent provenance.

## Engineering smoke status

The v2 adapter passed single-GPU and two-GPU four-step smokes using the local
scientific v2 `step1000.pt` only. The latest smoke records nonzero gradients in
`joint_geometry`, `shared_and_heads`, and `retargeter`, finite loss, explicit q
contract, and immutable checkpoint save. These runs are stored under
`outputs/ptv3-shared-wam-v2/robot-adapter-smoke-*` and are engineering-only.
The 660-sequence q-reference preflight also passed (all files finite, matching
frame counts and raw schema). The scientific Probe remains PLANNED until the
live remote 80k checkpoint can be reached and copied with its
schema/provenance intact.
