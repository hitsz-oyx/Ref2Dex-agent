---
schema: ref2dex.probe.v2
probe_id: P-20261015-rl660-reference-q-branch-replay
experiment_id: P-20261015-rl660-reference-q-branch-replay
date: 2026-10-15
task: ptv3-shared-wam
branch: main
git_commit: 75edfd80
claim_id: C1
hypothesis_family: HF-rl660-reference-q-replay
probe_index_in_family: 2
seed_pool: probe
seeds: [148]
decision_changed_if_positive: treat q_ref as a feasible geometric intent only after live-branch lifting, and keep it as a candidate input for a v2 adapter
decision_changed_if_negative: retain PD targets as the executable supervision and defer q_ref to a learned/explicit retargeter rather than raw execution
status: UNPROMISING
run_id: rl660-reference-q-branch-replay-20261015-r1
---

# RL660 branch-aligned DExplore reference-q replay

Result: COMPLETED (UNPROMISING for direct execution). The preceding raw q_ref replay is strongly negative at the
controller interface, but its mug and sphere failures begin when continuous
wrist Euler coordinates wrap to a different equivalent branch. This Probe
keeps the exact geometric q_ref pose while selecting the nearest equivalent
XYZ branch to the current live q at every control tick.

## Decision Note

The current evidence distinguishes a representation/control branch failure
from a semantic failure of q_ref. I will run the smallest matched follow-up
before deciding whether q_ref should be excluded from the v2 adapter. A
positive result only qualifies q_ref as a nominal geometric intent; it does
not make q_ref the DExplore teacher action or replace recorded PD targets.

## Protocol and contract

Use the same frozen RL660 panel, seed 148, controller, coupling, and four-step
control refresh as `P-20261015-rl660-reference-q-replay`. Read geometric
`interaction_hand_inspire.pt[:, 373:391]`, convert wrist Euler XYZ with the
nearest-live branch at every target consumption, then apply native finger
coupling and the existing PD inverse. No velocity feed-forward or oracle
finger substitution is permitted. The independent audit must reconstruct the
same branch operation from the manifest and live-q trace.

Engineering gates are reset/clock/command/PD reconstruction, finite tensors,
and no foreign GPU process. Native calibration is reported as a classification
gate, not a stop condition, because it is a known property of these short
replays. Probe success additionally requires the predeclared hand/object/hold
gates on the q_ref arm; otherwise the result remains UNPROMISING or UNCLEAR.

## Limitations / future evidence

This remains a single-seed oracle physics replay. Even a positive result does
not show that the DExplore actor optimizes q_ref, nor that a learned v2 model
can predict q_ref. The v2 fine-tune must retain separate fields for q_ref,
feedback-compensated PD target, and measured q_real.

## Result

The three full branch-aligned replays passed the independent reconstruction
audit, but did not make q_ref executable. Airplane hand/object mean was
128.90/174.11 mm with 66.86% clipping, mug 114.56/169.26 mm with 79.49%
clipping, and sphere 86.15/200.49 mm with 43.90% clipping; all had zero held
frames. Nearest-live lifting reduces some wrist wrapping but leaves large
pose/object errors, so the direct q_ref target remains UNPROMISING. Because
the native rows fail the short-replay calibration gate on two sequences, the
independent audit labels remain `UNCLEAR`; this is not a formal refutation of
q_ref as a geometric intent.

Decision: keep the PD-target route for the WAM fine-tune. Add explicit
`q_ref`, `q_pd`, and `q_real` fields to the v2 robot-data contract for audit and
future retargeter work, but do not use raw or branch-lifted q_ref as the action
loss target in this run.
