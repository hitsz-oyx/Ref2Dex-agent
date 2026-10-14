---
schema: ref2dex.probe.v2
probe_id: P-20261015-rl660-reference-q-replay
experiment_id: P-20261015-rl660-reference-q-replay
date: 2026-10-15
task: ptv3-shared-wam
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-rl660-reference-q-replay
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: qualify DExplore reference-q as a nominal target candidate and design a bounded q_ref retargeter comparison
decision_changed_if_negative: keep the current PD baseline, separate low-level tracking compensation from q_ref intent, and do not relabel the WAM supervision
status: UNPROMISING
run_id: rl660-reference-q-replay-20261015-r1
---

# RL660 DExplore reference-q nominal replay

Result: COMPLETED (UNPROMISING for direct execution). This Probe tests the missing third target explicitly: the raw geometric DExplore `q_ref` plan, rather than the RL export's realized `q_real` or the actor's `PD target`.

Decision Note: `ref2_3` identifies three distinct quantities. The current R learns the actor's feedback-compensated PD target; the RL export stores realized simulation q in the same raw columns that the geometric source uses for `q_ref`. The cheapest discriminating experiment is a matched native/PD/reference-q physics replay on the three already fixed dev sequences. This is an execution feasibility Probe, not a relabeling decision or a claim about DExplore reward semantics.

## Hypothesis and protocol

The primary hypothesis is that `q_ref[t+1:t+9]`, after the existing Inspire coupling and live-q inverse command conversion, is a usable nominal target. A positive result requires the q_ref arm to preserve native calibration and reach the predeclared hand/object/hold gates. A failure separates direct target feasibility from the question of whether a learned retargeter can predict q_ref.

The fixed panel is `s10_airplane_pass_1`, `s10_mug_pass_1`, and `s4_spheresmall_pass_1`, with seed 148, 30 Hz, wrist Kp/Kd 200/20, finger Kp/Kd 100/10, existing coupling and clamp, and K4 target refresh. Six frame-zero rows are used: recorded native, recorded PD, reference-q full replay, and reference-q replacements at the first, middle, and last admitted window. The q_ref rows read raw geometric `interaction_hand_inspire.pt[:,373:391]`; they do not read the RL-export trajectory q field. Each target chunk is `q_ref[query+1:query+9]`, coupled before conversion.

The runner records q/dq, hand points, object pose/velocity, contact/support signals, requested and actual PD, clipping, query clock, and fork state. It must pass frame-zero reset error <=1e-4 and independent command/PD reconstruction <=5e-6. Existing native and recorded-PD rows remain calibration references. The original `hand mean <=10 mm`, `hand p95 <=20 mm`, object mean <=10 mm, and held-45 gates are Probe gates; a short window cannot establish a full task success rate.

## Data contract

The frozen RL660 corpus is `outputs/ptv3-shared-wam/rl660-corpus-20261014-r2`. Its `source_membership` points to the read-only RL export, whose manifest states that raw columns 373:391 were overwritten by simulated q. The geometric source is `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore`; it is read-only and supplies q_ref. The reference motion is also used by the native environment to initialize the reference row, while the replay reset is checked against the recorded RL state. Raw q_ref is not presumed coupled or branch-normalized; the runner applies the existing controller coupling and reports clipping/rotation behavior.

## Resource and stop conditions

Use only an idle project GPU (GPU4/6; GPU5/7 remain reserved for the existing geometry-lift run). Each run is bounded to 300 seconds and the panel is at most three scientific runs plus required smoke/audit. Stop on nonfinite values, reset/clock mismatch, foreign GPU PID, or storage below 20 GiB. Native calibration is reported as a classification gate, not a stop condition, because these short replays have known native drift. Do not alter GPU5/7 geometry-lift training or the remote WAM process.

## Result and decision

All three full raw-q_ref replays completed and passed the independent command,
clock, reset, and PD reconstruction audit. The q_ref arm itself was
UNPROMISING for direct execution: airplane hand/object mean 128.90/174.11 mm
with 66.86% clipping, mug 370.55/158.64 mm with 98.97% clipping, and sphere
334.44/200.38 mm with 98.17% clipping; all had zero held frames. The audit's
overall label remains `UNCLEAR` where the native 5 mm calibration gate failed,
so these numbers are not a formal scientific refutation.

The failures are partly a wrist-Euler branch/control issue (mug and sphere
wrap at the first step), so a nearest-live branch follow-up is recorded in
`P-20261015-rl660-reference-q-branch-replay`. Until that follow-up is
resolved, do not relabel the RL corpus or train a q_ref-supervised R. The
current executable supervision remains the feedback-compensated PD target;
q_real is retained as measured state/diagnostic, and q_ref remains a separate
nominal-intent field.

## Limitations / future evidence

This is a single-seed, three-sequence oracle replay. DExplore's actor reward reads reference body/object/contact features; it does not directly optimize the reference q slice, so q_ref replay is an alternate nominal target feasibility test rather than a reproduction of the teacher actor. q_ref Euler branches can be unwrapped and the raw distal joints are coupled by the native controller. A positive replay does not prove a learned q_ref retargeter or closed-loop grasping; a negative replay does not refute q_ref as an intention representation if low-level tracking compensation is the limiting factor.
