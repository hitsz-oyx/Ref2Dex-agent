---
schema: ref2dex.probe.v2
probe_id: P-20261011-official-reference-screen
experiment_id: P-20261011-official-reference-screen
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: a9b0fa64
claim_id: C3
hypothesis_family: HF-official-reference-coverage
probe_index_in_family: 1
seed_pool: probe
seeds: [191]
decision_changed_if_positive: use competent references as candidate multi-task teacher sources before WAM
decision_changed_if_negative: restrict teacher collection to competent references and diagnose failed tasks before imitation training
status: RUNNING
run_id: official-reference-screen-20261011-r1
---

# Current corrected12 references: official actor physical screening

User requests actual official DExplore performance on repository reference tasks,
temporarily deferring WAM/PW. Decision: which references provide competent teacher
behavior? Cheapest useful evidence: one seed191 wave per corrected reference,
8 synchronous frame0 environments, deterministic official Inspire checkpoint,
native GPU PhysX, full frames-1 controls, no early termination or interventions.
No training. These evaluation traces are training_allowed=false.

Inputs: corrected-expert-inputs-20261007-r2 manifest (12 motions/10 objects),
native-scale corrected assets, ballSize=1, existing inspire environment configuration.
Official checkpoint SHA8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553;
archived Torch2.0.1/rl_games1.1.4 with single RMS compatibility and padded64-row
actor calls. One separate process per reference to avoid carrying simulator state.

Record actual object/hand pose, velocity, q/dq, requested/applied actions and native
PD targets, unsigned sampled surface proximity, table support and footprint.
Report geometric lift>=3cm with hand proximity<=1cm and no table support, held5,
held45, maximum consecutive held frames, controlled placing+terminal settle15,
object reference position RMSE/rotation mean and terminal error. Placing is reported
separately: a clip whose reference ends elevated is not required to place. Determine
reference terminal support from the same native-scale mesh/table geometry, record it.
Native survival success_rate is never manipulation success. Independent CPU summary
recomputes metrics and controller mapping from the retained traces; probe counts
are correlated repeats of fixed starts, not independent generalization estimates.

Task screen PROMISING if >=6/8 hold45, else UNPROMISING; full task behavior requires
reference-compatible terminal completion and acceptable tracking, reported separately.
Aggregate UNCLEAR when task coverage is mixed. No formal scientific conclusion.
No checkpoint or seed selection based on results; retain every failure.

Budget20 GPUmin/1GiB new output; one inspected idle GPU3. Bound each native process
300s and total worker elapsed<=1200s. Recheck GPU before each launch; existing foreign
PID2378894 is an inspected Isaac Sim job active on other GPUs with130MiB idle contexts
here. Allow only that existing context with GPU utilization<=5% and prior total<=512MiB;
stop on new PID/context growth>512MiB, nonfinite, unexpected reset, source drift or
budget. Do not operate foreign processes. No Campaign/ref modifications or new branch.

## Execution compatibility note

First cubesmall attempt stopped before any controls: native default hard-object
oversampling repeats the same input path because its name contains `small`.
The one-motion guard treated this alias as a second reference. Preserve FAILED
manifest/log; not a physical negative result. Permit loader aliases only after
every resolved path, frame count and processed reference tensor matches the
single authenticated input. Retry cubesmall in a fresh directory; no changes to
native simulation, sampling defaults, checkpoint, seed or evaluation thresholds.
