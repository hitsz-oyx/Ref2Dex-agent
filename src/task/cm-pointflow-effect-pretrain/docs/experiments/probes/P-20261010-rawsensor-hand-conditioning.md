---
schema: ref2dex.probe.v2
probe_id: P-20261010-rawsensor-hand-conditioning
experiment_id: P-20261010-rawsensor-hand-conditioning
date: 2026-10-10
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: pending
claim_id: C1
hypothesis_family: HF-rawsensor-hand-conditioning
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: test raw sensor prediction from human hand motion as an optional representation teacher
decision_changed_if_negative: defer sensor learning until longer diverse source labels and hand relations can be qualified
status: UNCLEAR
run_id: egotouch-diverse-raw-20261010-r1
---

# Raw sensor supervision and human hand conditioning readiness

Class Decision. Serves intermediate interaction representation pretraining for
action-conditioned Cm; does not replace the Mission's object dynamics or matched
self-trained robot policy objective. This card initially acquires/audits labels;
no learning signal, pressure calibration or sensor-usefulness claim follows.

## Decision note / fixed acquisition protocol

Two cheap1.4–1.5s records are inadequate: fitting USB has only65 changing raw
channels; held mouse137, persistence beats history OLS. Acquire longer diverse
labels before deciding whether a small hand-conditioned sensor teacher is useful.

Choose exactly2 distinct tasks per scenario Home/Office/Outdoor/Retail/Workbench,
all official TRAIN entries. Stable cost ordering: pressure JSON400000..2000000B,
complete four-label bundle<=5000000B. First eligible task per scenario is local
fit, second held-task development; no recording/task shares fit and held partitions.
These are neither official test nor participant-separated generalization scores.
Do not inspect future sensor values or contact flags for selection. Duration and
numeric qualification follow acquisition; file size is only a bounded heuristic.

Download only original `jq_pressure.json`, `wilor_hands.json`, `pressure_grids.npz`,
`manual_contact_annotation.json`, fixed revision
`cfdbb0ac31cc2af4247943820aa250575e7e6637`. Check total inventory size<=16MiB before
network, verify each original Git/LFS checksum and byte count. Prior same-release
ModelScope search was empty; try HF domestic mirror then project proxy, retaining
route failures. Cap all transfer16MiB and wall240s. Keep partial verified files and
failure manifest. No RGB, model weights, HDF5 expansion or external modifications.

Subsequent CPU qualification will retain original frames/timestamps, fixed raw/255
scale, missing validity and unknown channel hardware roles. Need >=128 contiguous
frames and finite512 byte-valued channels, matching21-joint dual-hand row identities,
and enough support in>=3 fit and>=3 held tasks before proposing neural learning.
WiLoR world units/calibration remain unqualified; use explicit relative shape if
later conditioning is tested. Contact false is not physical negative evidence.

Outputs: `outputs/cm-pointflow-effect-pretrain/egotouch-diverse-raw-20261010-r1/`.
Tool: `tools/audit/sample_egotouch_diverse_labels.py`. New artifacts<=32MiB;
CPU acquisition/arrays only, no GPU or model training. No new branch/push.

## Results

Pending fixed-commit acquisition and semantic qualification.

## Limitations / future evidence

Only10 recordings, selection by cost/metadata and pseudo hand estimates. Hardware
sensor channel/contact roles and calibration remain UNKNOWN. Fit/held examples are
separate tasks but not certified separate people/cameras. Raw sensor prediction is
not paired object flow or robot-force supervision. Any later future-hand conditioning
is an observed human-trajectory teacher/oracle, not online robotic access or causality.

Acquisition completed at `25d9986` in41.93s,14387483B,40 verified files. Original
pressure/WiLoR row IDs match all10 recordings, each224..751 frames. Many WiLoR
hands are missing/empty; preserve them as invalid rather than zero-valued points.

Before qualification: retain all original rows and hand shape/finite validity,
but any later matched retrospective future-hand comparison uses only complete
H4+K24 dual-hand windows, identically for all arms. This conditions evidence on
visible pseudo-action support; it is not a deployable history-only availability
policy. Future sensor values do not choose windows. Require>=3 supported tasks
in each split. Hand coordinates are wrist-relative per-frame shape with a scale
from median wrist–middle-MCP distance of the4 HISTORY frames only, never future
maxima/spans. This discards global wrist transport and does not invent a metric
world/camera frame. Predict raw sensor bytes/fixed255, not calibrated contact.

Prepare immutable packs under `processed/`; original empty hand rows remain NaN
and declared invalid. FIT dynamic-channel statistics may use FIT data only and
are frozen for held tasks. All original raw timestamps/frame IDs are retained.
CPU<=30s, <=16MiB new packs, no network/model/GPU. Future human hand inputs would
be an offline observed-trajectory teacher, not causal controllable robot actions.

Qualification at `02300b7` passed:4 fitting tasks330 windows,5 held tasks535
windows. Plush-toy fitting recording has no complete28-frame dual-hand windows;
it is preserved but not sampled. This is an explicit visibility-conditioned
pseudo-action subset. Row counts, NaN masks and source IDs remain preserved.

## Matched learning screen (fixed before execution)

Class Decision: does human future joint shape add raw sensor prediction information
beyond observed sensor/hand history? If positive, test an auxiliary point-feature
teacher; if negative, do not expand this sensor learner or claim tactile useless.
This is not the native point-dynamics main model or a robot-policy comparison.

Three identical128-hidden-layer MLP residual heads from identical initial weights:
HISTORY sensor4x512 + root-relative hands4x42x3, with future hand shape24x42x3
as actual/zero/shuffled input. Shuffle uses next sorted different task and modulo
window index, fixed without sensor values in fit/held separately. Original actual
future hand is a retrospective human observation/oracle, not a proposed robot
intervention. All arms see identical complete windows, task-uniform draws and
raw target labels; all have identical architecture/capacity.

Predict24x512 sensor delta from current raw/255; zero final layer initializes
persistence exactly. TRAIN/FIT input mean/std only, std floor.01, no record future
maxima. Loss fixed Huber scale.02 over frozen FIT-dynamic channels (range>=2 raw
counts). Same seed227,500 updates/arm, batch4 windows per fitting task16 total,
AdamW lr3e-4/WD.01, grad clip1, BF16, fixed final checkpoint. No dev selection.

Report each task all-dynamic and candidate normalization groupsA/B (raw mapping
indices excluding/including left208..222/right33..47, respectively). These are
processing candidates, not verified hardware pressure/bend labels. Primary gate:
held-task macro groupA endpoint MAE with actual future hand improves>=10% over
best of history/shuffle/persistence; no held task worse than its best control by
>20%. Compare groups separately to expose a trivial bend-channel-only gain.
Positive remains a teacher-potential Probe, not physical tactile/robot usefulness.

One freshly inspected free GPU, max300s whole three-arm loop, artifacts<=300MiB.
Stop on nonfinite, source/hash drift, conflict, cap/deadline; preserve failed runs.
GPU utilization/memory and ETA logged after startup and every10s. No external
process/system changes, dataset expansion or policy training. Output `matched-r1/`.
