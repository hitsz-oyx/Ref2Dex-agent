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
