---
schema: ref2dex.probe.v2
probe_id: P-20261008-contactpose-transport-auxiliary
experiment_id: P-20261008-contactpose-transport-auxiliary
date: 2026-10-08
task: cm-pointflow-effect-pretrain
branch: consequence-evaluator
git_commit: 163160c
claim_id: C3
hypothesis_family: HF-contactpose-transport-auxiliary
probe_index_in_family: 1
seed_pool: probe
seeds: [227]
decision_changed_if_positive: test a larger matched main plus history-only transport auxiliary recipe
decision_changed_if_negative: keep ContactPose out of main dynamics and stop this auxiliary recipe
status: UNCLEAR
run_id: contactpose-transport-auxiliary-20261008-r1
---

# Does a separate rigid-transport auxiliary help measured dynamic prediction?

Result: Main-three completed; native two-update auxiliary engineering r2 passes; the separate300-update matched Probe is next.
Decision: Run one bounded matched Probe with ContactPose future hands excluded from auxiliary inputs.

## Motivation and fixed comparison

User ref3 explicitly asks for a separate ContactPose auxiliary ablation.
This serves the Task's physical pretraining subgoal: can its rigid transport
prior help held-out real dynamic trajectories without learning the shortcut
from object-generated future hands? The completed three-source main run does
not answer that question. This is a Decision Probe, not robot policy evidence.

Initialize both arms from the same latest main-three checkpoint at step 50000.
Import model weights only; reset AdamW, RNG and sampler to the shared seed.
This parent previously learned ContactPose in four-source training, so the
comparison measures incremental auxiliary utility, not a ContactPose-free
historical ablation. Do not compare the new arms with old four-source panels.

Both arms consume the same 300 batches of 32 OakInk2/GRAB/ARCTIC examples,
using original main source proportions and frozen train-only physical loss
scales. Both also compute the same eight-example ContactPose history-only
forward/backward each update; auxiliary loss coefficients are 0 and 0.05.
The optimizer sees one combined update, fixed lr1e-5/wd0.01/clip1. Main loss
remains action-conditioned; ContactPose never enters its sampler or target.

The auxiliary forward whitelist contains only current/history scene geometry
and object descriptors. Future hand chunks, action validity, future effects,
quality and labels are absent from the forward dictionary. Its targets are
native rigid object effects, passed separately to the physical loss. Fixed
articulation remains an annotation assumption, not measured finger motion.
ContactPose auxiliary draws use actual moving windows and participant-held-out
val; no test split is accessed.

## Evaluation and screen

Use the parent's exact192-window main validation panel and a fixed64-window
ContactPose moving-transport validation panel. Evaluate with identical RNG,
restore training RNG afterwards, and verify identical initial model hashes
and initial validation. No epoch/checkpoint selection from test.

Predeclared PROMISING requires at least3% lower main equal-source moving-anchor
h24 EPE than the matched control, no >5% OakInk degradation, and at least10%
lower transport validation EPE. >5% main macro degradation is UNPROMISING;
remaining outcomes are UNCLEAR. One seed and this short budget cannot refute
ContactPose auxiliary learning in general.

## Resource and stops

Use one currently empty GPU0, <=1800s for the two arms including validation
and input freezing, <=2GiB new output, per-main-batch32/transport-batch8 BF16.
Global4GPU/300GB and20GiBfree limits remain. This is a new short auxiliary
Probe, not an extension of the completed50000-update run or its10:00 deadline.
Original checkpoints are read-only. Stop this owned run on foreign GPU use,
input/implementation drift, deadline, nonfinite loss/gradient or disk budget.
Freeze hashes of selected sequences/canonical geometry and all implementations;
use file identity/mtime guards during training and full hashes at completion.

Before the Probe, run two real updates per arm with small batches, debug seed27
and <=600s. That engineering result proves wiring/finite/backward/checkpoint
behavior only; it cannot supply the Probe screen. Observe GPU utilization,
memory and throughput before proceeding to the300-update run.

Outputs:
`outputs/cm-pointflow-effect-pretrain/contactpose-transport-auxiliary-engineering-20261008-r1/`
and `outputs/cm-pointflow-effect-pretrain/contactpose-transport-auxiliary-20261008-r1/`.
The new standalone runner never edits or resumes the original training process.

## Engineering acceptance repair

At implementation `0303ed8`, engineering r1 completes both two-update arms,
finite main/auxiliary backward and final checkpoints. Initial model hashes
match. Initial macro/Oak/transport EPE are bitwise equal; an ancillary GRAB
h12 rotation metric differs by1.52e-5rad from CUDA floating-point reduction
and acos sensitivity. The blanket JSON equality assertion therefore marks
the run FAILED. Preserve the original log/checkpoints; this is not a Probe
result. Eligibility now checks all primary and per-source h24 EPE within
1micrometre, while model/data/RNG identity remains exact. This does not
change the3%/5%/10% scientific screen. A regression rejects actual primary
metric drift. Re-run the same bounded engineering budget before production.

Engineering r2 at `163160c` completes in24.72s with ENGINEERING_PASS.
Both arms start with identical model hashes and all primary/per-source h24
initial EPE deltas are exactly zero. Main and auxiliary backward are finite;
both full checkpoints save successfully. Peak live CUDA allocation at the
two-example engineering batch is1.01/1.27GiB. It does not predict production
batch32 peaks or provide an auxiliary utility verdict. GPU0 is released.
