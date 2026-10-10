---
schema: ref2dex.probe.v2
probe_id: P-20261010-generated-tau-interface-diagnosis
experiment_id: P-20261010-generated-tau-interface-diagnosis
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-generated-tau-execution
probe_index_in_family: 2
seed_pool: probe
seeds: [298]
decision_changed_if_positive: repair proposal bootstrap coverage if handoff restores acquisition
decision_changed_if_negative: fix online tau executor interface before new proposal training
status: UNCLEAR
run_id: generated-tau-interface-diagnosis-20261010-r1
---

# Does the online tau execution interface preserve the GT acquisition control?

## Motivation / Decision Note

Decision for ref8 usable H->tau->A->Z, full Mission scope unchanged. Previous
complete native Probe made progress: established generated roles fail before
grasping despite calibrated GT4/4, with actual measured-history/command audit.
Training starts at tick8, deployment at tick0; however generated roles also
recalibrate current-q geometry and terminal-pad each chunk. Cheapest experiment
isolates these before spending more proposal training. Independent review
suggests the same interface control. No new branch, data/model/seed sweep.

Ranked falsifiable hypotheses: (1) online projection/chunk differs enough from
precomputed geometry to break even GT tau; online GT will fail acquisition.
(2) untrained padded bootstrap is dominant; with online GT passing, GT-prefix8
handoff will acquire when from-zero displacement does not. (3) generated future
quality/executor distribution remains inadequate; both displacement arms fail
while online GT succeeds. Separate live roles do not establish exact causal
same-state attribution; outcomes guide the next bounded repair only.

## Protocol

One frozen native16env seed298 wave, four4env roles in the same role positions
as the previous run: tau_gt original complete precomputed tau geometry;
tau_online GT hand points sent through exactly the current-query300step online
projection,24frame terminal-padded controller window, first8steps consumption;
displacement from tick0 only pure measured four-state H; displacement_handoff
uses the original GT hand/geometry controller for ticks0--7, then same pure-H
displacement pathway from tick8. All roles use the same frozen actor/FF/clamp.
Stop at128controls, before any full-task claim. No future robot q/object or
teacher action is a learned/generated-arm input; GT hand and original geometry
are privileged diagnostic controls and explicitly recorded. GT is injected
only after generating/scoring the ten-member pool, never eligible for selector.
Never train or promote this diagnostic GT prefix as a deployable policy.

Record raw/projected tau, current H, role-specific privilege and actually applied
897features/commands/PD/outcomes. Reconstruct all query and control paths, including
the first8steps of the handoff. Require original GT >=3/4 held45 for calibration.
Online GT >=3/4 held45 and clipping<1% means interface acquisition PROMISING;
otherwise local interface UNPROMISING if calibration passes, UNCLEAR if not.
Handoff rescue is exploratory only, no paired causal claim. If online GT fails,
compare geometry/first commands and fix that path before training proposal. If
it passes and handoff helps, train explicit padded measured-history bootstrap.
If both displacement arms fail, review candidate/executor distribution support.

## Resources / stop

One idle GPU2, <=5GPUmin/512MiB output, single128step run<=240s plus bounded
read-only audit. No training/full542step rerun/new seed. Check GPU and progress
at first step and64step intervals; preserve failed manifest. Stop on nonfinite,
source drift, native command/role/privilege mismatch or unexpected early reset.
CPU pure-file/numpy auditing is sufficient; no neural CPU evaluation required.

## Results

Not run yet. Implementation and manifest must explicitly mark diagnostic GT
privilege; prior generated execution remains unchanged and retained.

## Limitations / future evidence

Single motion/seed and separate live rows, oracle-warmstarted executor.128steps
only test acquisition/short holding, not original433frame/terminal gate, full
placing task, same-H candidate ranking, Cm-on/off or final self-trained policy.
