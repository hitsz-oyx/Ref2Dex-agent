---
schema: ref2dex.probe.v2
probe_id: P-20261010-generated-tau-interface-diagnosis
experiment_id: P-20261010-generated-tau-interface-diagnosis
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: e3f5a17
claim_id: C3
hypothesis_family: HF-generated-tau-execution
probe_index_in_family: 2
seed_pool: probe
seeds: [298]
decision_changed_if_positive: repair proposal bootstrap coverage if handoff restores acquisition
decision_changed_if_negative: fix online tau executor interface before new proposal training
status: PROMISING
run_id: generated-tau-interface-diagnosis-20261010-r2
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

8330b07 r1 completed128controls in33.88s, GPU2~7.5GiB/util52--69%. Original
GT4/4 held45, median64.5frames/terminal4; online GT1/4 held45,
median34.5/terminal0; displacement and eight-step handoff both0/4 even grasp.
Online GT has no clipping but loses the object. Local interface UNPROMISING;
bootstrap alone does not explain the difference. Independent numpy audit
passes: all measured H/current errors0, privileged GT hand transform2.98e-7,
actual897features2.86e-6/command2.38e-7/PD0, handoff first8steps reconstruct
original GT controls. Outputs `generated-tau-interface-diagnosis-20261010-r1/`
and `generated-tau-interface-audit-20261010-r1/` including behavior.png.

## Boundary derivative repair Decision Note

Saved online GT geometric q targets differ negligibly from precomputed GT q;
however q[:,0] is overwritten with live current q. Computing central velocity
at the first target as (q2-live_q0)/(2dt) mixes tracking error into nominal
velocity. At replanning boundaries its FF target difference versus precomputed
GT reaches XYZ[41.95,54.23,69.83]mm and Euler[.199,.300,.147]rad. This is an
implementation/input-semantic issue to repair before changing proposal data.
Other unisolated causes: fitted instead of intent points and chunk padding.

Cheapest next action: one additional same-seed128step diagnostic wave in the
same5GPUmin/512MiB total, <=240s. Four roles: original GT, original online GT,
online GT with velocity derived only from future geometric q (one-sided first
future target), and that corrected velocity plus raw intended GT tau as actor
conditioning. Both new arms still use the identical online geometric q/8step
consumption/padding. No model training or new seed, no true future q/object as
input; GT hand only privileged diagnostic. This sequential arm design isolates
velocity, then intent conditioning conditional on velocity repair. Same GT
calibration and held45/clipping gates, no longer-term success claim. Regression
requires first-future velocity invariant to changing the live q anchor, while
later central derivatives/angle wrapping stay intact. If either repair restores
short hold, retain it and test pure-H generated execution; otherwise investigate
remaining chunk conditioning mismatch, without more proposal training.

## Repair result and pure-H follow-up Decision Note

e3f5a17 r2 completed128steps in33.96s, ~7.5GiB GPU.
Original GT4/4held45/terminal4, no clipping; old online2/4held45/terminal0,
median37frames, no clipping. Future-only velocity with fitted points4/4held45/
terminal4, median69, clipping.390625%; future-only velocity with raw intent4/4
held45/terminal4, median64, clipping0. Interface repair PROMISING for short
acquisition; no formal causal or full-task claim. All query/current inputs0,
privileged tau transform3.58e-7, actual897features1.91e-6/command2.38e-7/PD0.
Saved masks and intended/fitted tau are independently checked in audit r2.
Original r1/r2 outputs and old velocity function remain for diagnostic controls.

Next minimal follow-up: default generated FF uses future-only velocity, retain
fitted points/8step replanning and frozen original proposal/T/executor unchanged.
One complete seed29816env/542step pure-H wave in
`generated-tau-native-execution-20261010-r3`, <=240s, included in the same total
5GPUmin/512MiB (two33--34s waves plus ~108s projected follow-up). No training,
extra seed, GT startup, new candidates or padding change. Original full-execution
>=3/4held433+terminal/clip<1% gates apply. Repair actual feedforward input only;
if generated acquisition still fails with GT calibrated, proposal/distribution
issues remain and then bootstrap data repair is justified. Previous original
combination negative cannot by itself be evidence that H->tau is ineffective,
because velocity semantics were wrong. Audit the corrected full execution before
any such local conclusion; future-only velocity invariance tests pass.

## Limitations / future evidence

Single motion/seed and separate live rows, oracle-warmstarted executor.128steps
only test acquisition/short holding, not original433frame/terminal gate, full
placing task, same-H candidate ranking, Cm-on/off or final self-trained policy.
