---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-utility
experiment_id: P-20261010-trajectory-utility
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 7c8a26e
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 1
seed_pool: probe
seeds: [408]
decision_changed_if_positive: continue candidate selection and then connect the trajectory interface to the execution route
decision_changed_if_negative: keep the offline ranking route UNCLEAR and audit candidate diversity/retarget contact state before online planning
status: UNCLEAR
run_id: trajectory-utility-fit-20261010-r1
---

# Does a real hand trajectory interface retain consequence ranking information?

Result: `UNCLEAR`. C0 (`H+tau`) has 62.82% strict pair accuracy; C1
(`H+tau+E_GT`) has 64.10% (+1.28pp), below the preregistered +3pp gain gate.
C2 (`H+tau+E_PW`) reaches 76.92%, but its tau-shuffle control is 78.21%, so
the apparent PW gain is not attributable to the candidate trajectory under
this small panel. This is an offline oracle Probe, not an execution or
deployability result.

## Motivation and decision note

The main route requires `H -> {tau_i} -> Q -> tau*`. The existing frozen
`OldUtility` accepts a native `[24,18]` action tensor and therefore cannot test
this interface; feeding tau into that branch would answer a different question.
This Probe adds a separate 24-step hand trajectory branch and keeps object
future effects separate. The cheapest discriminating data are already frozen
in `old-utility-data-20261009-r1` and `old-utility-pw-20261009-r2`, including
the 25 same-H x 7 candidate panel.

The three matched arms are:

* C0: current history `H` plus candidate `tau`;
* C1: C0 plus the measured GT object effect `E_GT`;
* C2: C0 plus frozen PointWorld object effect `E_PW`.

The result would change whether it is worth connecting candidate generation to
the evaluator. It does not change the global Cm claim, old U32 definition, or
the already failed native full-action retarget gate.

## Frozen input and model contract

`tau` is `pw_hand_future`, a 24 x 11 x 3 hand trajectory in the current
query-object frame. It is not `future[:,:,12:]`, whose points are expressed in
each future object frame. `E_GT` is `future[:,:,:12]`; `E_PW` is the first 12
channels of the frozen PointWorld `future.npz`. All three arms use the same
two-layer width-128 temporal encoder, train-only coordinate normalization,
MSE on frozen U32, episode-uniform/window-uniform batches, and val episode MSE
checkpoint selection. The panel is opened only after selection.

The panel has 25 anchors x 7 candidates, but only 5 anchors contain strict
label differences (`78` strict pairs). Candidate tau variation is finite but
small (mean panel RMS about 0.0129 in the stored coordinate units); this is a
known evidence limitation, not a reason to lower the gate.

## Fixed gates

The GT information screen requires C1 pairwise accuracy >= .70, gain over C0
>= .03, tau-shuffle drop >= .03, and no worse mean regret. The PW retention
screen additionally requires C2 >= .70, C1-C2 <= .05, and C2 regret no worse
than C1 + .05. These thresholds are exploratory screens, not formal support.

## Result

Fit output: `outputs/consequence-evaluator/trajectory-utility-fit-20261010-r1/`.
The bounded GPU2 fit completed 1200 updates in about 31 seconds with no
nonfinite values or input drift.

| Arm | Panel strict accuracy | Tau-shuffle accuracy | Mean regret |
| --- | ---: | ---: | ---: |
| C0 H+tau | .6282 | .3718 | .0229 |
| C1 H+tau+GT effect | .6410 | .6282 | .0600 |
| C2 H+tau+PW effect | .7692 | .7821 | .0596 |

C0's shuffle drop (25.64pp) confirms that the new model is actually using the
candidate trajectory. C1's incremental gain is only 1.28pp and its shuffle
drop is 1.28pp. C2's higher raw accuracy does not survive its tau-shuffle
control, and C1-vs-C2 retention is therefore not interpretable as a positive
PW result. Ordinary held-out test RMSE is recorded in `result.json`; it is not
used as a panel selection claim.

## Attribution and next step

The implementation contract is valid: C0/C1/C2 receive distinct tau/effect
branches, the panel remains same-H, and no future contact label is exposed.
The result is evidence-insufficient rather than a refutation of the complete
chain. The dominant limitation is the exposed panel: 20/25 anchors are tied,
only five are informative, and the observed tau/effect candidates are not a
controlled independent proposal bank. Do not increase epochs, change the
thresholds, or promote C2 to a deployable planner from this run.

Keep the frozen evaluator artifacts and move to the next cheapest route work:
audit or construct a candidate bank with non-tied same-H consequences and
explicit H-to-tau proposals, while separately preserving the known
source/backend mismatch in ref7_2 native execution. Any future online run
must use a fresh experiment card and cannot treat this oracle C2 as deployable.
