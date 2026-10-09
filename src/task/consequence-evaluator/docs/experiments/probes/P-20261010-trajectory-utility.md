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

The three arms are intended to be matched at initialization:

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

C0's shuffle drop (25.64pp) is an exploratory indication that the new model
can use the candidate trajectory. However, a post-run implementation audit
found that r1 actually constructed C0/C1/C2 from three independent random
initializations and saved a fourth, unused model as `initial.pt`. Its raw
C1/C0/C2 comparisons are therefore initialization-confounded and are not a
matched-arm method result. The r1 within-arm fit and shuffle numbers remain
as engineering evidence only; they cannot support or refute the GT/PW gates.

### Matched-initialization correction

This was a blocker for the intended decision, so one bounded correction was
run before expanding the candidate bank. It kept the exact data, split,
seed, sampler, batch, 1200-step cap, and gate, but deep-copied one shared
`initial_state` into all three arms and saved that same state as `initial.pt`.
The old r1 directory was not overwritten. The corrected fit was first
recorded as `outputs/consequence-evaluator/trajectory-utility-fit-20261010-r2/`
and then replayed once after committing the fix as
`outputs/consequence-evaluator/trajectory-utility-fit-20261010-r3/`. The clean
replay completed on GPU2 in 31.1 seconds; its initial and C0/C1/C2 checkpoint
hashes are byte-identical to r2, while the manifest records commit `15ca96f`.
Shared initialization is explicit in both manifests.

| Arm | Matched panel strict accuracy | Tau-shuffle accuracy | Mean regret |
| --- | ---: | ---: | ---: |
| C0 H+tau | .6282 | .3718 | .0229 |
| C1 H+tau+GT effect | .6538 | .6538 | .0725 |
| C2 H+tau+PW effect | .5897 | .5641 | .0729 |

The corrected C1 gain is 2.56pp, below the 3pp gate, and its tau-shuffle
drop is 0pp. The GT-information screen and PW-retention screen both remain
false. This is a valid matched implementation screen, still limited to five
informative anchors and 78 strict pairs; no C2 expansion, online planner, or
control claim follows from it.

## Frozen H-to-tau candidate-bank audit

The follow-up read-only audit (`e0b8e2f`, output
`outputs/consequence-evaluator/trajectory-candidate-bank-audit-20261010-r5/`)
feeds the existing H-only and action-conditioned (`HA`) bridge checkpoints to
the same panel. H-only proposals are identical across all seven candidates
(mean/max candidate RMS `0/0`), so H alone cannot form a candidate bank. HA
proposals vary, but only by mean/max RMS `1.09/1.40 mm`; its panel point RMSE
against observed tau is `59.53 mm` (H `59.54 mm`). Feeding these generated
proposals into the frozen trajectory evaluator gives C0 strict accuracy
`29.49%`, with selections concentrated on candidates 2 and 4. This is an
OOD diagnostic because the evaluator was fit on observed tau, not generated
bridge outputs; it is evidence that the current H-to-tau proposal diversity
is insufficient for a meaningful selector, not a new task-control result.

## Recovered rolling same-H panel screen

To test whether the five informative anchors were only a panel-size artifact,
the frozen ref13 recovery packets were read offline (no new simulation). The
held bank is deliberately one route cluster: the initial reanchor panel at
query 71 plus `new-o8` (parent `new-o0-actual`, query 79) and `new-o16`
(parent `new-o8-actual`, query 87), 25 synchronized anchors x 7 candidates
each. For every candidate the script uses its own `trace.pt` post-query
geometry, converts the 11 measured links into the current-object frame, and
rechecks exact H/prefix/valid32/zero-clipping contracts. The rolling cohorts
are treatment-conditioned views of one reconstructed e260 actor/motion run,
not 75 independent environments or Validation data.

The first replay was recorded as
`outputs/consequence-evaluator/trajectory-rolling-panel-audit-20261010-r7/`.
The bank has 13 informative anchors and 163 strict pairs. Candidate trajectory
spread is non-collapsed in the observed bank (initial mean/max RMS
`9.37/61.35 mm`, new-o8 `8.48/65.02 mm`, new-o16 `10.31/225.73 mm`). A
provenance audit then found that r7 loaded C0/C1 from the unmatched
`trajectory-utility-fit-20261010-r1` run. Its `.5706/.6319` comparison and
`+6.13pp` difference are therefore initialization-confounded engineering
evidence, not a matched-arm result; r7 is retained for traceability only.

The corrected replay
`outputs/consequence-evaluator/trajectory-rolling-panel-audit-20261010-r8/`
uses the matched-init r3 C0/C1 checkpoints on the identical bank and contracts:
C0 is `57.06%`, C1 is `65.03%` (`+7.98pp`), and C1 mean regret improves
`.03528` to `.03153`. However, the C1 tau shuffle is also `65.03%`, so the
tau-use drop is `0pp`; C1 remains below `.70` and `screen_gate=false`. This is
held offline Probe evidence, not a supported ranking or control result. No
C2/PointWorld pass or online control was started after this corrected negative
screen.

A read-only selector-margin audit of the r8 `panel-predictions.npz` found no
exact C1 score ties, but the top-two score margin was at most `.01` on 35/75
panels and at most `.03` on 53/75. The teacher top-two margin was at most
`.02` on 70/75 panels. Tau shuffling changed raw C1 scores (maximum absolute
change about `.110`) without changing strict pair accuracy. These margins do
not justify a deadzone or an online selector: candidate selection remains
blocked by the failed matched information screen and the tie-heavy teacher
labels.

## Frozen planner-chain wiring audit

The next bounded audit (output
`outputs/consequence-evaluator/trajectory-planner-panel-audit-20261010-r4/`)
connects the frozen HA bridge, PointWorld checkpoint, and C2 trajectory
evaluator on one complete 25-anchor panel. Each anchor supplies one measured
current state; the planner expands it to seven fixed bridge proposals,
predicts one PointWorld effect per proposal, and scores the result with C2.
The PointWorld repeat call is bitwise stable. This verifies the shape/frame/
hash contract for `H -> tau -> E_PW -> Q`, but does not make the bridge a
direct trajectory policy.

The generated bank has the same small spread as the prior audit (mean/max RMS
`1.09/1.40 mm`), and its point RMSE against the observed panel trajectories is
`34.37 mm` (H24 `44.97 mm`). The OOD C2 selections achieve `56.41%` strict
pairwise accuracy, `20%` top-1 agreement, and mean regret `.01625`; 92% of
anchors nevertheless fall in a teacher tie set. The selected candidates are
concentrated on candidate 5 (10/25). These numbers are not a ranking claim:
the evaluator was fit on observed tau and the generated bank is both narrow
and bridge-induced. They only show that the complete offline adapter is
executable and that candidate-bank diversity remains the dominant blocker.

Three audit attempts before r4 were preserved: r1 passed a directory instead
of `C2.pt`, r2 exposed a panel-indexing error, and r3 exposed a stale shape
assertion. None reached a scientific result; r4 is the corrected run.

## Frozen bridge plan-sensitivity audit

The candidate-collapse blocker has two plausible explanations: the HA bridge
may be ignoring the known plan, or the fixed residual plan amplitude may be
too small for its output to show useful trajectory diversity. A CPU-only,
inference-only audit held the observed 25 x 7 same-H panel and HA checkpoint
fixed, then multiplied the panel plan by `0/0.25/0.5/1/2/4`. It did not run
Gym, alter the checkpoint, or treat scaled plans as valid controls.

Output: `outputs/consequence-evaluator/trajectory-bridge-plan-scale-audit-20261010-r1/`.
The bridge is responsive: mean non-baseline candidate spread (RMS over the
panel's 24 x 11 x 3 points) is approximately `0/0.30/0.60/1.18/2.31/4.39
mm`; the corresponding maximum is `0/0.56/1.11/2.18/4.20/7.82 mm`. The
original scale-1 bridge error remains about `59.53 mm`, and scale 4 slightly
worsens H24 error (`77.90` to `78.28 mm`). Thus the prior narrow bank is not a
shape/wiring failure—the plan branch has a measurable, roughly linear
response—but simply amplifying the residual does not reach the observed
trajectory spread and leaves the proposal out of distribution.

This is an attribution `UNCLEAR` result, not a route rescue. Do not insert a
scale multiplier into the online planner or claim candidate ranking/control
benefit. The next useful route remains an explicit H-to-tau proposal/candidate
bank with non-tied same-H consequences, or a separately justified data
collection change; further evaluator epochs and amplitude sweeps are stopped.

As a separate frozen configuration check, the existing `physical` plan-unit
bridge (`hand-execution-fit-20261009-r2`) was evaluated on the same panel in
`outputs/consequence-evaluator/trajectory-candidate-bank-audit-20261010-r6/`.
Its HA spread rose only to `1.59/2.12 mm` mean/max, with point RMSE
`59.45 mm`; the matched C1/C2 evaluator screens were `.6795/.5256` strict
accuracy on the same 5 informative anchors/78 pairs, both below the fixed
`.70` screen. This closes the simple uniform-vs-physical plan-unit
explanation without creating ranking or control evidence.

## Attribution and next step

The implementation contract is valid: C0/C1/C2 receive distinct tau/effect
branches, the panel remains same-H, and no future contact label is exposed.
The corrected result is evidence-insufficient rather than a refutation of the
complete chain. The dominant limitations are the exposed panel (20/25 anchors
are tied, only five are informative), the lack of a controlled independent
proposal bank, and the failed tau-use screen after matched initialization.
Do not increase epochs, change thresholds, or promote C2 to a deployable
planner from these runs.

Keep the frozen evaluator artifacts and move to the next cheapest route work:
audit or construct a candidate bank with non-tied same-H consequences and
explicit H-to-tau proposals, while separately preserving the known
source/backend mismatch in ref7_2 native execution. Any future online run
must use a fresh experiment card and cannot treat this oracle C2 as deployable.
