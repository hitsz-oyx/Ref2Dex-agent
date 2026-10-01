# P-20261001-differential-response-learning

Status: revised before learning; the physical diagnostic is already known.
Initial conditional design was written before physical metrics and was not run.
This revision follows D-20261001-vector-response-learning.md; it does not change
the original physical Probe's UNPROMISING label or gate.

## Hypothesis and decision

Conditional on both physical-resolution subgates passing (the monotone-lift
component failed), test whether explicitly supervising full vector physical
differences improves prediction of action effects on a different frozen
actor's planned action sequences. If it fails to beat strong simple controls,
stop this small-data implementation. This cannot establish task utility.

## Fixed design

- Use only complete contact-response r1 outputs, verifying output hashes.
- Fit all windows from actor seed286 (both eval seeds); test all windows from
  actor seed287. No test normalization, selection, calibration or optimization.
  The three motions and object remain shared. This is actor-panel transfer,
  not independent object/task generalization. These are reused pilot panels.
- Per arm input: four actual pre-intervention physical states (zero pad/mask
  if insufficient history), and ten planned normalized native actions starting
  at the pulse. Plans are predefined reference replay actions plus the pulse;
  no future physical state is input. Plans are retrospective diagnostic banks,
  not deployable actor predictions. Repeats of a motion are not new motions.
- Targets: object world-position displacement at horizons5 and10, subtracting
  that arm's actual pre-intervention object position.
- Capacity: two128-unit SiLU hidden layers; same train-only normalization and
  fixed optimization mini-batches for each method. Seeds301/302/303,1000updates,
  Adam learning rate0.001, batch96, no checkpoint selection.
- Three methods: factual action-aware MSE; factual plus paired plus-minus
  contrast MSE (coefficient1, contrast scaled by fit contrast RMS); and
  no-pulse-input factual MSE, replacing each arm's plan with its zero-arm plan.
  The last control retains the common future reference plan and actual past
  physical history; it cannot observe the assigned pulse.
- Strong simple baseline: predict the fit mean plus-minus effect for every
  test window. Also report a per-motion fit-mean baseline with privileged motion
  identity; it diagnoses whether there is evidence for a learned state-dependent
  representation beyond motion templates.
- Primary: for BOTH horizons, seed-mean differential model contrast RMSE must
  be >=10% lower than factual AND global constant-effect controls. Differential
  z-sign agreement with measured test contrasts must be>=75% at both horizons.
  Report every optimization seed and the no-pulse/per-motion baseline. Positive
  gate is PROMISING for this predictive screen only, not causal policy benefit.
- GPU4 after physical collection releases it; one admitted GPU,2CPUthreads,
  <=600seconds, <=100MiB artifacts. Stop on nonfinite loss, source/input drift,
  empty split or resource conflict. Do not add seeds or alter loss to pass.

## Interpretation boundaries

The actual warm states differ slightly across native processes despite identical
cold snapshots. The pair loss uses each arm's observed causal past and its plan;
it does not pretend that warm hidden solver state is identical. One plus/minus
replicate per panel is insufficient to calibrate per-window intervention noise.
Training can still fit this noise. Held-out errors compare measured diagnostic
responses; they do not establish correct physical potential outcomes. Sample
counts do not equal the number of independent actor training seeds or tasks.
The test panels' aggregate physical contrasts have already been inspected in
the pilot. This is a reused-data feasibility screen, not a pristine held-out
Validation. Only optimization and normalization exclude test-panel rows.
