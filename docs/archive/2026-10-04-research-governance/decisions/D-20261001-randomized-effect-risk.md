# Decision: replace exact paired replay with a randomized estimand

- **Problem:** Fresh32/32physical runs completed, but seven arms in one panel
  exceed frozen pretreatment joint/quaternion tolerances. The zero-repeat also
  fails. Original status stays FAILED; no causal model winner is reported.
- **Evidence:** Initial-state/property/RNG checks succeed and input hashes stay
  fixed. Small exposed-state divergence before treatment prevents the intended
  individual paired-state comparison. Tightening replay diagnostics indefinitely
  does not advance the learned-effect question.
- **Action:** Use new genuinely randomized pulse assignment, independent of
  pretreatment history, and known propensities. Compare conditional-effect
  *risk differences*, identifiable from randomized outcomes, instead of claiming
  access to each state's two cloned potential outcomes. Freeze models and gates.
- **Cost/stop:** Four new768-environment batches (3072windows), one idleGPU,
  <=3600s/2GiB. Stop if any environment lacks a complete preselected window,
  assignment/probability audits fail, actor updates, clipping, drift or budgets.
  Positive risk improvement permits a new action-selection design; failure ends
  this direct archived-factual-model route. No refitting on its test responses.
- **Authorization:** User-authorized autonomous experiments in independent
  worktree; within existing resource limits. No external edits or physical robot.

This is a different new trial, not reanalysis that salvages the failed paired gate.
