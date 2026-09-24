# P-20260924-reference-multitrajectory-feasibility

- Classification: Decision Probe.
- Cm: off; no official actor actions.
- Question: Can the 12 corrected GRAB reference trajectories, executed by
  the repository's reference-action controller, create broad physical
  held-lift on the same multi-object input where PPO gave 16/128?

Run two 64-environment complete first-episode evaluations on the fixed
12-motion input, seeds203/204, with `--reference-action-lead 1`. The loaded
checkpoint is only the evaluator's required player shell; executed actions
are replaced by `inspire_reference_action`. Report overall and per-object
held-lift/contact. If at least six object identities have >=1 held-lift and
the pooled rate is >=50%, collect observation/reference-action pairs and
train a bounded self-trained BC baseline; otherwise the reference controller
is not a sufficient teacher for this pool and the next baseline must alter
trajectory filtering/physical data or training curriculum. This gate only
chooses a route, not a claim of multi-object policy success.

One or two idle GPUs, <=20 minutes total, <100 MB output. Stop for data or
code drift, GPU conflict, incomplete evaluation or nonfinite metrics.

## Results

Pending.
