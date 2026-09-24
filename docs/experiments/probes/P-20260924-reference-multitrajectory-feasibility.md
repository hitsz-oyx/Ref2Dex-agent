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

Both complete evaluations finished. The reference controller achieved
**9/128 held-lifts (7.0%)**: mug 8/10, airplane 1/30, and zero on the
other eight identities. Contact was sometimes present without supported
lift. The prespecified six-identity and 50% gates failed decisively. Status:
`UNPROMISING` as a physical multi-object BC teacher for these 12 motions.
This does not judge the quality of raw GRAB demonstrations or the official
actor; it judges this particular reference-action controller in DExplore.
Results and manifests are in
`outputs/Dexplore/agent_multitrajectory12_s70_e300/eval_s20{3,4}_e300_full_reference_lead1/`.

Next decision: isolate one physically feasible but shared-policy-failed
identity, duck, to distinguish policy interference from lack of an effective
reward/curriculum transfer on that object.
