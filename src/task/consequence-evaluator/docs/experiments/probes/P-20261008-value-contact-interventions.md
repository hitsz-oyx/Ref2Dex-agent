---
schema: ref2dex.probe.v2
probe_id: P-20261008-value-contact-interventions
experiment_id: P-20261008-value-contact-interventions
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: b3c6d78
claim_id: C3
hypothesis_family: HF-consequence-value-interventions
probe_index_in_family: 3
seed_pool: probe
seeds: [251, 252, 253, 254]
decision_changed_if_positive: prepare independent outcome groups with the frozen measured contact-error sampler
decision_changed_if_negative: close this bounded error-sampling recipe and record the unresolved negative-coverage decision
status: RUNNING
run_id: official-value-contact-train-20261008-r1
---

# Is initial contact more informative than already stable holding?

Result: Pending first64train episodes.
Decision: Both placing and stable-held empirical banks reached physics but
failed negative coverage. Test initial geometric contact before stable holding,
using newly computed learner/expert errors at that same phase and unchanged
amplitude bounds and task labels. No label thresholds are tuned.

## Minimal Decision Probe

This serves consequence-evaluator's S/P/M data route and eventually C3, without
changing the Mission's final self-trained-policy benefit claim. Distinguish
timing failure (contact perturbations yield negatives) from a bounded error
sampler with too little task authority (even initial contact stays successful).

Reuse the [same-H policy error method](P-20261008-value-policy-error-interventions.md),
but bank observations start at the first measured near-hand, unsupported-lift
not-yet-achieved state in32nominal train seed230episodes. Each has a full24step
source segment before placing. Expert replay<=1e-5 remains mandatory; actor
SHA8f6823db… and learner SHA8882fabd… stay frozen. Own RMS once per actor;
learner-minus-expert errors, gains0.25/0.5/1, smooth taper, zero coupled channels,
0.2control/0.05translation caps. Empirical candidate bank, not fitted covariance.

64train episodes at251,32env/two waves, halfclean/halfcontact. Online trigger
uses first valid geometric near-hand phase, before lift overrides it, no force
proxy. At most one complete24plan then same official continuation to full
reference end, no forks. Gate:>=24actual triggers,>=24clean successes and>=4train
failures; only then collect64val at252 and64test at253 using the unchanged bank.
All are Probe seed groups. Prepare with pair seed254/stride8, success/failure
minimum4/2/2. Source/seed splits disjoint, no held-out sampler tuning.

## Decision Note and stops

The two preceding effective Probes excluded the current placing and stable-held
samplers as efficient negative generators; no emulator fit was started. This
last stage test either permits a small split-disjoint dataset or closes the
bounded empirical recipe. Do not append stages/seeds after a failed coverage
gate simply to meet counts. Keep measured-input failures separate from negative
method evidence, and defer stronger residual or other disturbance routes to a
new evidence-grounded decision.

One idle GPU2; bank inference<=120s, total collection<=900s, each<=250s,
<=32env and64episodes/run, <=2GiB expected raw total, <=1GiB labels/audit.
CPU only for file/statistical preparation<=120s per step. Commit before
collection and freeze/check all source hashes. Stop on occupancy, drift,
incomplete plans/episodes, nonfinite geometry or budget. No external writes,
checkpoint/environment changes, branch or push. Sampled geometry/support and
one-object/reference weakness remain; no formal utility/generalization claim.
