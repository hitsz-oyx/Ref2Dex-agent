# No-Cm oracle r1: actual execution, conditional-query contract invalid

P-20261003-effect-interaction-oracle-r1 COMPLETED / UNCLEAR. Actual source
101a493,16query panels +4 deployed panels,1920trajectories/387840controlticks,
6000task-Q updates, zero Cm/actor/PPO updates. One idle GPU6, CPU data/GPU
PhysX/CUDA inference+fits. Own scratch P0 only; same nominal corrected airplane,
TRAIN761/EVAL762, one model seed806, eight fixed options at decision36/H32.

| Actual deployed selector | Full105 successes /96 | Motion0/1/2 |
| --- | ---: | --- |
| P0 |28|0/26/2|
| State + option |58|0/26/32|
| True short effect query |62|0/30/32|
| True attributed interaction query |59|0/27/32|
| Joint effect + interaction query |59|0/27/32|

These are execution records, **not certified oracle gains or a negative oracle
utility result**. All candidates share a bit-identical entire36tick prefix,
but selected per-env options executed in a mixed whole scene do not reproduce
the short future that was queried with that same option applied globally.
Max object-position discrepancy: state9.946mm, effect41.469mm,
interaction/joint162.639mm. Native q, velocities, contact descriptors likewise
fail the fixed tolerances. Four semantic oracle gates fail; numeric outcome
gates also fail. No relaxation of criteria or headline success-rate promotion.

All20panels independently pass actual25002vertex support/105tick labels,
native action range/null/PD decoding, actual actor mass and attributed normal
force reconstruction. All4 final-Q NumPy predictions max2.38--4.77e-7 and
all96 argmax choices match. TRAIN-only normalization maxerrors6.46e-6/9.33e-6;
independent first3 AdamW steps per arm loss error<=5.92e-8.12/6000steps
replayed, remaining5988 not independently replayed. These checks qualify
implementation components but do not fix the conditional future mismatch.

Minimal saved-array diagnosis: state selector's32option7 environments have
identical targets/actions untiltick44. First net-force difference appears
tick43 at1.86e-9N; first q/object-state difference tick44, then target/action
differences tick45 follow changed feedback. Contact records show zero pairs
between different environments. This does not yet identify root cause; GPU
solver aggregation/hidden warm states and host target tensor lifetime are
engineering hypotheses. Baseline-only replay passes; mixed-action isolation
does not follow from it.

Next: minimal native engineering query/mix qualification with persistent
CPU PD target storage, followed by a deterministic-physics control if needed.
No extra training, horizon/option/seed tuning or core Cm rejection from r1.
