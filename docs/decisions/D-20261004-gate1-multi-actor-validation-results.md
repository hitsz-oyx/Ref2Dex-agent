# Gate 1 multi-actor consequence validation result

**Experiment.** `VAL-20261004-gate1-actor-holdout-h32` uses 14 pre-step raw runs,
380 episodes and 190,897 complete windows. The data contain five checkpoint
namespaces (the 14 runs are not 14 independent actors):

`0fe81f67`, `16fd261b`, `84bab7f7`, `afdb2cdf`, `cf777223`.

The frozen recipe is history length 10, horizon 32, deterministic quaternion
sign repair, contemporaneous object-frame interaction, GRU temporal encoders,
exact Monte-Carlo return, and episode-balanced MAE. Outcome coverage is 13
stable-success and 10 drop-after-success episodes; the remainder are ordinary
failures. No Cm or online policy training is used.

**Pooled actor-hash fit.** With the same composite episode split for all arms,
`V_HEI` relative to `V_H` improved by 14.0%, 21.8%, 33.2%, 40.0% and 40.7%
for five seeds. The five-namespace cluster bootstrap intervals were:

`[−0.12, 3.85]`, `[0.73, 4.40]`, `[0.93, 20.08]`, `[1.97, 7.19]`,
`[2.26, 11.02]` MAE reduction; four intervals excluded zero and one crossed it.

The matched future-action control `V_HFEI` relative to `V_HF` improved by
13.4%, 16.9%, 24.5%, 28.9% and 31.7%; four of its five namespace-cluster
intervals were positive. Therefore the direct gain is reproducible, but future
on-policy actions are also a strong proxy and cannot be ignored.

**Outer namespace holdout.** Holding out each checkpoint namespace in turn gave
`V_HEI` improvements of 57.7%, 40.3%, 27.1%, 32.0% and 29.1%. Four held-out
episode intervals excluded zero; the e300 namespace interval crossed zero
(`[-5.49, 23.32]`). The paired incremental estimand
`(H−HEI)−(HF−HFEI)` was `5.01`, `2.31`, `2.45`, `−1.36` and `3.70` MAE for
the five held-out namespaces. Its five-namespace bootstrap mean was 2.42 MAE
with CI `[0.39, 3.98]`, while one namespace remained directionally negative.

**Decision.** The corrected consequence representation is **PROMISING as a
pooled and outer-namespace mechanism probe**, but the formal Gate 1 claim is
**INCONCLUSIVE**. The evidence does not justify saying that E/I is a uniquely
sufficient mediator of value, nor that the five checkpoint hashes are five
independent training lineages. The current split and reports do not establish
unseen-real-actor generalization.

Do not start online Cm or distillation from this result. Preserve the frozen
bridge and the paired-control audit, then resolve provenance/lineage and repeat
with predeclared lineage-disjoint actors if a formal Gate 1 claim is required.
The artifacts are `tmp/gate1_validation_all14_h32_actorid.pt`, the
`tmp/VAL-20261004-gate1-pooled-actorhash-h32-*.json` reports, and the
`tmp/VAL-20261004-gate1-namespace-holdout-*.json` reports.
