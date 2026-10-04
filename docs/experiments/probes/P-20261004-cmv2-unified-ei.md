# P-20261004-cmv2-unified-ei

## Question

Can the spatial Cmv2 encoder predict a hand-identity-free interaction consequence on the same Inspire trajectories used by a direct command baseline?

## I contract

For fixed object-surface anchors, the pooled interaction target is 13D:

`[contact_mass, centroid_xyz, covariance_6, normal_approach_speed, tangential_speed, q90_tangential_speed]`.

It uses unordered hand surface points in the current/future object frame. It does not use finger or body IDs, joint indices, MANO shape, or force channels. The definition is therefore suitable for MANO, Inspire, or reconstructed video point clouds.

## Matched data and input permissions

- Source: the same Inspire physical trace (`plain_off_t286_s288_first/trace.pt`), not the old Gate2 dataset and not the MANO cache.
- Split: environment IDs 0–15 train and 16–23 held out; 32 windows per environment.
- Object: airplane mesh, 256 fixed anchors; Inspire hand surface, 512 fixed points.
- Direct baseline: current 55D state plus recorded 18D command chunk.
- Cmv2: current object/hand geometry plus hand flow generated from that same command through the exact DExplore target mapping and Inspire FK. Measured future hand geometry is used only for E/I labels.
- K1 and K4 are trained separately with the same horizon-specific targets. Best validation checkpoints are retained.

## Results

Run artifacts: `tmp/P-20261004-cmv2-unified-ei-k1-v4.json` and `tmp/P-20261004-cmv2-unified-ei-k4-v4.json`; context and explicit-field follow-ups are in the corresponding `v6`/`v7` artifacts. The second explicit-field seed is `tmp/P-20261004-cmv2-unified-ei-k1-v7-s2.json` and `...k4-v7-s2.json`.

The learned I head is evaluated separately from a derived audit. The derived audit uses the predicted E and the same command-derived hand flow, then applies the geometric field function; it is not the Cmv2 output.

| Probe | Direct baseline | Cmv2-only | Cmv2 + state/action | Cmv2 + explicit field, seed 1 | seed 2 |
|---|---:|---:|---:|---:|---:|
| K=1 | 0.1069 | 0.1695 | 0.1131 | 0.1063 (-0.4%) | 0.1193 (+7.5%) |
| K=4 | 0.3318 | 0.3990 | 0.3924 | 0.3984 (+1.8%) | 0.4334 (+10.5%) |

Cmv2 has lower E RMSE in both probes. The I labels have nonzero contact mass in 45.0% of held-out windows after using the correct actor order and batched geometry. The state/action-conditioned Cmv2 I head nearly closes the gap, but does not beat the direct baseline in this probe.

## Status and limits

The corrected Cmv2-only probe is `UNPROMISING` for I. Per-dimension target normalization (`tmp/P-20261004-cmv2-unified-ei-k1-v5.json`, `...k4-v5.json`) did not close the gap. Adding causal global state/action conditioning reaches near parity but does not improve direct prediction. Explicit contact-field pooling produced a one-seed K=1 tie, but the second seed was worse on both horizons, so this route remains `UNPROMISING` for a robust improvement. The earlier v3 results are invalid because the actor order and batched link broadcast were wrong. The derived-I audit is also poor, so “predict E, then directly calculate I” is not ready to adopt. A 64-transition MANO audit produced finite 13D labels with the same field contract (`tmp/P-20261004-cmv2-mano-i-audit.json`), but it did not train a MANO predictor. Decision: retain the hand-agnostic I contract and corrected geometry pipeline, but stop tuning this Cmv2 head and move to a different I prediction design before any formal validation.

The previous Cmv2 K4 result that compared a single-step checkpoint against cumulative flows is retired as an unfair comparison; both branches in this card were retrained for their own K.
