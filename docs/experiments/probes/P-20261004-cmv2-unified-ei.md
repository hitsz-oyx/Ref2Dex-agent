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

Run artifacts: `tmp/P-20261004-cmv2-unified-ei-k1-v3.json` and `tmp/P-20261004-cmv2-unified-ei-k4-v3.json`.

The learned I head is evaluated separately from a derived audit. The derived audit uses the predicted E and the same command-derived hand flow, then applies the geometric field function; it is not the Cmv2 output.

| Probe | Best direct action I RMSE | Best Cmv2 I RMSE | Cmv2 relative change | Cmv2 derived-I RMSE |
|---|---:|---:|---:|---:|
| K=1 | 0.4500 | 0.4442 | 1.3% lower | 3.2558 |
| K=4 | 1.1203 | 1.1003 | 1.8% lower | 6.2118 |

Cmv2 also has lower E RMSE in both probes, but the E target is a compact rigid effect and is not directly comparable to the previous 13D Gate2 effect metric.

## Status and limits

This is a `PROMISING` command-proxy feasibility signal for adding I to Cmv2. The learned I advantage is small and comes from one trace family, one object, one seed, and 16/8 environment splits. The label builder was corrected so current contact points are compared in the future object frame; the v3 artifacts are the reported results. The derived-I audit is much worse, so the current result supports a learned interaction head rather than “predict E, then directly calculate I”. It is not yet a cross-hand or policy-utility conclusion. A 64-transition MANO audit produced finite 13D labels with the same field contract (`tmp/P-20261004-cmv2-mano-i-audit.json`), but it did not train a MANO predictor. A larger multi-object Inspire split remains before formal claims.

The previous Cmv2 K4 result that compared a single-step checkpoint against cumulative flows is retired as an unfair comparison; both branches in this card were retrained for their own K.
