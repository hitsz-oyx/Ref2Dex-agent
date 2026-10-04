---
schema: ref2dex.probe.v2
probe_id: P-20261004-ref5-surface-i-gt-value
date: 2026-10-04
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 624dfe3
claim_id: C3
hypothesis_family: REF5-SURFACE-I
decision_changed_if_positive: retain surface I as a candidate auxiliary and require a fixed independent repeat before any predicted-I head
decision_changed_if_negative: do not train an I head or K4 I; keep point-flow E to G as the main route
probe_index_in_family: 1
seed_pool: probe
status: UNPROMISING
---

# Ref5: does GT spatial surface I add G information after H and E?

This Decision Probe asks whether eight spatial object-surface interaction
patches warrant a predicted-I route. The cheapest discriminator is a GT
information ablation, without new collection, an I prediction head, or policy
training. The final objective remains a causal Cm contribution to a trained
grasp policy; a G regression gain cannot establish that contribution.

## Fixed method and bounds

Run `surface_i_gt_rootfix_s202`: audited historical dataset
`tmp/e260_all4_h16_histfix.pt`, four fresh pre-step source_e260 shards,
112 recorded episodes, 64 evenly spaced decision rows per episode, 7,168
total rows. Held-out split groups are `(source_run, episode_id)`, using the
historical split identifier `20261004`, 90 train and 22 test episodes.
Training RNG seed is **202**, in the Probe pool; the historical split and
mesh sampling identifiers are not new validation RNG allocations.

H is ten past state/action/context/progress packets. E is one-step 6D
object-relative translation and rotation vector, derived from the exact
stored GT pose. This deliberately differs from the previous K8/13D E
bridge, so numerical comparisons with that bridge are not valid.
G is recorded exact discounted Monte Carlo reward return, with no bootstrap.

I uses the airplane's fixed 256 mesh anchors and 512 material hand-surface
samples. Eight deterministic farthest-point/Voronoi object patches retain
spatial location. Each has geometric soft contact mass, weighted surface
distance, normal relative velocity, tangential relative velocity, and contact
mass change. These are geometric proximity proxies, not measured physical
contact or force labels. Current and future material points are expressed
in their respective object frames; the same material hand point is used for
velocity, avoiding a nearest-point identity switch.

Three identical H/E/I GRU bridges use the same initial seed, parameter count,
16 epochs, 512 batch size, AdamW 0.002 and TRAIN-only normalization:
`HE_zero_I`, `HE_surface_I_GT`, and `HE_pooled_I_GT` (global pool repeated
across eight slots). Zero I is the capacity-matched H+E control.
Predeclared pass condition: spatial GT I reduces episode-balanced MAE by
at least 3% versus zero I, with positive lower episode-bootstrap delta.
Otherwise do not train an I predictor. GPU7 was idle before use; admitted
budget was one GPU, 20 minutes, 2 GB artifacts. Actual effective run took
20.43 seconds; no policy, actor, or Cm updates were made.

## Engineering audit and correction

Two initial audit-only attempts failed before training: the historical
assembled source_run IDs use metadata order rather than the current
assembler's sorted shard hashes; and the assumption of an identity hand
actor root fails on fresh shards. Runs `surface_i_gt_audit_s202` and
`surface_i_gt_audit2_s202`/`surface_i_gt_audit3_s202` are engineering failures,
not scientific negatives. The first episode at decision step >=9 has
sub-micrometer identity-root FK error, but later episodes contain real
root translations up to **22.33 mm**, with negligible rotation.

The corrected GT reconstruction obtains a common rigid hand root from one
measured body pose and its FK transform, then verifies against the other
four independently stored bodies. Maximum residual across all selected
current/future samples and five bodies is **5.35e-6 m**. State versus shard
alignment and next-state continuity errors are zero; state object pose
(allowing quaternion sign) error is **1.19e-7**, and reconstructed E versus
stored E error is **2.38e-7**. No force or body identity enters the surface
field representation. Measured future geometry is solely a GT label input.

The later audit-only cache run `surface_i_gt_rootcache_s202`, code `f102567`,
saves absolute source-row indices and explicitly separate current/future
hand-root transforms for reuse. Future roots are forbidden predictor inputs.
Only current observed roots may repair the parallel point-flow G probe.

## Results and decision

| Matched bridge | Episode-balanced test MAE |
| --- | ---: |
| H + GT E + zero I | 22.6737 |
| H + GT E + eight spatial GT I patches | 20.2171 |
| H + GT E + pooled GT I | 18.8559 |

Spatial I gives a **10.83%** mean gain over zero I, but its paired
episode-bootstrap absolute MAE delta interval is **[-0.283, 7.438]**.
Spatial I is 7.22% worse than pooled I; its delta against pooled I is
**-1.361**, interval **[-4.758, 0.815]**. It fails the predeclared gate and
is `UNPROMISING` for starting a spatial-I predictor now. This does not
refute all interaction priors or establish absence of independent I information.

The high-return held-out episode `18620000000`, source_run 0, dominates
the apparent benefit: zero/surface/pool MAE **349.48/296.86/262.42**.
Excluding this single episode as a diagnostic sensitivity (not changing
the primary metric), mean spatial-I absolute gain is just **0.068** over
the remaining 21 episodes. No validation seed, architecture, or gate was
changed after viewing this diagnostic. Root remains responsible for the
parallel G decision and the next policy-facing route.

| Held-out source_run | Episodes | Zero I MAE | Spatial I MAE | Pooled I MAE |
| --- | ---: | ---: | ---: | ---: |
| 0 (fresh_pre_s86e) | 7 | 52.8587 | 45.4126 | 40.5130 |
| 1 (fresh_pre_s86f) | 3 | 10.5999 | 9.6364 | 11.8052 |
| 2 (fresh_pre_s86g_retry) | 7 | 9.5892 | 9.3985 | 9.0715 |
| 3 (fresh_pre_s86h_retry) | 5 | 5.9775 | 6.4377 | 6.4644 |

## Reproduction and evidence

The implementation is
[`probe_surface_i_gt_value.py`](../../../tools/run/probe_surface_i_gt_value.py).
Executed command (GPU remapping is physical GPU7):

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 \
  /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  src/task/cm-interaction-oracle/tools/run/probe_surface_i_gt_value.py \
  --output-dir outputs/cm-interaction-oracle/surface_i_gt_rootfix_s202
```

The complete result, input/shard/mesh/URDF/script hashes, argv, grouped
metrics, split groups, training losses, and audit live in
`outputs/cm-interaction-oracle/surface_i_gt_rootfix_s202/results.json`.
The cache is
`outputs/cm-interaction-oracle/surface_i_gt_rootcache_s202/geometry_cache.pt`;
its hash and audit are in the adjacent `audit.json`.

## Limitations and future evidence

One seed, one fixed source actor/object, single-step GT labels, and the
high-return episode sensitivity prevent formal independent-value or policy
claims. Geometric soft proximity is not a calibrated contact label; mesh
vertices approximate surface area and patches have different anchor counts.
Global pooling uses equal patch weighting, so its comparison tests this
specific pooling recipe. Multi-seed/robust-target checks and predicted-I
utility remain deferred evidence, not an immediate experiment queue. No
I head, I K4 sweep, or online policy experiment is licensed by this result.
