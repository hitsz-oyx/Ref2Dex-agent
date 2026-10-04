# P-20261004-surface-token-i

## Decision question

Does a compact object-surface interaction token give a more useful, hand-identity-free
interaction target than the previous globally pooled 13-D interaction vector when the
existing Cmv2 E encoder is reused?

## Hypothesis and probe

Ref4 suggests retaining spatial interaction topology and testing K=1 before extending
to multiple tokens. This probe uses one object-surface token with five quantities:

`[contact_mass, hand_object_distance, normal_velocity, tangential_velocity, contact_change]`.

The quantities are pooled over 256 fixed object anchors, use unordered 512-point hand
surface samples, and are expressed in the future object frame. No hand link, finger,
joint, MANO, or Inspire topology identifier enters the target. The source is the same
Inspire physical trace used by the corrected unified-E/I probe, not Gate2 or MANO.

The Cmv2 branch loads `tmp/P-20261004-cmv2-unified-ei-k1-v7.best1.pt`, freezes the
existing V1.3 spatial E encoder and trains only a new token head. A state+action MLP/GRU
predictor is the matched direct baseline. Both use train environments 0--15 and held
out environments 16--23, 16 rows per environment, six epochs, and physical GPU 6.

Command:

```bash
CUDA_VISIBLE_DEVICES=6 \
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
  scripts/probe_surface_token_i.py \
  --trace src/task/CmResidual/research/physical_value/output/P-20261001-paired-evaluator-resolution-r3/plain_off_t286_s288_first/trace.pt \
  --output tmp/P-20261004-surface-token-i-k1.json \
  --device cuda:0 --train-envs 16 --val-envs 8 --rows-per-env 16 \
  --epochs 6 --batch-size 8
```

## Results

The frozen-E Cmv2 head improves contact-mass RMSE but is worse on the aggregate token
metric. At epoch 6:

| predictor | token RMSE | normalized token RMSE | contact-mass RMSE |
|---|---:|---:|---:|
| direct state/action | 0.06899 | 0.8161 | 0.08094 |
| frozen-E Cmv2 surface head | 0.07525 | 0.8709 | 0.05281 |

The frozen-E route is therefore `UNPROMISING` as a complete K=1 token predictor on this
single probe: it is 6.9% worse in raw RMSE and 6.7% worse in normalized RMSE, despite a
34.8% lower contact-mass RMSE. The distance and velocity components remain close to the
direct baseline, while the aggregate gap is mainly from tangential velocity.

The prior 13-D reference was K=1 direct RMSE 0.10668 versus Cmv2 0.10625 (v7 seed 1),
but those values are not numerically interchangeable because the target dimensionality
and normalization differ. The new artifact records the per-dimension errors and train
normalization in `tmp/P-20261004-surface-token-i-k1.json`.

## Decision and limits

Do not extend this frozen-E K=1 head to K=4 based on this probe. The target contract is
kept as a viable hand-agnostic diagnostic, and the lower contact-mass error is useful for
the separate G bridge ablation. Whether GT surface I has independent value after E is
still unresolved here; that requires the parallel G bridge experiment. This is a single
seed Probe, not a formal validation or a policy claim.
