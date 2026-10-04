# P-20261004 Gate 2 action chunk K=8

## Question

Does replacing a single action with an 8-step action chunk make consequence
prediction useful enough to preserve Gate 1's long-return signal? This probe
also removes the previous H32 flat-output bottleneck by using a per-step
Transformer encoder/decoder.

## Design

The probe is fixed to `K=8` and uses the audited all18 dataset with n3 held
out. `Cm1` receives `(H_t, a_t)` and `Cm8` receives
`(H_t, a_t:t+7)`, where the latter uses the recorded future action chunk as a
planning-input diagnostic. Both predictors have the same two-layer
Transformer architecture and emit `[B, 8, 93]` consequence outputs; only the
action sequence length differs. A K=8 value bridge is trained on GT E/I in the
training namespaces and then frozen for H-only, GT-E/I, Cm1-predicted E/I, and
Cm8-predicted E/I evaluation.

## Results

| Quantity | Result |
| --- | ---: |
| Held-out episodes | 134 |
| H-only bridge MAE | 7.724 |
| GT E/I bridge MAE | 12.250 |
| Cm1 predicted E/I bridge MAE | 7.732 |
| Cm8 predicted E/I bridge MAE | 7.727 |
| Cm8 minus Cm1 | -0.005 MAE |

The consequence predictor itself learned a modest signal. Cm1 effect RMSE was
3.23% below its train-mean baseline and Cm8 was 2.09% below; interaction was
0.71% and 0.60% below, respectively. Test normalized loss decreased from
0.713 to 0.666 for Cm1 and ended at 0.677 for Cm8 (best 0.676), so this is not
the previous 8-epoch flat-decoder underfit.

The long-return value gate failed: GT eight-step consequences did not improve
the H-only bridge. A follow-up used the exact discounted local return over the
same eight decision rows. There, GT E/I did carry information: H-only MAE was
`1.229`, GT-E/I MAE was `0.883` (28.1% improvement). With the same local GT
bridge frozen, the predictor outputs scored `1.235` for Cm1 and `1.248` for
Cm8, preserving `-2.0%` and `-5.8%` of the GT gain. Thus K=8 consequence is
locally informative, but neither current-action nor action-chunk prediction
recovers that information.

Artifacts:

- `tmp/P-20261004-gate2-action-chunk-k8-ns3.json`
- `tmp/P-20261004-gate2-action-chunk-k8-ns3.pt`
- implementation commit `153beaa`

Additional artifact:

- `tmp/P-20261004-gate2-chunk-local-value-k8-ns3-with-cm.json`

## Decision

**K=8 oracle mechanism: PROMISING locally; current Cm predictability:
UNPROMISING.** The Transformer removes the flat-output issue and GT E/I
explains a short local return, but neither Cm1 nor Cm8 preserves it. Stop here
instead of running n1 or sweeping K. The next design question is how to train a
predictor against the value-relevant consequence rather than adding more
action/chunk controls. This does not justify online Cm, PPO, or distillation.
