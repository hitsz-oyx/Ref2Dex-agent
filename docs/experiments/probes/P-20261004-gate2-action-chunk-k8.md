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

## Value-aware follow-up

Because the exact K=8 local return is predictable from GT E/I but neither Cm1
nor Cm8 preserved that signal, a single diagnostic value-aware probe added a
training-only local-return head to the shared K=8 Transformer decoder. The
fixed auxiliary weight was `lambda=0.5`; at evaluation the auxiliary head was
discarded and only predicted E/I was passed to the same frozen GT bridge.

| Quantity | Value |
| --- | ---: |
| H-only local bridge MAE | 1.2286 |
| GT E/I local bridge MAE | 0.8833 |
| Value-aware Cm1 predicted E/I bridge MAE | 1.2369 |
| Value-aware Cm8 predicted E/I bridge MAE | 1.2516 |
| Cm1/Cm8 local-value-head test MAE | 1.2305 / 1.2200 |

The auxiliary head learned a local-return signal, but that signal did not
survive through the predicted E/I channel. The value-aware predictor is thus
`UNPROMISING`; it does not justify sweeping auxiliary weights or adding online
training. The predictor artifact is
`tmp/P-20261004-gate2-value-aware-chunk-k8-ns3.json`.

## Decision

**K=8 oracle mechanism: PROMISING locally; current Cm predictability:
UNPROMISING.** The Transformer removes the flat-output issue and GT E/I
explains a short local return, but neither ordinary MSE nor the value-aware
auxiliary objective makes Cm1/Cm8 preserve it. Do not run n1, sweep auxiliary
weights, or start online Cm/PPO/distillation. The remaining decision is whether
the consequence horizon itself must be shortened; any follow-up should be one
minimal horizon test with a predeclared target, rather than another broad
architecture or hyperparameter sweep.

## Pre-registered short-horizon decision probe

The next and only planned follow-up is a K=1 diagnostic on the same n3
namespace holdout. It predicts only `(E,I)_{t+1}` from `(H_t,a_t)` with the
same Transformer family and the plain consequence MSE objective (`lambda=0`).
Cm8 is not opened: K=1 is intended to isolate horizon/decoder aggregation,
using the existing K=8 plain-MSE result as the training-objective reference.
The target is the stored immediate reward on row `t`, with no episode-tail
aggregation. Because this reward contains the recorded shaping components
(base/source, approach, held, progress and stable), a negative result is
interpreted as failure to recover value through this consequence channel, not
as a claim that every reward component is physically determined by E/I.

The decision thresholds are fixed before running it: GT one-step E/I must
reduce the H-only bridge MAE by at least 10% to show one-step value utility;
the predicted-E/I bridge must then be below H-only and retain at least 25% of
the GT gain. If either condition fails, stop Gate 2 consequence prediction
without opening a Cm8/K sweep or online training. A positive result would only
justify designing one short-horizon route; it would not be a formal validation.
