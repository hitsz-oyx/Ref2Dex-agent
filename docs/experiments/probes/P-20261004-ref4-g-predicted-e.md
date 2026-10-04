# P-20261004 Ref4 predicted-E bridge

**Type.** Decision Probe follow-up.

**Question.** Can a small deployable-input predictor preserve the E-to-G signal
seen with ground-truth E, without training an I predictor? This follows the
previous Ref4 bridge result, where GT E had a directional benefit and the
current GT I sequence showed no independent gain after E.

**Protocol.** The probe reuses `tmp/e260_all4_h16.pt` and exactly the previous
Ref4 episode split: seed `20261004`, group key
`(source_namespace, source_run, episode_id)`, 22 held-out groups, and the first
`K=8` effect steps. The source file has one audited namespace but no explicit
`source_namespace` tensor, so the script records namespace `0` for all rows;
`source_run` and episode identifiers remain in the split key. The target G is
the stored exact Monte-Carlo return-to-go.

First, a GRU/MLP predictor is trained on training rows:

```
(H_t, A_t) -> E_hat_{t+1:t+8}
```

Then a value bridge is trained on GT E in the training rows and evaluated on
held-out rows using H alone, GT E, or predicted E. I is not predicted or
passed to any arm. The predictor uses 8 epochs and the bridges 16 epochs on
GPU 7. This is a probe, not a policy or formal Gate-2 validation.

**Results.**

| Bridge input | Episode-balanced test MAE | Relative to H |
| --- | ---: | ---: |
| H | 22.716 | — |
| H + E_GT | 21.376 | 5.9% improvement |
| H + E_hat | 22.681 | 0.15% improvement |

The effect predictor itself did not beat its train-mean baseline: test RMSE
`2.290` versus `2.211` (`-3.6%`). The GT-E bridge improvement had a descriptive
episode bootstrap delta CI95 of `[-0.358, 4.292]`; the predicted-E bridge delta
CI95 was `[-0.318, 0.405]`. Thus the predicted E path recovered essentially
none of the available GT-E gain on this split.

**Decision.** `UNPROMISING` for this short `(H,A)->E_1:8` predictor as a
deployable G input. Keep the GT bridge as evidence that E may be useful, but do
not add predicted E to the main route or open an I predictor/online training
loop from this result. A future E route would need a better temporal decoder or
lower-dimensional target and a predeclared probe; this card does not justify a
capacity sweep.

**Artifacts.**

- Implementation: `scripts/probe_ref4_g_predicted_e.py`
- Result: `tmp/P-20261004-ref4-g-predicted-e.json`
- Wiring smoke: `tmp/P-20261004-ref4-g-predicted-e-smoke2.json`

Validation performed: Python bytecode compilation, one-epoch GPU smoke, and
the fixed 8/16-epoch GPU probe. User documentation migration files and
`AGENTS.md`/`docs/CAMPAIGN.md` were not modified by this route.
