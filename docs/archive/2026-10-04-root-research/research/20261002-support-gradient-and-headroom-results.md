# Gradient control variate and positional headroom: two closed screens

Code ef71c33, `P-20261002-support-gradient-cv-r1` COMPLETED2.120s on admitted
GPU1. Frozen r2 physical30 models, all1536 previously inspected test rows,
true physical105 reward, no model training or new physics. This is explicitly
posthoc/reused-data. Coefficient1 and actor751 are fixed. Exact finite-action
correction is established methodology, not a new theorem.

| Full520-coordinate gradient second moment | Cm | State-only | Global motion/arm |
| --- | --- | --- | --- |
| All1536 | 0.23147464 | 0.28453415 | 0.27423374 |
| Seed525 | 0.22044096 | 0.25984432 | 0.27708365 |
| Seed526 | 0.24250832 | 0.30922398 | 0.27138384 |

Pooled relative improvements18.648%/15.592% fail the fixed20% gate against BOTH
controls. Per-seed lower moments pass. **UNPROMISING**, stop this exact candidate.
Assigned-action probability replay error0. All10protected inputs remain intact.
Independent NumPy reduction of every coordinate/row, seed and motion agrees
within2.22e-16; mean-gradient norm differences also agree. The arbitrary fixed
predictor cancellation identity has two exact-sum unit tests. These do not prove
learned-policy improvement, population equivalence from finite samples, or novel
methodology. Physical forecasting primary remains UNPROMISING.

Next separate structural screen f66aa01, `P-20261002-support-state-dependence-r1`,
uses FIT-only position medians and four bins per motion. FIT physical105 chooses
bin/global arms; TEST525/526 remain reused exploratory data. All3motions retained.
Known propensity1/8 gives fixed-policy IPW estimates, NOT executed success rates.

| Test IPW estimate | Position | Global | Unchanged |
| --- | --- | --- | --- |
| All1536 | 34.375% | 38.021% | 32.813% |
| Seed525 | 33.333% | 40.625% | 32.292% |
| Seed526 | 35.417% | 35.417% | 33.333% |

Position minus global=-3.646pp, versus unchanged+1.563pp. Both frozen gates
fail: **UNPROMISING**. This rejects this exact four-bin recipe, not all possible
state dependence. Unequal arm counts plus Beta smoothing can select a different
arm in all-zero bins; this is retained rather than repaired using test results.
No model-selection scan, extra fitting, selected subgroup or metric substitution.

Three current executable-support candidates fail their gates. End the old
eight-primitive / initial-XY recipe. The next high-level decision changes the
physical decision problem to post-lift disturbance recovery, checking whether
there is physically recoverable state-dependent support before another Cm fit.
Original journal objective, trained-policy causal utility and novelty remain open.
