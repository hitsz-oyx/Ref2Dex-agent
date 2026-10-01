# Current-policy collector repair and factual V diagnostic

**Completed by root under explicit direct-takeover authorization.** Scientific
label: `UNCLEAR`. HF08 remains `PAUSED`, its utility slot remains used at 1/1,
and Cm policy utility remains `OPEN`.

## Collector repair and verification

The old collector passed raw network `logstd` to a Gaussian as its standard
deviation. The repaired collector uses the full native model forward once,
preserving the actual player preprocessing, action clipping, RNG consumption
and same-forward PPO critic. Critic-only inspection does not draw another action.

The real DExplore checkpoint uses external `CommonPlayer._preproc_obs`
normalization with model-local normalization disabled. The earlier assertion
that the old path necessarily skipped observation normalization was incorrect.
Both actual e420 checkpoints now pass native CommonPlayer action/value/RNG
equivalence checks with saved external RunningMeanStd.

Real collection also exposed two export defects: context was hardcoded as 605
instead of the saved model's 435, and a complete-episode validator was invoked
at an arbitrary 32768-row partial export. The corrected bounded collector exports
after all first episodes complete. A 400-step / 38400-row mock reproduces the old
partial-export failure and passes after repair. Mean lift is recorded as a
trajectory average, alongside maximum lift.

Full focused verification: **24 passed**. After adding per-checkpoint and
lambda-vs-MC reporting, the **11 diagnostic tests passed** and the actual
107584-row diagnostic completed successfully.

## Actual data and resource evidence

| Saved policy | Rows | Complete first episodes | Primary success | Subsequent drop |
|---|---:|---:|---:|---:|
| s286 e420 | 53792 | 96 | 6 | 6 |
| s287 e420 | 53792 | 96 | 1 | 1 |
| Total | 107584 | 192 | 7 | 7 |

Each panel has 32 episodes for each of the three fixed airplane motions,
frame 0, seed290, native Gaussian actions without extra noise. No teacher,
training or parameter update was performed. Episodes excludes zero rows;
all192 native primary/drop labels match reconstruction. Next critic values
match same-episode successor values; terminal bootstrap is zero. Checkpoint,
environment, training and source-runner hashes remain unchanged.

r4 failed before export after76.468 seconds. Repaired r5 completed after203.002
seconds; combined runtime279.469 seconds, under the takeover1200-second ceiling.
Collection/cache/log artifacts total455907862 bytes before the small diagnostic
and reporting additions, under2GiB. CPU diagnostic used2 threads and32.087
seconds. GPU4 UUID `GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307` had no existing
compute owner at admission and no compute owner after collection. Old runs are
preserved; this accounting does not erase their costs.

## Factual diagnostic results

All values below are row-weighted RMSE in raw reward-return units. GAE means
the current frozen PPO critic's diagnostic lambda target, gamma=.99, tau=.95,
with recursion cut at32-step global rollout boundaries; it is not a recovered
historical training label. MC is the realized full-episode discounted return.

| Error | s286 | s287 | Combined |
|---|---:|---:|---:|
| Saved physical V vs frozen GAE | 28.23 | 22.28 | 25.43 |
| PPO critic vs frozen GAE | 20.56 | 12.94 | 17.17 |
| Saved physical V vs complete MC | 148.47 | 48.83 | 110.52 |
| PPO critic vs complete MC | 142.11 | 46.07 | 105.64 |
| Frozen GAE vs complete MC | 128.52 | 40.81 | 95.35 |

Saved V has greater error than the PPO critic against frozen GAE in both
panels. Frozen GAE and realized MC also differ substantially. On the seven
primary-success episodes, saved V's row-weighted MC error is549.29 and mean
prediction-minus-return is-391.36; failure episodes have MC error38.37. This
rare high-return stratum strongly affects the pooled MC error. The s287 success
stratum has only one episode and cannot support a calibration conclusion.

**Decision:** keep both factual V fit and target/return disagreement as active
uncertainties. These observations do not support calling V sufficiently fitted
or calibrated, and do not establish that increasing V updates alone would fix
HF08. Any next experiment should distinguish those uncertainties before another
policy-training sweep.

This frame-0 Gaussian diagnostic panel is not the full training reset distribution
or the original deterministic actor-only utility evaluation. One realized MC
trajectory is stochastic, so the error does not prove conditional-expectation
miscalibration or bootstrap bias. No candidate-action ranking, convergence,
policy-utility or formal causal conclusion is claimed.

## Reproduction and artifacts

Code is on `agent/current-policy-value-repair`: commits `884e25c` and `56b898e`.
The pinned checkpoints and all inputs are listed with hashes in the
[JSON report](CURRENT_POLICY_VALUE_DIAGNOSTIC_DIRECT_20261001.json).

From `third_party/DExplore`, use graspenv and absolute collector paths:

```text
collect_current_policy_value.py --collect --checkpoint-seed 286 --run-root <new-task-output-root> --gpu-index <idle-physical-index>
collect_current_policy_value.py --collect --checkpoint-seed 287 --run-root <same-root> --gpu-index <same-index>
```

The exact bounded launcher, both process commands, GPU admissions, raw results,
shards and diagnostic command are preserved under:

```text
src/task/CmResidual/research/physical_value/output/P-20261001-current-policy-value-diagnostic-r5/
src/task/CmResidual/research/physical_value/output/P-20261001-current-policy-value-diagnostic-r5-support/
```

The report records SHA256 of the raw diagnostic and launcher. Large data and
checkpoints remain local and are not committed.
