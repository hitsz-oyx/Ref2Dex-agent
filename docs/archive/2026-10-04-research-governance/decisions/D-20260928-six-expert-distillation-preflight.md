# D-20260928: six-expert distillation preflight

## Decision

Record the current distillation route as `UNCLEAR` and spend the next bounded
step on a provenance-complete six-role transition collection. Do not start
student GPU training, PPO, or online aggregation from the existing exports.

## Evidence

The CPU probe `P-20260928-six-expert-distillation-r2` completed on 16,384 rows
using a 49-dimensional `q + dof_vel + object_state` input. Held-out action MSE
was `0.0025601456873118877` and cosine similarity was `0.9141373038291931`.
These are wiring metrics only. The preflight found no full 1442-dimensional
observation aligned with actions, usable action/state exports for only two of
the six roles, and no deployment normalization contract.

## Chosen action and cost

`agent_rl` receives one bounded collection task: two episode-disjoint splits,
all six teacher actions and checkpoint provenance, the frozen C1 router label,
the pre-action observation, object pose at `t` and `t+1` in the object-local
frame, and five contact targets. The collection budget is at most one GPU,
20 minutes, and 1 GiB. It must stop on missing arms, hash or propensity drift,
split overlap, missing targets, or budget overflow.

## Stop and next step

If the contract fails, preserve the `UNCLEAR` label and do not fit or train;
return to a higher-level route review. If it passes, run only the predeclared
CPU calibration and then let root select a separate short student Probe with
matched C1-router control and fixed-label placebo. No formal Cm or grasp claim
follows from this collection or the reduced-state probe.
