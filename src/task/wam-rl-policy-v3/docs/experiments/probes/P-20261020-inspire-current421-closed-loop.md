# P-20261020-inspire-current421-closed-loop

## Decision

Verify that the measured347 plus task10 Current421 contract can drive the
native Inspire physics host without reference observations, force-field
fabrication, episode mixing, or action-layout corruption.

## Protocol

- `task`: `wam-rl-policy-v3`
- `experiment_id`: `P-20261020-inspire-current421-closed-loop`
- `run_id`: `inspire-current421-closed-loop-20261020-r4`
- `branch`: `main`
- `git_commit`: recorded in the run manifest
- checkpoint: Current421 best at step 3000
- native runtime: pinned DExplore Torch `2.0.1+cu118`, Isaac PhysX GPU pipeline
- two fixed audited TRAIN motions: airplane and alarmclock
- 60 controls, GPU6 policy / GPU7 native host, JSON RPC, `--extended-state`
- output: `outputs/wam-rl-policy-v2/inspire-current421-closed-loop-20261020-r4/`

## Result (2026-10-20)

The run completed all 60 controls and 120 valid-force observations. Native
checks passed with physical248 versus legacy observation maximum error
`2.98e-7`, canonical-to-PD maximum error `0`, and zero cross-episode action
mismatches. The policy server performed no optimizer updates. The run validates
the measured input/action transport and task-controller clock only; it is not a
grasp or task-success result and does not validate Scheduled686 tau inference.

An extended 300-control deterministic replay (`inspire-current421-closed-loop-20261020-r5`)
consumed 600 valid-force observations. Both environments remained in `ACQUIRE`
with path progress `0`, without terminal decisions. Native checks again passed
for all 300 steps. This is a control-behavior blocker for treating Current421 as
a successful executor, not evidence that the observation transport is broken.
