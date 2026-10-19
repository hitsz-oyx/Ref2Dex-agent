# P-20261020-inspire-scheduled686-closed-loop

## Decision

Check whether the Scheduled686 policy can consume online spatial tau from the
frozen v4 WAM runtime while preserving the measured state, task scheduler and
native action contracts.

## Protocol

- `task`: `wam-rl-policy-v2`
- `experiment_id`: `P-20261020-inspire-scheduled686-closed-loop`
- `run_id`: `inspire-scheduled686-closed-loop-20261020-r1`
- `branch`: `main`
- `git_commit`: recorded in the run manifest
- checkpoint: Scheduled686 best at step 1000
- WAM: frozen v4 `step80000-model.pt`, read-only geometry cache binding
- WAM mode: spatial goal tau only; event conditioning is false
- two fixed audited TRAIN motions: airplane and alarmclock
- 60 controls, GPU6 policy/WAM and GPU7 native host, JSON RPC, `--extended-state`
- output: `outputs/wam-rl-policy-v2/inspire-scheduled686-closed-loop-20261020-r1/`

## Result (2026-10-20)

The run completed all 60 controls and 120 valid-force observations. The first
three history ticks per environment correctly emitted invalid zero tau; the
remaining 116 observations passed the online WAM workspace gate. Native checks
reported physical248 error `2.98e-7`, canonical-to-PD error `0`, and zero
cross-episode action mismatches. This validates the spatial tau transport and
the Scheduled686 input boundary only. It does not establish task success,
event-conditioned behavior, or a tau performance gain over Current421.
