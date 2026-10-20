# P-20261020-inspire-bc-action-scale

## Decision

Determine whether the zero-progress closed-loop BC runs are caused by action
collapse or by compounding/generalization under physical feedback.

## Protocol

- `task`: `wam-rl-policy-v3`
- `experiment_id`: `P-20261020-inspire-bc-action-scale`
- `run_id`: `inspire-bc-action-scale-20261020-r1`
- split: the 5,693 paired active-and-valid DEV rows
- checkpoints: Current421 best and Scheduled686 best
- output: `outputs/wam-rl-policy-v2/inspire-bc-action-scale-20261020-r1/metrics.json`

## Result (2026-10-20)

Teacher action absolute mean was 0.19095, Current421 was 0.18532 and
Scheduled686 was 0.18784. Mean action-vector norms were 0.9856, 0.9607 and
0.9713 respectively. In phase ACQUIRE, the corresponding absolute means were
0.19154, 0.18209 and 0.18513; in TRACK they were 0.19015, 0.18963 and
0.19146. The policies therefore emit nontrivial commands with teacher-scale
magnitude. The 300-control zero-progress result is a closed-loop
generalization/conditioning problem, not a simple output collapse; PPO must
first receive an explicit measured TaskReward bridge and bounded smoke.
