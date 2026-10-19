# P-20261020-inspire-task-reward-bridge

## Decision

Determine whether the measured Current421 and Scheduled686 online executors can
feed the task-local `TaskReward` from causal measured transitions while keeping
the native action and state contracts intact. This is a PPO bridge prerequisite,
not a policy-performance experiment.

## Protocol

- `task`: `wam-rl-policy-v2`
- `experiment_id`: `P-20261020-inspire-task-reward-bridge`
- `run_id`: `inspire-current421-reward-smoke-20261020-r1` and `inspire-scheduled686-reward-smoke-20261020-r1`
- `branch`: `main`
- `git_commit`: `aa3b7e8893177d032264a95435b1009ba02e52e4`
- Current421 best checkpoint at step 3000; Scheduled686 best checkpoint at step 1000
- two fixed audited TRAIN motions: airplane and alarmclock
- 60 controls, GPU6 policy/WAM and GPU7 native host, JSON RPC, `--extended-state`
- causal reward uses previous/current measured pose, hand, object points, object
  velocity, contact force, task decision, and previous action; no native reference
  reward, observation, or termination is consumed
- outputs:
  `outputs/wam-rl-policy-v2/inspire-current421-reward-smoke-20261020-r1/`
  and `outputs/wam-rl-policy-v2/inspire-scheduled686-reward-smoke-20261020-r1/`

## Result (2026-10-20)

Both runs completed all 60 controls. Current421 produced reward totals
`[2.721500, 2.336822]`; Scheduled686 produced `[2.368382, 2.720846]`.
The final reward parts were finite and had positive shaping with only the small
time and effort penalties; neither run emitted a terminal or task-success
claim. The Scheduled686 arm consumed 116 valid tau observations after startup
and 120 valid force observations.

Both native hosts passed physical248 maximum error `2.98e-7`, canonical-to-PD
error `0`, and zero cross-episode action mismatches. This qualifies the
measured `TaskController -> TaskReward` adapter for a bounded PPO collector
smoke. It does not establish reward calibration, task completion, critic quality,
or a PPO improvement, and it does not justify a long PPO run yet.

## Next decision

Run a small actor/critic and collector shape smoke against the 421/686
observation contracts and the recorded reward transition, checking finite
advantages, synchronized parameter shapes, behavior log-prob consistency, and
bootstrap at the measured terminal boundary before any multi-hour PPO training.
