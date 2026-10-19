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

A bounded actor/critic and collector shape smoke was run from commit
`7cf9f41d` as
`outputs/wam-rl-policy-v2/inspire-measured-ppo-shape-smoke-20261020-r2/`.
Both real BC checkpoints loaded at their declared 421/686 dimensions and
12-action boundary. A 6-by-2 causal transition batch produced finite actor and
critic updates; behavior log-prob replay error was `0`, retained KL was about
`1.1e-9`, and both updates accepted one actor step. The fresh critic's initial
bootstrap was finite (zero from its zero-initialized output head). This qualifies
the collector shape for a future online PPO smoke, while reward calibration,
native bootstrap at a real terminal, critic quality, and any PPO improvement
remain unvalidated.

The next blocker is an online collector smoke that records the actual next
measured state and terminal boundary through the native host. Do not start
multi-hour PPO from the offline shape result alone.

That online Current421 collector smoke is now complete at commit `afb8948b`:
`outputs/wam-rl-policy-v2/inspire-current421-ppo-collector-smoke-20261020-r2/`.
The native host supplied 30 controls for each of two environments, producing
30 transitions per environment. Reward totals were `1.174277` and `1.166288`;
the one-update actor/critic report had behavior log-prob error `0`, retained KL
`0.01803` under the `0.02` limit, finite value loss `0.15579`, and an actual
final measured observation as bootstrap source. Native physical248/PD/episode
checks passed again and no task terminal occurred. This qualifies the Current421
online collector contract. The matching Scheduled686 online collector smoke is
complete at commit `fa0a80f9`:
`outputs/wam-rl-policy-v2/inspire-scheduled686-ppo-collector-smoke-20261020-r1/`.
It produced 30 transitions per environment, 56 valid tau observations out of
60 force-valid observations, reward totals `1.166200` and `1.162621`, behavior
log-prob error `0`, finite critic loss `0.15393`, and actual final measured
bootstrap. The actor candidate was rolled back by the KL gate (retained KL `0`),
which is an expected safety outcome for this bounded smoke. Native checks again
passed and no task terminal occurred.

Both 421 and 686 online collectors are now qualified for a future bounded PPO
rollout. This still does not establish reward calibration, task completion, or
PPO improvement, so multi-hour training remains inadmissible without a separate
policy-improvement decision.
