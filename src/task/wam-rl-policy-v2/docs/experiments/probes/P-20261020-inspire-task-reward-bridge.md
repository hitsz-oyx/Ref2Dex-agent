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

The remaining decision is whether the qualified collector should be used for a
bounded policy-improvement probe. The closed-loop BC runs still show zero path
progress, so this card does not admit multi-hour PPO by itself.

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


## Review correction and regression checks (2026-10-20)

The prior 30-control collector runs used distribution means rather than sampled
Gaussian actions. Their log-probability replay/finite-loss results establish only
transport and tensor compatibility; the earlier statement that they qualified
an on-policy PPO collector is withdrawn. They also did not exercise task terminals.

The repair shares a .96 discount between potential shaping and GAE, samples raw
behavior actions before clipping only the native command, and saves both forms.
A bounded single-episode batch stops at its first measured task terminal after
recording that last transition. Other environments retain their actual final
bootstrap; no action is sent after termination. This is not yet a multi-episode
reset/resume training collector. Native episode/tick changes are rejected before
another decision. The host now supplies same-execution object contact force and
actual table pose for the same CAD/net-force support proxy used by admission.

Scheduled WAM goals now come from the identical TaskController decisions used
for task10, including acquisition/recovery/release gates, and geometry is selected
by environment order rather than the 50-object global index. Current421 expansion
explicitly zeros all 265 new columns and preserves shape/normalization/old weights.
The repaired tau layout is v2 and repaired scheduled BC packets are v2; old v1
artifacts stay immutable and cannot silently enter the new route.

126 Task tests pass, including stochastic mean-gradient, warmstart invariance,
450-control acquisition timeout with a second live environment, hidden reset
rejection, nonleading/repeated object binding, support-dependent release, and
archive replay goal/task equality. Planned bounded native checks use run_id
`inspire-measured-interface-fixes-20261020-r1`, frozen Task code and existing
Current421 best; the Scheduled686 fixture is an untrained expansion of that same
parent, only for interface verification. No long BC/PPO run is launched.
