---
schema: ref2dex.probe.v2
probe_id: P-20261016-v2-base-residual-adapter
experiment_id: P-20261016-v2-base-residual-adapter
date: 2026-10-16
task: trajectory-policy
branch: main
git_commit: 216ecf1c201a0b36c529192bebefc561f03d2d66
claim_id: C3
hypothesis_family: HF-v2-base-residual-interface
probe_index_in_family: 1
seed_pool: probe
seeds: []
decision_changed_if_positive: run the bounded v2-base zero-residual adapter smoke, then consider a residual probe
decision_changed_if_negative: retain v2 BC as a standalone nominal policy and do not launch PPO through the old residual server
status: UNCLEAR
run_id: v2-base-residual-adapter-20261016-r1
---

# v2 BC as the nominal policy for trajectory-policy residual control

## Motivation and decision note

The current v2 BC is the best candidate nominal policy because it uses the
same-tick 721D closed-loop observation and its own native execution bridge. The
trajectory-policy residual route already has a measured-state residual actor,
but it was written around an older WAM plus Chunk-R base. This is a Decision
Probe: determine whether a v2 checkpoint can be inserted without silently
changing action units, clocks, or the zero-residual behavior. A positive result
only authorizes an engineering adapter smoke; it does not authorize PPO or a
physical claim.

## Read-only contract audit

The v2 bridge (`src/task/ptv3-shared-wam/tools/run/policy_server.py`,
`--model-version v2 --shared-actor`) requires the live 87D native state,
same-tick `native_current_obs721`, H4 object/hand history and a planned object
goal. It returns an 8×15D geometry-lift action chunk plus the 24-step native trajectory.
The first command is obtained through v2's `freeze_chunk` and
`consume_target` path, which applies the v2 native controller and preserves the
chunk clock.

The existing residual path
(`src/task/trajectory-policy/tools/run/wam_native_server.py` and
`src/task/trajectory-policy/src/trajectory_policy/residual_ppo.py`) expects the
old WAM/Chunk-R server to return one 18D `base_action` per query. Its actor
receives a fixed 393D observation and emits a 12D residual that is added with
`alpha=0.15` on the old active native coordinates. It has no 721D input or v2
geometry-lift/chunk contract. Therefore a direct checkpoint swap is rejected.

The machine-readable audit is
`outputs/trajectory-policy/v2-base-residual-adapter-20261016-r1/compatibility.json`.

## Minimum next probe

Implement a task-owned adapter only after the four current action-horizon arms
finish. At each live tick it should query v2, decode the first target, and make
the residual coordinate system explicit. The first engineering test must use a
zero residual, compare the adapter's 18D command and PD target with the v2
native bridge, verify same-tick 721D admission and chunk timing, and replay the
fixed airplane/sphere/mug panel. Any nonzero discrepancy closes the adapter
until fixed. Only a clean zero-residual audit can justify a small residual
probe; it does not justify PPO by itself.

## Current conclusion

The architecture is feasible with an adapter, but the existing residual route
cannot consume a v2 checkpoint directly. The static audit is `UNCLEAR` rather
than a scientific success or failure because no adapter code or physical
zero-residual run has yet been executed. Existing residual integration evidence
remains bounded and negative on clipping/placement; it does not transfer to v2.

## Post-panel zero-residual contract smoke

After the four action-horizon arms completed, the successful v2 native panels
were replayed through the old residual module's `apply_residual(base, 0)`
function. Across 10 successful airplane/sphere/mug panel roles, the 18D base
command after a zero residual matched v2's applied command exactly (maximum
absolute difference `0.0`); the recorded same-tick current input remained
`(controls, 3, 721)` and the one-step chunk clock was valid. The machine-readable
engineering result is
`outputs/trajectory-policy/v2-base-residual-adapter-20261016-r1/zero_residual_audit.json`.

This closes only the coordinate/clipping part of the adapter contract. It is
not an old-server integration: no residual IPC server was modified, and the
zero-residual replay does not establish placement or PPO utility. The next
implementation should expose this exact 18D applied command as `base_action`,
retain v2's same-tick 721D admission and chunk state, and then run a fresh
zero-residual fixed panel before any nonzero residual training.
