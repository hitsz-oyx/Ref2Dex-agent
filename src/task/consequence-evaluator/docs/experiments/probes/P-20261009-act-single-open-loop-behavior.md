---
schema: ref2dex.probe.v2
probe_id: P-20261009-act-single-open-loop-behavior
experiment_id: P-20261009-act-single-open-loop-behavior
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 706f0a2
claim_id: C3
hypothesis_family: HF-consequence-act-proposal
probe_index_in_family: 3
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain single-env open-loop24 ACT as a possible frozen-chunk execution container and consider one serial ACT cluster
decision_changed_if_negative: close ACT execution for the current native policy and keep only the reactive behavior baseline
status: UNCLEAR
run_id: gate1-act-single-open-loop-20261009-r2
---

# Can a single native GPU environment execute one frozen ACT chunk?

## Decision Note

The deployment-conditioned proposal is promising in action space, and the
existing four-environment open-loop behavior packet reached `0.7939 m` and
478 held frames. The missing evidence is a single native environment: the
four-environment packet can mix environment-row scheduling with policy
behavior. This bounded screen tests only whether a fresh single environment
can execute a reactive prefix through tick48, generate one 24-step native
chunk, and continue that chunk without observation feedback.

The screen uses the frozen engineering checkpoint from
`act-native-chunk-clean-airplane-base-20261009-r1`, seed282, native GPU PhysX
and GPU tensor pipeline, and 72 steps. It is not a candidate comparison and
cannot satisfy strict same-state Gate1. A positive result only permits one
later serial ACT noise diagnostic; a negative result closes this execution
route without changing Y, the reference bank, or policy weights.

## Predeclared stopping rule

The ACT arm must reach at least `0.20 m` maximum lift in the 72-step screen
and must not terminate early. If it fails either condition, do not run a
serial ACT cluster. Native policy behavior remains an engineering screen, not
a task-success claim.

## Result

The corrected worker completed the native GPU PhysX/GPU-pipeline screen in
`outputs/consequence-evaluator/gate1-act-single-open-loop-20261009-r2/`.
The executed-control contract was exact (`native_action_max_abs_delta=0.0`):
the reactive prefix ran through tick47, one proposal was generated at tick48,
and the 24 controls were then fed open loop with no future observation
feedback. The episode did not terminate early, but the object never lifted in
the 72-step screen (`max_lift_m=0.0`, maximum held frames `0`, terminal height
`-0.001388 m`). This fails the predeclared `0.20 m` behavior threshold.

For a same-command-shape engineering control, a fresh reactive-only 72-step
worker at the same seed/backend (`outputs/consequence-evaluator/gate1-act-single-reactive-20261009-r1/`)
reached `0.2915 m` and held 13 frames. The ACT and reactive packets are fresh
processes rather than a same-state pair: exposed contact buffers diverged from
tick44 and the state/RNG hashes are not equal, while the visible prefix stayed
close through tick47. The control therefore confirms that the frozen chunk
screen is execution-sensitive, but cannot isolate the chunk effect from the
known GPU contact/cache divergence.

The first attempted run in `...-r1/` stopped before producing a packet because
the new audit compared `(72, 1, 18)` requested controls against `(72, 18)`
captured controls. That runner shape bug was fixed in `706f0a2`; the r1
artifact is retained as an implementation failure and is not evidence about
ACT behavior.

The predeclared screen gate is consequently closed for this single-environment
execution contract, and the serial ACT cluster is not run. This remains
`UNCLEAR` engineering evidence rather than a policy, value, or Gate1 scientific
negative result; the reactive native behavior container and the
physical-reference/value route remain unchanged.
