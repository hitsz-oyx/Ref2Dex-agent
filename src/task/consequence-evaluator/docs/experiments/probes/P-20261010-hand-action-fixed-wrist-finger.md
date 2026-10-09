---
schema: ref2dex.probe.v2
probe_id: P-20261010-hand-action-fixed-wrist-finger
experiment_id: P-20261010-hand-action-fixed-wrist-finger
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 18a726d90531a001795e74b5c2c09afdcd2dab55
claim_id: C3
hypothesis_family: HF-hand-action-retarget
probe_index_in_family: 3
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain the frozen analytic wrist decoder and continue narrow finger/contact command decoding
decision_changed_if_negative: keep the fixed wrist decoder as an offline engineering reference and stop native finger-decoder sweeps on this source
status: UNPROMISING
run_id: ref7_2-hand-action-fixed-wrist-finger-20261010-r2
---

# Does a fixed wrist decoder expose a viable learned finger command branch?

## Decision note

The full v2 contextual R execution and the v3 contact-context offline Spike both
leave the native upper-bound gate failed. The preceding object-relative campaign
did, however, reproduce a reset-calibrated 11-point wrist inverse with train-only
one-step PD statistics. This Probe executes the exact frozen-wrist/learned-v2-
finger combination as the next minimum question from ref7_2: keep wrist decoding
fixed and test whether R's finger commands can preserve source-matched teacher
behavior.

This is one bounded native Probe, not a representation or Cm claim. A positive
result would retain the wrist decoder and justify only a narrower finger/contact
follow-up. A negative result stops this source-level native finger sweep; it does
not refute all hand representations.

## Contract

The model is the frozen v2 contextual checkpoint
`outputs/consequence-evaluator/ref7_2-hand-action-context-fit-20261010-r2/best.pt`.
At every 24-step query, the runner supplies the same privileged source hand
displacement future and current live `q/dq`, current hand/object context, and the
previous actually dispatched hybrid command. The model's normalized action is
used unchanged for finger coordinates `6:18` (including dependent coordinates;
the native Inspire task overwrites their PD targets with its mechanical coupling).

Only wrist coordinates `0:6` are replaced. The frozen decoder reconstructs the
wrist from the six calibrated root points `[0,1,3,5,7,9]` of the current/future
11-point hand geometry, then applies the train-only approach-frame coefficients
from `pd-step-inverse-calibration-20261009-r1/statistics.json`:

```
u_wrist = q_live + a * (q_goal_from_hand - q_live) + b * dq_live
```

The resulting xyz increments and nearest-Euler XYZ rotation increments are sent
as the native first six action coordinates. It reads no future q, dq, force, or
command label. The root template is recalibrated from the current source packet's
reset hand/q, because the old object-relative artifact used a different GPU
source/runtime. The wrist target is explicitly the **current-anchored source
hand displacement** used by the v2 runner, not the absolute-source geometry
target from object-relative r14/r15; their positive results are not transferred
to this contract. The coefficients remain the frozen train-only statistics from
the earlier GPU launches and are therefore an explicit GPU-train to current
CPU-pipeline runtime transfer, not a newly calibrated test-source parameter.
The source hand future remains the GT upper-bound input; this
Probe does not claim deployability.

## Protocol and gate

Use the source-matched CPU packet
`outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`,
the existing four-role native contract, seed 282, GPU2, and a 900-second cap.
The teacher is retained in env0; three retarget roles use the hybrid. The run
must complete all 542 controls with requested=applied and no nonfinite values.
The primary screen is candidate maximum held frames at least 90% of
`max(live teacher held, source teacher held)` with no uncontrolled loss; the
secondary screen is hand coordinate RMSE <40 mm. Wrist and finger clipping are
reported separately. Existing full-v2 source-matched execution is a historical
engineering reference, not a same-process control arm.

Output: `outputs/consequence-evaluator/ref7_2-hand-action-fixed-wrist-finger-20261010-r2/`.

## Implementation audit

The first launch was retained at
`outputs/consequence-evaluator/ref7_2-hand-action-fixed-wrist-finger-20261010-r1/`
but is `INVALID_IMPLEMENTATION`: the runner used query-boundary `q/dq` for all
24 dispatch ticks of the frozen wrist chunk in source code. In this CPU-pipeline
run the logged actions happen to match a live-state recomputation because of
tensor aliasing, but that is not a backend-independent contract. Commit
`18a726d` reads measured live `q/dq` at each dispatch tick and adds fail-closed
train-statistics provenance checks; r2 is the only run eligible for the Probe
conclusion (its packet is numerically identical to r1, but the implementation
contract is explicit).

## r2 result

The corrected r2 completed all 542 controls in 44.4 seconds on GPU2. The live
teacher held 480 frames; the source teacher held 481, so the 90% reference floor
was 432.9 frames. The three fixed-wrist/v2-finger roles held only `4/6/8`
frames. Their hand coordinate RMSE was `7.87/7.82/7.86 mm`, with zero overall,
wrist, and finger clipping; requested and applied controls were exact. An
independent packet audit confirms that dispatched finger coordinates `6:18`
are exactly the v2 output and that all wrist coordinates are replaced by the
analytic decoder.

The wrist tracking therefore remains geometrically accurate, but the learned
finger command branch does not preserve contact in this source-matched native
execution. Probe status is `UNPROMISING` for this one source/seed contract.
Do not interpret it as a formal refutation of every finger representation.

## Limitations / stop rule

This is one source, one seed, one fresh native launch and a privileged GT future
hand input. A negative result cannot distinguish all possible finger command
representations, contact sensing, or source/backend effects. Do not train another
R, change the PD coefficients, add force labels, run a native v3 checkpoint, or
enter H-to-hand, PointWorld, evaluator, selector, MPC, or Cm integration from
this Probe. Preserve the packet and perform only provenance/contract audits if
the launch fails.

## Artifacts

- code commit: `18a726d90531a001795e74b5c2c09afdcd2dab55`
- frozen decoder: `src/task/consequence-evaluator/src/consequence_evaluator/fixed_wrist_decoder.py`
- execution runner: `src/task/consequence-evaluator/tools/run/run_hand_action_retargeter.py`
- train-only coefficients: `outputs/consequence-evaluator/pd-step-inverse-calibration-20261009-r1/statistics.json`
- invalid r1 packet: `outputs/consequence-evaluator/ref7_2-hand-action-fixed-wrist-finger-20261010-r1/`
- valid r2 packet: `outputs/consequence-evaluator/ref7_2-hand-action-fixed-wrist-finger-20261010-r2/`
- contract/unit tests: `217` Task tests passed; fixed-wrist composition audit passed

The earlier source-matched full-v2 execution (`6/9/76` held in a separate
launch) is retained only as a directional engineering reference; it is not a
same-process matched control for r2.
