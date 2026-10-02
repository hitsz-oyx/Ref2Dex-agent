# Joint option opportunity: matching failed, interpretation remains unclear

Design01f81c8/fallback clarification before data; execution7ad72a8. Fresh601,
768trajectories/202ticks. COMPLETED / UNCLEAR,156.338s/249505391bytes. Native
reset, private pairing/draws, full P0 NumPy forward, held-option timing,
bounded projection, actual PD and full-mesh105 labels independently replay.
P0forward error1.8734e-7, target2.3842e-7, clearance2.3413e-7m. All protected
inputs unchanged; all owned phases terminal.

| Motion | P0 | Duplicate P0 | Option+ | Option- | Retrospective P0/option OR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, /64 | 0 | 0 | 0 | 0 | 0 |
| 1, /64 | 52 | 56 | 33 | 37 | 59 |
| 2, /64 | 11 | 11 | 14 | 15 | 29 |
| Pooled, /192 | 63 | 67 | 47 | 52 | 88 |

The numerical opportunity gates pass, including49option-disagreement groups
and25new successes relative to anchor labels. Those counts are descriptive
only: initialq/dq/object states match EXACTLY, but the pre-intervention physical
prefix fails ALLfive state/velocity/rotation gates. Baseline label agreement
186/192(96.875%) passes, which does not override the prefix failure. Prefix
object positions differ18.833mm, robot XYZ4.404mm, joint angles.164rad. Do not
interpret88/192 as a conditional action-value ceiling, a deployable selector,
Cm utility or policy-training benefit. No thresholds/subgroups were changed.

Saved-trace reproduction locates divergence after the FIRST physical tick,
before candidate actions; initial context and first goals/actions are identical.
Original diagnosticr1 preserved; r2 excludes intervention-tick commands from
the prefix while retaining decision PREstates. Prior table XY differs15.26µm.

Property-only engineering605 completes93.495s/20607462bytes, ZERO physics or
model forwards. All768 paired SDK DOF, hand/object body and material properties
are identical; per-actor property variation is not the explanation in this
snapshot. Origins differ across768environments, maximum280m. A subsequent
engineering check2c902d9 changes only envSpacing5->0 on the same native/private
seed601;56P0-only ticks, no options/task outcomes. It completes118.606s/
72287577bytes, verifies origins0 and P0/PD forward1.9404e-7/1.1921e-7, but
matching still fails and position divergence reaches1.451m. Co-location is NOT
adopted; it may affect scene isolation/solver behavior. This failed layout test
does not identify the original microscopic cause or rule out origin effects.

Stop layout/threshold scans and any same-state oracle interpretation. Actual
observed transitions/returns remain valid records of their executed environments,
but future inference must use actual decision states and randomized executed
comparisons instead of assuming exact cross-environment replay. Next useful
direction is a model/value chain on joint options followed by prospective
evaluation of genuinely trained actors under matched randomized conditions.
Generic model-based offline RL is not novel by itself; journal readiness and
positive matched Cm policy-training utility remain unproved.
