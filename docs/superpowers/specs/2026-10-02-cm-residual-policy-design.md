# Cm Residual Policy Probe Design

## Purpose

This route tests whether Cm can improve a strong frozen Cup/rotation controller by
predicting the short-term consequences of a small residual, rather than generating
an entire 18-dimensional action. The first Probe is deliberately limited to a
three-dimensional translation residual expressed in the current object frame. It
does not change the mission claim or start PPO training.

The frozen reference is the HF15 `rotation_cup7` feedback law: the Cup expert's
current XYZ and finger feedback is executed while wrist rotation is anchored at
the trigger observation. A zero residual must reproduce the reference native PD
target exactly. The residual can change only the first three native translation
targets; wrist rotation, finger targets, mimic joints, and the reference feedback
law remain owned by the baseline.

## Data and split

The source collector runs the real DExplore native simulator on airplane only. It
selects first-episode contact states after three consecutive joint-contact proxy
ticks, with both early and already-clear strata capped independently. At each
trigger it samples a zero residual or a bounded nonzero residual and applies that
residual for two native control steps. The baseline feedback then owns the
remaining H10 window. Every source row stores the pre-action history, object pose,
baseline and actual native targets, requested and applied residual, raw forces,
full two-step and H10 outcomes, mesh clearance, episode identity, motion/start
group, and assignment propensity. Partial or reset-contaminated windows are
rejected.

Residuals are sampled in the fixed object-local box
`[-2, 2] mm x [-2, 2] mm x [-3, 3] mm`, with a known zero arm. The object-local
translation is rotated into world coordinates using the trigger quaternion before
being added to the baseline native target. Native joint limits and existing mimic
couplings are checked before execution. Any requested residual that is clipped or
otherwise fails the execution contract is recorded and cannot silently become a
valid scientific row.

Motion/start groups use the existing deterministic hash split: fit buckets `<50`,
calibration buckets `50..69`, and held buckets `>=70`. Held rows are never used for
normalization, model fitting, policy-label selection, or fallback thresholds.

## Cm and policy

Cm is an ensemble of three identical models. Its inputs are current history,
current contact/force and clearance context, object-frame geometry, and a candidate
three-dimensional residual. Its targets are the actual two-step object translation
in the trigger object frame, supported lift, joint-contact retention/contact loss,
and mesh-clearance loss. The model never consumes post-action fields as inputs.

The policy is a separate bounded actor. It emits only a three-dimensional `tanh`
residual in the fixed box; it cannot emit or replace a native action. Fit rows are
used to generate safe candidate residual labels from the Cm ensemble on a fixed
small residual grid. The actor is trained to reproduce those labels and is then
frozen. A matched shuffled route uses the same model and actor topology after
shuffling residual/context association within fit strata.

At inference, the actor's residual is accepted only when input and residual values
are in-distribution, ensemble disagreement is below the fit calibration limit,
predicted lift exceeds the zero-residual reference by the fixed margin, predicted
contact loss is no worse than the reference by 2 percentage points, and predicted
clearance loss is no worse by 2 percentage points. Otherwise the output is exactly
zero and the frozen baseline runs. The native collector logs both the proposed and
executed residual so fallback is auditable.

## Native Probe

Held eligible states are randomly assigned with probability `1/3` to:

* `baseline`: zero residual and the frozen `rotation_cup7` controller;
* `residual`: the frozen Cm-trained bounded residual policy;
* `shuffled`: the matched shuffled policy.

Residual and shuffled owners execute their accepted residual for two steps, then
return to the same baseline feedback law for the remaining H10 steps. The collector
records real post-step state, force, object pose, and mesh clearance. Shadow
recommendations never count as counterfactual outcomes.

The primary outcome is H10 retained supported height relative to trigger height.
Secondary outcomes are two-step local object displacement, last-three-step joint
contact, contact loss, and any mesh-clearance loss. Policy coverage, residual-change
rate, fallback rate, residual norm, saturation, and input/OOD counts are reported
separately.

The predeclared minimum support is 96 complete windows per arm, 24 episodes per
arm, and 8 motion/start groups. The residual arm must change at least 48 windows
across 12 episodes and 8 groups. A Probe is `PROMISING` only if residual minus
baseline retained height is at least 2 mm with positive 90% cluster-bootstrap
lower bounds at both episode and motion/start-group levels, contact-loss and
clearance-loss increases are at most 2 percentage points, and last-three-step
joint contact does not decrease by more than 5 percentage points. The shuffled arm
is a mechanism control; it cannot substitute for the strong baseline comparison.
Insufficient support is `UNCLEAR`, and a supported gate failure is `UNPROMISING`.

## Artifacts and verification

The implementation will add a residual coordinate/target contract, a source
collector, Cm/policy fit and calibration scripts, a native three-arm Probe runner,
an independent replay/statistics audit, and focused tests. Tests must cover object
frame rotation, zero-residual identity, native mimic/limit handling, finite bounded
policy output, uncertainty fallback, and rejection of future-field leakage. A
successful Probe remains an exploratory mechanism result; it does not prove final
stable grasping or Cm policy utility until a later matched policy validation.
