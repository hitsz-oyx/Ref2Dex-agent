# D-20260928: from-scratch Cm teacher arbitration

## Decision

Use a new, from-scratch **physical transition envelope** only to select the
teacher action label for six-expert to one-student offline distillation.  The
model is not called online and does not modify an executed action.

The model receives pre-action observation, one candidate expert action, and a
candidate id.  It predicts the signed one-step object displacement in the
object-local frame plus quantiles for five-step contact retention.  The
teacher label is the candidate with the highest certified displacement along
the object lift axis among candidates whose contact lower confidence bound
passes the fixed threshold.  If none passes, the frozen C1 observation-router
teacher label is used.  A static `source_e260` fallback is forbidden because
it would change the matched Cm-off teacher and confound the distillation
comparison.

## Why this is a distinct route

* HF01 and HF05 ranked or gated an action at execution time.  This route only
  chooses an offline supervision label before student training.
* A critic or residual model would estimate value or alter an action.  This
  model predicts physical transition effects and leaves the action unchanged.
* The `docs/ref.md` substrate supplies six teacher actions and a student
  architecture, but no Cm target or Cm checkpoint.  The Cm-on and Cm-off
  students use the same states, candidate actions, architecture, optimizer,
  steps, and seeds; only the offline teacher label differs.  Cm-off is always
  the frozen observation-router label, while Cm-on substitutes only a
  certified Cm label and falls back to that same router label.

This is a mechanism design, not evidence of policy utility.

## Identifiability and stop condition

Identifiability requires six-arm randomized candidate actions at matched
pre-action states, propensity `1/6`, and episode-disjoint fit/holdout splits.
The old HF03/HF04 wrist-z records do not provide this support for the six
expert actions.  No new transitions are collected in this task.  A future
execution must stop before fitting if any arm is absent, a fit/holdout
episode overlaps, or only the old wrist-z support is available.

## Artifacts

* Contract: `src/task/CmResidual/scratch_teacher_arbitration_contract.py`
* Design config: `src/task/CmResidual/configs/cm_scratch_teacher_arbitration_v1.json`
* CPU smoke: schema validation, leakage rejection, six-arm support, and
  deterministic arbitration passed locally; no model update or simulator was
  run.

## Deferred execution

`agent_rl` may prepare the randomized six-arm transition manifest and run the
CPU calibration gate only after root accepts this design.  A later matched
offline distillation Probe must include `cm_on`, fixed-route `cm_off`, and a
fixed-seed candidate-label placebo.  Any held-lift validation requires a
separate Decision Checkpoint and is outside this task.
