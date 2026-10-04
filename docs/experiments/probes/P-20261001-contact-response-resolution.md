# P-20261001-contact-response-resolution

Status: designed; exploratory mechanism experiment, not policy Validation.

## Question and decision

Can a single wrist-z intervention produce an object response distinguishable
from fresh-simulator replay noise, and over which short horizons? If yes,
collect differential physical targets for a contact-conditioned Cm; if no,
do not train a counterfactual model on unresolved differences.

Latest user authorization explicitly permits autonomous idea extraction,
experiments and paper writing and supersedes previous local route freezes.
Historical negative results remain unchanged. Resource and data protection
limits remain respected. No new agent/session is created.

## Design frozen before execution

- Four reference panels: training seeds 286/287, evaluation seeds 288/289;
  each has 96 environments and three airplane motions, 32 per motion.
- Freeze existing e420 plain-off actors, snapshots and normalized applied
  action/RNG traces from completed HD02 r3. No training or actor modification.
- In a fresh simulator, use the same complete cold-state restoration and
  physical-property fingerprint check as the previous evaluator. Replay
  reference normalized actions and per-step RNG; do not restore warm solver
  state. Measure physical state immediately before every executed action.
- Per-environment trigger is one action after the first reference event in
  which both native hand/object contact-force proxies exceed their thresholds.
  This retrospective schedule is a diagnostic oracle, not a deployment
  detector. The proxies do not establish pairwise hand-object contact.
- Four runs per panel: zero_a, zero_b, plus, minus. Plus/minus change only
  normalized wrist-z action by +/-0.01 for one control step. Native wrist
  translation scale is 1, so the requested PD target offset is +/-1 cm.
  Clipping or any pre-window termination invalidates the panel.
- Outcomes at 1,2,5,10,20,30 control steps: world object position response,
  linear/angular velocity and contact proxies. Responses subtract each run's
  actual pre-intervention state. Also report pre-intervention mismatch; equal
  cold snapshots do not imply equal warm hidden solver state at the trigger.
- Primary screen: at horizons 5 AND 10, pooled RMS plus-minus position contrast
  >=2x RMS zero_a-zero_b contrast and signed mean z contrast >0 in every panel.
  A failed screen is UNPROMISING for this pulse/schedule; no threshold tuning.
  SNR here is a descriptive contrast ratio, not calibrated uncertainty or
  a causal proof. Baseline repeats diagnose process variation, not independent
  model prediction error. Episode-level intervals condition on the four panels
  and do not claim training-seed/generalization evidence.
- Admission: GPU4 only if idle, 2 CPU threads, <=1800 seconds, <=1 GiB new
  artifacts. Stop on source/input drift, nonfinite tensors, unexpected terminal,
  physical property mismatch, occupied GPU or budget exhaustion. Retain failures.

## Motivation and novelty boundary

The existing true closed-loop repeat has 34 versus 35 successes but 63/384
success-label disagreements. This motivates a physical short-horizon diagnostic
before another long-return controller experiment. Particle-based cross-hand
models, value-aware models and action-discriminative models already exist;
none is claimed as new here. A possible contribution is a contact-conditioned
measurement and learning protocol for differential action effects with explicit
noise resolution and closed-loop evidence. Its novelty and benefit are unproven.

Sources are recorded in ../../archive/2026-10-04-root-research/research/20261001-contact-response-literature.md.
