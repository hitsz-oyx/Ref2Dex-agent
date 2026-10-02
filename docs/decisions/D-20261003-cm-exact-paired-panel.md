# Decision Memo: exact same-state Cm action panel

Date: 2026-10-03  
Decision type: bounded physical action-ranking Probe

## Question

Does Cm's frozen physical consequence score rank native candidate actions by a
real local effect when every candidate is executed from the same hot simulator
state?  This separates action-data coverage and simulator pairing noise from
the value-conversion problem seen in the randomized H10 panels.

## Evidence

The existing randomized candidate panel has adequate marginal support, but its
positive offline selector signal did not survive fresh native validation.  The
decision-interface residual fit cannot be evaluated because its held-out
random arm has only five rows.  Both failures leave open whether the remaining
problem is cross-state confounding or Cm-to-value conversion.

## Fixed design

At 24 eligible contact/lift states, freeze the six expert actions, base-hold,
and fixed-Cup candidate.  From one live state, run each of the eight candidates
for one native physics tick using the validated `paired_sim_step` restore
contract; fixed Cup is the within-state baseline.  Record actual object-height
effect, model score/std/risk, and repeatability errors.  No future state,
post-action label, training, threshold search, or policy update is available to
the selector.

The Probe passes only if all candidates have complete paired support, Cm's top
candidate differs from Cup in at least 25% of states, its actual top-vs-Cup
height effect has a positive motion/start cluster-bootstrap 90% lower bound,
and the per-state score/effect rank correlation is positive.  A pass permits a
fresh short-rollout A/B; a failure closes this exact action-ranking interface
without PPO or ordinary Cm-data expansion.

The smoke reached one eligible state and failed the restore contract before
any non-Cup effect was observable: root/DOF restored exactly, but rigid-body
state differed by 8.4–40.5 units.  This is an engineering-boundary `UNCLEAR`
result.  The tolerance remains fixed and the exact hot-state route is closed;
the existing randomized native evidence remains the utility boundary.

## Cost and safety

One GPU, at most 24 states and roughly 8 one-step panels per state, with a
300-second process budget and an owned output directory.  Existing checkpoints
and external projects remain read-only.  The paired helper must reject any
state whose same-action replay exceeds its fixed tolerances.
