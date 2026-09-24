# P-20260924-crossobject-contact-future-cm

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

The self-trained policy's apple contact fraction is ~0.27 versus ~0.75
for the read-only diagnostic actor; short action pulses and even ten-step
forced options did not produce lasting lift. Can an action-conditioned,
geometry-aware Cm at least predict **future contact persistence under the
self-trained policy** on an unseen object? If not, using it as an actor
representation/critic target is premature.

## Minimal protocol and decision

Use the existing self-trained train3 e320 rollout transitions for
airplane/mug/toothpaste fitting and the apple e320 rollout only for one
exploratory held-out evaluation. Inputs are pre-action q, q-velocity,
object state and current actor action; target is mean hand/object contact
over the next 20 executed simulator steps, restricted to the first
episode and complete 20-step followup. Randomly sample at most 500 rows
per train object/current-contact stratum, and at most 500 apple rows per
stratum. Calibrate action-to-hand flow and all feature normalization on
train objects only. Train a shared six-region geometric Cm with action
flow, same architecture with action flow zeroed, and raw state+action MLP,
same 400-update schedule. Evaluate future-contact RMSE on apple overall
and in current-contact states.

Continue to a small matched PPO auxiliary Probe only if geometric
action-aware RMSE on current-contact apple states is >=10% lower than
**both** action-blind geometry and raw-state+action, with a nontrivial
action-conditioned score spread. Otherwise mark this target/representation
`UNPROMISING` and do not infer causal action utility from observational
rollouts. No apple row may enter fitting, calibration, normalization or
model selection. Apple has already been used in route exploration;
formal validation must use fresh object identities.

CPU only, <=45 minutes, <=20 MB output. Stop on source/provenance drift,
first-episode leakage, non-finite values or missing contact strata.

## Result

Pending.
