# Decision Memo: object-projected Cm MVE interface

Date: 2026-10-03
Decision type: bounded decision-interface Probe

## Question

Can the model-based short rollout become useful by letting Cm predict only the
object translation/velocity/contact consequences while retaining the current
hand configuration and object orientation for the value continuation?  This
tests whether full-state model error, rather than the physical object signal,
caused the earlier MVE failure.

## Evidence

On the pre-existing episode-hash held split (209,788 real transitions), the
decision-time object projection reduced pessimistic MVE RMSE from 26.83 to
26.05 and increased row-level Spearman from 0.589 to 0.605 relative to
direct-Q.  Full predicted hand state remained worse at RMSE 28.31 and episode
Spearman 0.724 versus direct-Q 0.743.  The offline gain is a target-quality
diagnostic, not an action-utility result.

A fresh native five-arm panel then evaluated Cup, direct-Q, object-projected
MVE, the previous Cm MVE, and a uniform random candidate.  It produced 127
complete ten-step windows.  Object-projected MVE changed 25 actions, but its
motion/start cluster-bootstrap difference versus Cup had lower90 bounds of
`-47.16 mm` for last-three-step minimum height, `-0.492` for local reward,
`-0.324` for contact fraction, and `-0.324` for clearance fraction.  The random
arm had 23 windows, so the full support gate was also short by one row.  On
those random rows, object-projected MVE's height/reward Spearman was 0.106/0.443,
below direct-Q's 0.434/0.723.

## Decision

Close this object-projected action-ranking recipe.  Do not lower margins, add
ordinary Cm data, start a short rollout A/B, or train PPO under this contract.
Keep the offline target-quality improvement as a clue for a future critic-only
MVE design; it does not pass the required action ranking, changed-action local
utility, and native support gates.

## Cost and safety

The native run used one idle GPU, 96 environments, a frozen physical checkpoint,
and an 80-second process budget.  The failed launch-directory attempts are
preserved as engineering records; no external project, checkpoint, or other
GPU process was modified.
