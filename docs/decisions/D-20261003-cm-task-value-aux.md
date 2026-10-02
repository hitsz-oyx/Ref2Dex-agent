# Decision Memo: Cm task-value auxiliary supervision

Date: 2026-10-03

## Question

After the direct task-value representation was locally better offline but its
full action-teacher policy was worse, can the same frozen representation help
PPO learn a task-relevant shared representation without allowing Cm to replace
or argmax the action?

## Evidence

The frozen Cm physical ensemble plus nonlinear residual MLP reduced held RMSE
`26.833 -> 26.737` and MAE `9.108 -> 8.797`, while episode Spearman fell
`0.743 -> 0.731`. The prior action-teacher use of this representation produced
stable success `9/192` versus direct-Q `23/192`. The unresolved distinction is
therefore the decision interface: direct action override versus auxiliary
representation learning.

## Action

Run one bounded matched Probe with a fixed auxiliary coefficient `0.002`.
Cm-on and Cm-off both compute the same frozen task-value target and carry the
same auxiliary head; only the loss coefficient differs. The target is the
current-action task-value residual predicted from direct-Q plus Cm-predicted
physical consequences. It is normalized using fit statistics, masked to current
contact, and never becomes an actor action, reward, success predictor, or future
observation. Start from the same self-trained epoch-260 source, seed 287, train
to epoch 280 with 64 environments, and evaluate seeds 290 and 291 with 96
complete episodes each.

## Decision rule and stopping

Proceed only after both 16-env epoch-262 smoke runs have finite targets and a
nonzero Cm-on auxiliary gradient. The policy Probe is locally promising only if
Cm-on is nonnegative in each evaluation seed and improves aggregate stable
success by at least 8 percentage points; otherwise close this exact auxiliary
route without coefficient or seed scans. Do not treat a learned auxiliary loss
as policy utility.

## Cost and boundary

One GPU, two short matched training runs, four evaluations, and less than one
hour of wall time are expected. The existing direct-Q/Cm representation
checkpoints and all user worktree changes remain untouched. This memo changes a
bounded decision interface, not the MISSION claim or the Cm physical-prediction
role.
