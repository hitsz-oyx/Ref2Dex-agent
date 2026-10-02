# P-20261003-cm-task-value-aux-policy

Family: Cm decision interface  
Type: Task-value auxiliary supervision matched policy Probe  
Status: COMPLETED EARLY — `UNPROMISING`, close this interface

This Probe separated the previous Cm action-teacher failure from a train-time
representation hypothesis. Both arms computed the same frozen task-value target:
current direct-Q plus the nonlinear residual MLP over frozen Cm-predicted physical
consequences. The target was normalized and masked at current contact. Cm-on added
a fixed auxiliary loss coefficient `0.002`; Cm-off carried the same target/head
path with coefficient `0`. Neither arm changed the action, reward, success label,
or actor observation.

Both 16-env epoch-262 engineering smokes completed. Cm-on produced a finite
nonzero auxiliary gradient (`5.70e-4` in the smoke, `4.16e-4` in the matched
run). The matched Probe used the same epoch-260 source, training seed 287, 64
environments, and epoch 280 endpoint. Cm-off and Cm-on both trained successfully
and saved checkpoints. The first evaluation seed used 96 complete episodes per
arm:

| arm | stable success | post-success drops |
|---|---:|---:|
| Cm-off | 20/96 | 20/96 |
| Cm-on | 1/96 | 1/96 |

The predeclared per-seed nonnegative condition therefore failed by `-19` stable
successes on seed 290. The runner stopped before seed 291, as specified; this is
an early-stop Probe result, not a missing completion hidden as a positive result.
The task-value target can backpropagate into the shared head, but under this
contract it sharply degraded the learned policy. Close the interface and do not
scan the coefficient, seed, or target. The frozen Cm physical predictor remains
valid as a physical model, while this task-value auxiliary route does not supply
policy utility.

The full manifest, checkpoint hashes, and completed evaluation artifact paths are
in [`P-20261003-cm-task-value-aux-policy-results.json`](P-20261003-cm-task-value-aux-policy-results.json).
