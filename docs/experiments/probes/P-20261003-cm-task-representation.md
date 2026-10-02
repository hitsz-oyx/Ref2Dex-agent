# P-20261003-cm-task-representation

Family: Cm decision interface
Type: Nonlinear task-representation and matched policy Decision Probe
Status: COMPLETED — offline critic signal `PROMISING`; policy interface `UNPROMISING`

This Probe tested a new representation after the linear critic residual route was
closed. A frozen two-layer SiLU MLP predicted realized complete return minus the
frozen direct-Q. Its input was direct-Q, candidate action, current state, and the
frozen Cm ensemble's predicted object delta/velocity/contact/reward/terminal mean
and uncertainty. It never received a future state and it did not train an actor.
The fit/held data contract was 831,811/209,788 rows over 1,536/384 episodes.

On held rows, direct-Q had RMSE `26.833`, MAE `9.108`, row Spearman `0.589`, and
episode Spearman `0.743`. The matched Cm representation had RMSE `26.737`, MAE
`8.797`, row Spearman `0.615`, and episode Spearman `0.731`. Episode-level
bootstrap versus direct-Q gave RMSE delta mean `-1.532` with 90% interval
`[-1.851,-1.213]` and MAE delta mean `-0.275` with interval `[-0.482,-0.053]`.
This is a local held critic signal: episode ranking declined, so it did not justify
a value or policy claim by itself.

Because the local signal was credible enough for a bounded follow-up, a matched
policy Probe trained direct-Q and Cm-representation arms from the same epoch-260
source, seed 286, 64 environments, and 40 epochs. Each arm was evaluated on seeds
288 and 289 with 96 complete episodes. Direct-Q achieved stable success `23/192`
and post-success drops `23/192`; Cm representation achieved stable success
`9/192` and drops `8/192`. The predeclared aggregate stable-success delta was
`-14`, so the policy interface is `UNPROMISING`. The Cm representation changed
what the critic could fit offline but did not produce a safer or more successful
policy under this training contract.

An initial launcher attempt reached epoch 300 but failed while resolving a relative
checkpoint path inside the DExplore working directory. Its artifacts remain under
the `policy_probe_r1` output directory and are excluded from the result. The
successful rerun used absolute output paths in `policy_probe_r2`; the runner fix is
part of the committed implementation.

Close this actor wiring and do not add epochs, seeds, ordinary data, or PPO. Keep
the frozen MLP checkpoint and offline result as a representation-learning clue for
a future route that changes the decision experiment rather than merely inserting
this residual into the current policy. The policy manifest is in
[`P-20261003-cm-task-representation-policy-results.json`](P-20261003-cm-task-representation-policy-results.json),
and offline metrics are in [`P-20261003-cm-task-representation-results.json`](P-20261003-cm-task-representation-results.json).
