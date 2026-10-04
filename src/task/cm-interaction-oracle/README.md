# cm-interaction-oracle

Task for action-conditioned physical consequence E/I and task-relative action
quality, on branch `agent/cm-interaction-oracle`. The Task name is the branch
suffix; Git branch and experiment identities remain separate.

New execution and audit tools live here. Existing reusable spatial models in
`ObjectInteractionCmv2` and native Inspire geometry in `CmResidual` are reused.
Historical scripts/cards remain at their recorded paths during gradual migration.
New cards live under `docs/experiments/`; outputs use
`outputs/cm-interaction-oracle/<run_id>/`. Mission, campaign, state and seed
ownership remain in the repository-wide docs.

Ref5 point-flow G and surface-I probes are completed; absolute RTG gains were
weak or outlier-dependent. Task ref1's cross-fitted three-class advantage
labels stopped at G0. Task ref2 then explicitly motivated a DIFFERENT
matched ranking probe using the same continuous labels, without V/threshold
re-tuning: H59.71% versus current-action59.62% held-out conditional pair
accuracy, weak action permutation effect; UNPROMISING for this frozen contract.
Actual between-V test Spearman0.540 contradicts using Pearson0.978 as evidence
of strong sorting stability. Neither discrete G0 nor ranking G1 was rescued.

Current prerequisite is physically credible action/outcome contrast, rather
than another I/G architecture on these proxy labels. Retain relative-outcome
and physical E/I as candidates; no core Cm refutation or policy utility claim.
Do not select future-active/agreed test labels to rescue a failed gate.
Neither offline oracle correlation nor prediction loss demonstrates the final
matched Cm-on/off trained-policy utility.

Run tools: `tools/run/probe_relative_action_critic.py`; audit:
`tools/audit/audit_relative_action.py`. Protocol and completed evidence:
[relative action Probe](docs/experiments/probes/P-20261004-recap-relative-action.md).

Ranking run: `tools/run/probe_action_ranking.py`; independent raw-score audit:
`tools/audit/audit_action_ranking.py`.
[Ranking Probe](docs/experiments/probes/P-20261004-relative-action-ranking.md)
records the conditional negative branch and deferred true candidate tests.
