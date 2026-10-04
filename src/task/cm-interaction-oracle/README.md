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
weak or outlier-dependent. Task `docs/ref/ref1.md` now motivates cross-fitted
task advantage → action critic, before predicted E/I contribution. The first
relative-label Probe stopped at G0: fit-label agreement failed despite broad
episode support; action critic and new Cm fitting were not executed.

Current prerequisite is reliable local physical labels/value calibration.
Do not extend I/G capacity or select future-active test rows to bypass G0.
Continuous advantage trends remain a candidate, without an action-information
claim. Neither offline oracle correlation nor prediction loss demonstrates
the final matched Cm-on/off trained-policy utility.

Run tools: `tools/run/probe_relative_action_critic.py`; audit:
`tools/audit/audit_relative_action.py`. Protocol and completed evidence:
[relative action Probe](docs/experiments/probes/P-20261004-recap-relative-action.md).
