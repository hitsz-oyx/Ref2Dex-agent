# Workflow integration record — 2026-10-03

This commit brings the active single-session workflow from
`agent/cm-residual-policy` into `main` without merging that branch's residual-policy
research history.

Source snapshot: `snapshot/residual-workflow-source-20261003`

Source commits reviewed:

- `4e1710a` — workflow broker repair (retained as historical source context; the
  Broker implementation itself is not reactivated);
- `d7f5186` — retire the legacy agent workflow and keep the capacity watchdog;
- `3829792` — GPU-first execution policy;
- `efdf2b4` — structured capacity-error detection and delayed watchdog recovery;
- `d889e21` — final retired-agent skill wording.

The integrated tree keeps the current `main` research records, archive index,
skills, and registry for historical traceability. It does not import the
residual-policy experiments or delete the historical Broker files. The active
runtime is the single-session capacity watchdog; the old Broker files remain
available only for historical inspection.

The pre-integration refs are preserved as:

- `snapshot/main-before-workflow-integration-20261003`;
- `snapshot/oracle-before-workflow-integration-20261003`;
- `snapshot/residual-workflow-source-20261003`.
