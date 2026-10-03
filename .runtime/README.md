# Local workflow state

New supervision uses a machine-local workflow.json, its configured state_dir/research.sqlite,
supervisor/guardian process locks and logs, plus isolated external backend stores. These are
not research evidence and are not committed. Commands and lifecycle are documented in
[workflow README](../docs/workflow/README.md).

Old AGENT_BINDINGS, leases, tasks.sqlite, AGENT_STATE and watchdog cursors remain drain-only.
Do not copy them into the new executor or enable old/new dispatch owners together.

AUTONOMOUS_DECISIONS.md is a rebuildable view of consequential choices; use researchctl decisions --export to save a new reviewable research record. No credentials or experiment manifests belong in control exports.
