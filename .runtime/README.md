# Local workflow state

New supervision uses a machine-local workflow.json, its configured state_dir/research.sqlite,
supervisor/guardian process locks and logs, plus isolated external backend stores. These are
not research evidence and are not committed. Commands and lifecycle are documented in
[OPERATIONS](../docs/workflow/OPERATIONS.md).

Old AGENT_BINDINGS, leases, tasks.sqlite, AGENT_STATE and watchdog cursors remain drain-only.
Do not copy them into the new executor or enable old/new dispatch owners together.
