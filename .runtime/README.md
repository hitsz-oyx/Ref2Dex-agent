# Local runtime state

Files in this directory are machine-local and are not research evidence.

The current workflow has one optional runtime helper:
`session_capacity_watchdog/state.json` and its lock belong to
`scripts/codex_research_supervisor.py`. The helper reads Codex metadata and
rollouts, and queues one `继续` 60 seconds after a structured capacity failure.
Ordinary text mentioning capacity does not trigger it, and a successful queue
clears that recovery arm. It does not
create agents, dispatch research tasks, manage leases, or change experiment
state.

Do not restore the retired Broker, role registry, root watchdog, pollers, or
their lease files from old runtime artifacts.
