# Local runtime state

Files in this directory are machine-local and are not research evidence.

The current workflow has one optional runtime helper:
`session_capacity_watchdog/state.json` and its lock belong to
`scripts/codex_research_supervisor.py`. The helper reads Codex metadata and
rollouts, and only queues `继续` after a recent capacity failure. It does not
create agents, dispatch research tasks, manage leases, or change experiment
state.

Do not restore the retired Broker, role registry, root watchdog, pollers, or
their lease files from old runtime artifacts.
