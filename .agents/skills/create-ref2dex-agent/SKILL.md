---
name: create-ref2dex-agent
description: Retired. Ref2Dex now uses a single-session workflow and does not create or rebind agent runtimes.
---

# Retired skill

This skill is kept as a tombstone so old references fail clearly instead of
silently restoring the retired Broker workflow.

The current workflow is defined by [`AGENTS.md`](../../../AGENTS.md). The active
session performs research, implementation, validation, and documentation
directly. Do not create or resume subagents, fixed roles, worktree runtimes,
task leases, or Broker queues through this skill.

The only runtime helper currently supported is
[`scripts/codex_research_supervisor.py`](../../../scripts/codex_research_supervisor.py),
which only queues `继续` after a recent model-capacity failure.
