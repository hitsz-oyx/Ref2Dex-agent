# Fixed-pool workflow cleanup — 2026-09-27

## Scope

The repository was migrated to the fixed pool declared in
`docs/AGENT_REGISTRY.json`:

```text
main
agent/cm
agent/rl
agent/eval
agent/infra
```

Historical experiment branches were not merged wholesale into `main`. Their
committed tips are preserved as annotated `archive/20260927/*` tags.

## Completed cleanup

- 46 fully merged, unreferenced local branches were deleted.
- Clean inactive worktrees for the selective Cm, temporal-credit and workflow
  lifecycle branches were removed.
- Dirty but inactive poller, router and root-policy worktrees were byte-archived
  and removed. Their tracked changes are in `tracked.patch`; untracked files are
  copied under `untracked/`.
- The old `agent_result_poller.py --root-goal-resume-once` process was stopped by
  exact PID after verifying its command line. The new worker event poller and root
  watchdog remain the only supported workflow daemons.
- Legacy runtime bindings for `agent_poller`, `agent_c1_validation`,
  `agent_cm_temporal` and `agent_workflow` were moved to
  `retired_conversations`. The machine-local `.runtime/AGENT_BINDINGS.json` now
  binds only root; fixed worker roles are unbound and ready for explicit rebind.

## Pending cleanup

`agent/grab-full-baseline` remains checked out at
`Ref2Dex-agent-baseline` because multiple Codex processes still have that exact
worktree as their current directory. Its committed tip and all uncommitted
tracked/untracked content were archived at:

```text
/home2/wyy/oyx_ws/ai_ws/.ref2dex-branch-archive/20260927/agent__grab-full-baseline/
```

This directory also contains the **common Git database** in `.git`; the current
main worktree points into it. Even after its Codex processes exit, do not remove
the directory or `.git` without separately migrating and verifying the common
Git database. Process exit alone is insufficient permission to delete it. The
archive manifest is at
`/home2/wyy/oyx_ws/ai_ws/.ref2dex-branch-archive/20260927/MANIFEST.json`.

During final verification, the legacy workflow conversation recreated
`Ref2Dex-agent-workflow-v22-audit` on
`agent/root-goal-lifecycle-test-20260927` at `98c1a7b`. Its registered rollout
still contains an active own turn, despite Goal status `NONE`. The worktree is
clean and its tip is already archived, but is retained to avoid deleting an
active owner's workspace. Its retired runtime binding remains retired; this
recreated checkout does not authorize a new permanent role or research task.
Wait for a terminal handoff and recheck files and processes before pruning it.
