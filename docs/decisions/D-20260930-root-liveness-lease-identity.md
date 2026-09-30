# D-20260930: root liveness lease identity (draft)

**Status:** `DRAFT_FOR_ROOT_INTEGRATION`

## Current decision

Repair the supervisor lease writer so an enabled lease is populated from the
canonical root runtime identity and a pause or migration preserves identity
fields. The watchdog keeps its exact `conversation_id`, `codex_home`,
`worktree`, `branch`, and provider checks. A lease without a resolvable
canonical identity fails closed instead of weakening those checks.

The controlled research choice supplied by root is two parallel, bounded
engineering/Probe routes: one fit-only Cm calibration repair using at most 2
CPU, 15 minutes, and 1 GiB; and an independent six-expert trajectory
distillation using at most 1 GPU, 60 minutes, and 5 GiB. Neither route creates
a formal Cm claim. The old r2 result was a ridge plus in-sample residual
screen; it does not decide the design MLP or the entire Cm route.

## Evidence and stop conditions

This task changes lease identity plumbing and adds regression coverage only. It
does not read or write a Goal database and does not start a daemon. The Cm
repair stops if its local calibration gate fails; the independent distillation
route may continue under its own preflight. Any Probe result remains
`PROMISING`, `UNPROMISING`, or `UNCLEAR`.

## Controlled activation command

After root imports the reviewed paths, refresh the lease with the canonical
registry and both Broker databases:

```bash
python3 scripts/researchctl.py supervisor resume \
  --registry docs/AGENT_REGISTRY.json \
  --lease .runtime/SUPERVISOR_LEASE.json \
  --broker-tasks-db .runtime/tasks.sqlite \
  --broker-state-db .runtime/AGENT_STATE.sqlite \
  --broker-bindings .runtime/AGENT_BINDINGS.json
```

The command only refreshes the local lease and desired Broker state. The
watchdog must still perform its own identity check and app-server readback.
