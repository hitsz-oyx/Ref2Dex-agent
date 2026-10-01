# Ref2Dex workflow repair design

Status: approved for implementation on `agent/workflow-repair`.

## Goal

Make the fixed-role Broker workflow truthful and safe for local operation while
preserving the current research policy, evidence records, runtime bindings and
external provider choice. A task being stored is not treated as a task being
started.

## Contract

- `RUNNING` is the only supervisor state that admits a new dispatch or claim.
  `PAUSED` and `STOPPED` reject new work; leased/running work may finish or
  hand off so pausing does not kill experiments.
- A claimed task has one lease token. Updates and handoffs require that token,
  require an unexpired lease, and may only use documented task transitions.
- The Broker remains provider agnostic. A runtime adapter consumes the persisted
  dispatch message and reports `STARTED`, `FAILED`, or `COMPLETED` through the
  Broker. A deterministic fake adapter covers the local end-to-end contract;
  real provider launch remains an explicit binding/adapter concern.
- `AGENT_ROLES.yaml` is the tracked role source. Registry and binding checks
  validate branch, role, provider, identity and lifecycle consistency without
  copying a second role definition.
- Current docs describe one canonical path. Legacy poller helpers may remain
  for migration compatibility, but current poller/watchdog code must not depend
  on legacy recovery or queueing behavior.

## Components

1. Broker state guards and lease validation.
2. Runtime adapter protocol and file/SQLite-backed fake adapter for tests.
3. `workflow doctor` for role, binding, database, lease and adapter checks.
4. Compact workflow contract and corrected operational examples.

## Verification

The implementation must include tests for paused dispatch/claim rejection,
allowed in-flight completion, missing/expired/wrong leases, duplicate dispatch,
adapter idempotency and failure reporting, role/binding drift, and a complete
fake dispatch-to-handoff flow. Existing governance and research tests remain
unchanged except where stale workflow assumptions are corrected.
