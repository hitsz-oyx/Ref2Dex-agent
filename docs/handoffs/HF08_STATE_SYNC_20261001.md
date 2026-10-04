# HF08 STATE sync handoff — 2026-10-01

`TASK_ID=T-20261001-hf08-state-sync`  `STATUS=COMPLETED`

## Source and exact diff

The canonical source was read from `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent` at
`source_commit=d889e215784152302706c60c8ff324b3e0207abc`. Its complete `docs/STATE.md`
bytes were used as the baseline before editing; the agent branch's older STATE was not
used as a replacement baseline.

Input SHA256 (rechecked before handoff):

- `docs/STATE.md`: `d81801a094c0a6bab8f805eebbc3ed0f4a4eb91b1a231143e237ffe46be9cb1d`
- `docs/archive/2026-10-04-research-governance/RESEARCH_QUEUE.yaml`: `99fe825275e36061890805ced954faca9a193ed6daa6b9b205cf3bc814e0e8b6`
- `docs/experiments/probes/P-20260930-cm-physical-value.md`: `5db73733fe9b59722e1437b998c0805a9cc1436550c73ba9092506e75417737f`

Exact `docs/STATE.md` diff from that source baseline:

1. Date `2026-09-30` → `2026-10-01`.
2. Replaced the stale Current decision block with the completed HF08 facts: r7
   `48/48`, `plain_off 41/384`, `direct_q 40/384`, `cm_value 33/384`, native
   `UNPROMISING`, family `PAUSED`, budget `1/1`, no Validation/local tuning, unchanged
   North-star, active Mission-level Goal, fixed Broker dispatch, and the pending
   `T-20261001-hf08-value-target-audit` CPU-only `agent_cm` audit.
3. Reworded the earlier parallel calibration/distillation dispatch as historical context;
   retained the resource caps, no-formal-claim boundary, and the old-r2 ridge/in-sample
   limitation.
4. Changed the HF08 family row from `ACTIVE / r7 执行中` to
   `PAUSED / r7 native gate UNPROMISING / 1/1 budget used`.
5. Replaced the stale Next step and `HF08 current execution` sections with the current
   CPU-only audit and a concise completed-Probe record. Historical HF01–HF07 and support
   contract evidence remains in the source-baseline body.

The source-to-output unified diff is 32 additions and 68 removals; `git diff --check`
passes. No source worktree, main worktree, queue, probe card, runtime database, or other
path was modified.

## Output and resources

- `docs/STATE.md`: SHA256 `beb9667c6900ac8b7bd646c8e6a3a20f02c2ac1161943833a6f398c96cec9435`, mode `0664`, size `17328` bytes.
- `docs/handoffs/HF08_STATE_SYNC_20261001.md`: mode `0664`; hash recorded in the
  Broker handoff after this file was finalized.
- Branch: `agent/infra`; commit: `none` (use `BYTE_EXACT_ROOT_IMPORT`).
- Resource evidence: CPU-only document edit, one CPU thread, GPU `0`, experiments/training
  `false`, no daemon or collector, no lingering process, under 15 minutes and under the
  0.01 GiB output budget.
- Verification: source hash recheck, required STATE content checks, and
  `git diff --check` passed. No broad tests were run because this was a documentation-only
  synchronization.

## Root integration

Import the two allowed files byte-exactly after reviewing the listed source and output
hashes. The active `agent_cm` audit result is intentionally not anticipated here.
