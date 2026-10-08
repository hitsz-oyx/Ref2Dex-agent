schema: ref2dex.probe.v2
probe_id: P-20261009-gpu-group-contract
experiment_id: P-20261009-gpu-group-contract
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 29bec32
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 2
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain a same-process GPU group only if a nonzero-origin zero pair also preserves native behavior and candidate margin
decision_changed_if_negative: close environment-role/layout adjustments and require a different replay contract
status: UNPROMISING
run_id: gate1-gpu-precontact-group-20261009-r1, gate1-gpu-spacing-group-20261009-r1, gate1-gpu-nonzero-pair-group-20261009-r1-r2
---

# Native GPU group role/layout contract

## Decision Note

The previous four-environment group paired env0 at the world origin with a
nonzero-origin env. A read-only audit found a repeatable tick-1 drift in that
pair (about 3.46 mm hand position and 0.0116 rad joint position), before the
large tick-44 contact-force divergence. Historical packets showed that
nonzero-origin roles can be much closer to each other. This bounded Probe tests
whether moving the query before contact or selecting nonzero-origin zero roles
can make the synchronous group usable.

The cheapest discriminating design is a 72-step, native GPU PhysX/GPU-pipeline
group with 256 actor rows and the same broadcast controls. It remains an
engineering calibration and never enters the strict Gate1 scorer.

## Implementation contract

- `--query-tick` is exposed for bounded group diagnostics; zero preserves the
  established tick48 query.
- `--zero-env-pair` selects the two noncandidate roles used for calibration.
- Optional `--engineering-env-spacing` writes a task-owned YAML copy and adds
  its hash and override to replay provenance. This is a scene-layout diagnostic,
  not a native backend replacement.
- Query, environment-layout hash, backend, actor layout, physics and controller
  identity are checked when replay packets are reused.

## Results

1. **Precontact query.** Four envs, query tick32, 72 steps, default spacing.
   The selected zero pair already diverged at tick1 despite identical controls;
   pre-query p95 was `4.19e-3 m` for hand points and `1.15e-2` for joint
   position. The baseline max lift was only `0.245 m`; zero-pair and candidate
   margin gates failed. Audit:
   `outputs/consequence-evaluator/gate1-gpu-precontact-group-20261009-r1/group-contract-audit.json`.

2. **Layout diagnostic.** A reduced `envSpacing=0.5` run did not restore the
   contract (baseline `0.297 m`; zero p95 and effect gates failed). The smaller
   spacing can alter broadphase workload, so this run is retained only as an
   engineering diagnostic and is not evidence for a production backend.
   Audit:
   `outputs/consequence-evaluator/gate1-gpu-spacing-group-20261009-r1/group-contract-audit.json`.

3. **Nonzero-origin zero pair.** Eight envs with 32 actor copies each kept the
   fixed 256-row actor contract. With pair `[1,4]`, the selected zero pair passed
   its p95 geometry gate, but the all-zero-role pairwise gate failed; baseline
   max lift was `0.183 m`. With the more stable pair `[4,6]`, selected zero p95
   was zero for visible geometry and baseline max lift was `0.234 m`, but the
   pairwise gate and candidate effect gate still failed. In both runs controls
   were broadcast exactly while group state hashes were not. Audits:
   `outputs/consequence-evaluator/gate1-gpu-nonzero-pair-group-20261009-r1/group-contract-audit.json`
   and
   `outputs/consequence-evaluator/gate1-gpu-nonzero-pair-group-20261009-r2/group-contract-audit.json`.
   The predeclared full-horizon confirmation with `[4,6]` reached only
   `0.153 m` and `9` held frames; its selected zero pair stayed mechanically
   exact before query, but the full behavior, pairwise, and effect gates all
   failed. Audit:
   `outputs/consequence-evaluator/gate1-gpu-nonzero-pair-group-20261009-r3/group-contract-audit.json`.

## Decision

Role reordering and an early query do not provide a valid native GPU candidate
contract. The group can make selected nonzero-origin geometry pairs nearly
repeatable, but it does not simultaneously preserve the policy behavior,
common zero-role noise floor, and candidate margin. Close this group
role/layout route. Strict Gate1 still needs a verifiable hidden-state fork or a
different execution contract; no evaluator, PointWorld, execution bridge, or
MPC work follows from these packets.

## Limits

All runs are single-seed, 72-step engineering probes. The spacing override may
change GPU broadphase scheduling and is explicitly not a scientific backend
comparison. No result supports a task-success, utility, or deterministic-physics
claim.
