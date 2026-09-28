# Cm scratch support-contract audit handoff

TASK_ID: `T-20260928-cm-support-contract-audit`
AGENT: `agent_cm`
STATUS: `HANDOFF_READY`
DECISION: `NO_GO`

The existing HF02 temporal records are not sufficient to execute the scratch
teacher-arbitration contract.  They contain pre-action state/history,
`candidate_actions[N,6,18]`, known `1/6` assignment, route/checkpoint/motion
provenance, and twenty future contact slots.  They do not contain a one-step
object pose or signed object-local displacement, and they do not contain the
frozen C1 observation-router teacher label.  Fit and holdout use different
simulator seeds, but `env_id` is reused and there is no globally unique
episode identifier, so split disjointness is not independently machine
verifiable.

The machine-readable audit is
`docs/handoffs/CM_SCRATCH_SUPPORT_MANIFEST_20260928.json`.

## Matched Cm-off correction

The P0 control is the frozen C1 six-expert observation-router teacher label on
the identical state/candidate row.  Cm-on may replace only that offline label
with the certified Cm candidate.  If no candidate passes, Cm-on must fall back
to the exact router label.  A fixed `source_e260` fallback or Cm-off label is
forbidden because HF02 `route_expert=source_e260` is the fixed simulator route,
not the observation-router label.

The accepted design config and contract were updated accordingly:

* `cm_off = frozen C1 observation-router teacher label`;
* `fallback = frozen C1 observation-router teacher label`;
* contract selection requires a per-row router label when falling back.

## Minimal collection contract for a future re-dispatch

Do not collect under this task.  If root later authorizes a new collection, use
six balanced arms with propensity `1/6`, execute one candidate at each matched
pre-action trigger, and record all six candidate actions plus the frozen router
label.  Each row must include globally unique `episode_id`, pre-action
observation and object pose, `object_pose_t_plus_1_object_local_frame`, the
signed one-step target, five contact indicators, assignment/propensity,
checkpoint and router hashes, and split-owned route/config hashes.  Use two
episode-disjoint splits (minimum 30 valid rows per arm), one first episode per
environment, at most one GPU, 20 wall minutes, and 1 GiB output.  Stop on any
missing arm, provenance mismatch, missing target/contact step, split overlap,
propensity drift, hash drift, non-router fallback, or budget overflow.

This is a data contract only.  No fit, policy utility claim, or formal Cm
conclusion follows from this audit.

## Audited input hashes

The source hashes and observed shapes are pinned in the JSON manifest.  Key
inputs are:

* fit records: `38f311850f31549f6d3a050cff12df7ebccd00bd9a5ef65ed3a9b12a250f2506`;
  187 rows, six-arm counts `31/31/32/31/31/31`;
* holdout records: `cb851a96229df87501d890732cbdf0b90b2c2952e07d018edb6d83d47d641fe8`;
  186 rows, six-arm counts `30/32/31/32/31/30`;
* HF02 collector config: `47162342929f2b2889197d2d80b6a9069f6ee513d6a83d33cf99c25cba681536`;
* HF02 route config: `afedfa54c8573096c4d2104d3328efba32b5daf11445323792f45eca19c04d16`;
* HF02 evaluator: `8297c5a09c4bec5f3fa0987d9bb57a552d5a5e43e2dc162a3324ae8ce5b2e0ba`;
* frozen C1 router model hash: `1fa84c94891d63824be0100b5eb78f4aa1e37825c6d71fff65e517c0c7f2fc14`.

## CPU/static evidence

Read-only `torch.load(..., map_location="cpu")` verified both record payloads,
finite tensors, exact candidate shapes `[187,6,18]` and `[186,6,18]`, contact
shapes `[187,20]` and `[186,20]`, all six arms, and propensity `1/6`.  Static
source inspection confirmed that the evaluator obtains one action from each
of the six expert models.  No Isaac Gym import, fitting, simulator, collector,
GPU process, or new transition was started.

## BYTE_EXACT_ROOT_IMPORT

BRANCH: `agent/cm`
HEAD: `a0ecee73db7bc7bf9ddb6b9016d028ba16943237` (unchanged; working-tree additions only)

The final SHA256/mode/size values for the four updated design files and the two
audit handoff files are recorded in the Broker handoff evidence and must be
used for root import.  No forbidden path was modified.

NEXT: root may re-dispatch a minimal six-arm provenance collection to `agent_rl`
only after a separate resource authorization; do not fit or claim Cm utility
from the current HF02 records.
