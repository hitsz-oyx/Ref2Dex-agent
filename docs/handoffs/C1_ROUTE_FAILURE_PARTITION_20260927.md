# C1 route-vs-downstream failure partition — 2026-09-27

This is post-hoc descriptive evidence from the accepted, frozen
`VAL-20260926-observation-six-expert-c1` matrix. It is not a new Validation,
does not alter the accepted C1 claim, and does not establish a Cm mechanism or
Cm policy utility.

## Provenance and join contract

The immutable source records are:

- Parent manifest: `outputs/CmResidual/val_observation_router_c1/run_manifest.json`
- Preflight: `outputs/CmResidual/val_observation_router_c1/preflight.json`
- Native per-arm records:
  `outputs/CmResidual/val_observation_router_c1_s{400,401,402,403,404}_{fixed,obs}/`
- Frozen route mapping: `src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json`
- Evaluator schema: `third_party/DExplore/dexplore/evaluate_object_router.py`
- Immutable analysis index: `docs/experiments/validations/VAL-20260926-observation-six-expert-c1-results.json`

For each seed, join `fixed/results.json` and `obs/results.json` by exact
`env_id`. The Validation pairing contract also requires equal `motion_id`,
`start_frame`, and `steps`; all 320 pairs pass. Reconstruct the fixed expected
expert as:

```text
motion_id -> preflight.motion_files[motion_id] -> object name -> route_config.object_route[object]
```

The observation expert is `obs/initial_routes.json.expert_by_env[env_id]`,
which the evaluator records after the first inference-visible observation.
The router model provenance is the saved `model_sha256` and native manifest
hashes.

## Descriptive partition

- Initial route match: **311/320**.
- Initial route mismatch: **9/320**; all nine observation episodes missed
  held-lift. They are `source_e260 -> cup_e340` predictions on seven
  `cubesmall` and two `waterbottle` environments.
- Among the 311 route-matched environments: **123** observation held-lifts and
  **188** observation held-lift failures.
- All 188 route-matched failures have positive aggregate
  `hand_object_contact_fraction`; **139** have positive
  `max_contact_lift_m`, while **49** have zero; only **10** have a positive
  `max_lift_contact_run_steps` and **178** have zero.
- The 188 route-matched failures pair as 169 fixed-fail/observation-fail and
  19 fixed-success/observation-fail.

The evidence supports the bounded statement that initial routing errors are a
minority of the observed held-lift failures; most observed failures occur
after the initial route agrees with the frozen expected route. It does not
identify a causal mechanism for those downstream failures.

## Missing fields and decision use

The saved records do not contain the raw 1442-D initial observation, router
confidence or margin, action sequences, per-step expert choices, or per-step
contact/force trajectories. Therefore this partition cannot explain why a
route mismatch occurred or attribute a route-matched failure to a particular
contact phase, joint, or expert mechanism.

The smallest deterministic CPU-only follow-up, if later authorized, is a
schema/hash check followed by the join above and counts stratified by route
match, `lift_success`, `hand_object_contact_fraction`,
`max_contact_lift_m`, and `max_lift_contact_run_steps`. It needs no simulator,
GPU, fitting, new labels, or new data. This handoff only preserves the already
audited counts; it does not run or promote that follow-up.

## Integration handoff

```text
BRANCH=agent/observation-router-reliability
BASE_COMMIT=3697ddd2b37339990c97aaa025bd37924a673c3c
MAIN_REFERENCE=d4f6c2a
COMMIT=none
COMMIT_BLOCKER=linked Git metadata is read-only
INTEGRATION_MODE=BYTE_EXACT_ROOT_IMPORT
FILES=docs/handoffs/C1_ROUTE_FAILURE_PARTITION_20260927.md,docs/STATE.md
```

The accepted Validation card, result index, native manifests and queue are
untouched by this handoff.
