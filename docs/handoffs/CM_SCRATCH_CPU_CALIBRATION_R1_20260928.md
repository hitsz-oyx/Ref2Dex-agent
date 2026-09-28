# Cm scratch CPU calibration r1 handoff

TASK_ID: `T-20260928-cm-scratch-teacher-arbitration-cpu-calibration-r1`
AGENT: `agent_cm`
STATUS: `HANDOFF_READY`
DECISION: `NO_GO`

The r6 fit and holdout records provide the required six-arm actions and most
of the physical target contract.  CPU read-only checks found:

* fit: 189 rows, arm counts `31/32/32/31/32/31`, actions `[189,6,18]`;
* holdout: 186 rows, arm counts `32/30/30/31/31/32`, actions `[186,6,18]`;
* both have 1442-D pre-action observations, one-step object-local pose pairs,
  exact `target_delta_object_local_1`, five contact indicators, globally unique
  episode IDs, disjoint split IDs, all six arms, and propensity `1/6`;
* assigned executed actions and router actions match their candidate IDs, all
  audited tensors are finite, and router provenance is `c1_observation_router`.

The scratch contract still cannot admit these rows.  `object_lift_axis` is
absent from both records, both manifests, the collector config, and evaluator
source.  The stored object-local displacement is a valid three-vector, but no
recorded unit lift axis identifies the certified direction.  Assuming local z
or using future scalar lift would fabricate the missing target semantics.

`validate_rows` was invoked on in-memory rows using only recorded fields and
losslessly derived five-step contact retention.  It stopped as required with:

```text
row is missing fields: ['object_lift_axis']
```

Therefore no transition-model fit, contact calibration, or Cm-on labels were
produced.  The observed router label is id `4` (`source_e260`) for all 375
rows, but it is recorded C1 router output and was not substituted as a static
fallback.  No policy or formal Cm claim follows.

The machine-readable audit is [CM_SCRATCH_CPU_CALIBRATION_R1_20260928.json](CM_SCRATCH_CPU_CALIBRATION_R1_20260928.json).

## Exact input and resource evidence

Fit records SHA256:
`7f38b1fd310f0dfea46d9e1e754893020223436dbc2a2ba13628d39a9c5c1a71`

Holdout records SHA256:
`c7abaf9524f06971b4bff18bd1cfef769b3ce7653345d8853629525b8738dff8`

Fit manifest SHA256:
`dd1558cf25238dcd631e937ee5e4f09133651c94e5aea5e1141f2388798bf7e0`

Holdout manifest SHA256:
`49bb9b0c45868d7b7377586519f61c9ed9a8c4440527d5f6debbc7f432515b70`

No GPU, Isaac Gym, collector, fitting, Cm training, student distillation,
PPO, online process, or new transition was started.  Existing outputs were
read only.

## Minimum repair

Add an explicit `object_lift_axis` field to each future support row, including
unit-vector, coordinate-frame, and provenance metadata.  Preserve the current
pose pair and delta.  Do not rerun calibration until this field is present and
the contract passes without synthetic values.

BRANCH: `agent/cm`

HEAD: `a0ecee73db7bc7bf9ddb6b9016d028ba16943237`
