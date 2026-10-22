---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-executor-qualification
experiment_id: P-20261022-wam-v6-executor-qualification
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded in each arm manifest
claim_id: C3
hypothesis_family: HF-wam-v6-executor-qualification
probe_index_in_family: 1
seed_pool: probe
seeds: [185]
classification: Blocker
status: RUNNING
run_id: executor-qualification-20261022-r1
decision_changed_if_positive: only after robot GT qualification compare healthy and FM tau to locate a generator execution gap
decision_changed_if_negative: prioritize execution chain; do not call tau decoding the main bottleneck
---

# Fixed executor on unseen robot sequences

This tests whether the existing tau-adapted executor can execute robot future
references before attributing physical failures to WAM. Cheapest discriminating
comparison: paired native robot command replay, frozen executor with GT robot
future, healthy4371, and frozen FM8500. No new generator training and no
identifiability diagnosis. P0 has priority over the independently prepared P1.

## Frozen protocol

- Executor: Scheduled686 `inspire-scheduled686-bc-20261020-r4/latest.pt`, step50000,
  actual trained nonzero tau columns, not step0 DEV-best. No fitting between arms.
  This executor was adapted to v4 spatial tau; it is NOT certified as v6-adapted.
  A failure therefore diagnoses this bridge, not all possible v6 executors.
- Panel: DEV/val sequences disjoint from this executor's corpus TRAIN. Select
  deterministic sorted-object round-robin first16 among prefix>=80 with reference lift>=5cm and
  motion-coupled>=15 consecutive frames; all candidates and denominators saved.
  Selection uses reference only, never policy outcomes. WAM lifetime exposure
  cannot be certified; do not call it lifetime WAM heldout.
- Same native seed185, fixed motion membership and initial state arrays checked
  across arms. Physical GPU2 inference,3 simulation, GPU0/1 unused; 4/5/6/7 excluded.
- 300 controls/10s per arm,30Hz; metrics censored at each reference's admitted
  prefix. Full traces retained separately. Reference command replay tests actual
  replayability; offline safe prefixes alone do not certify it.
- GT future is robot hand reference in world coordinates, future-minus-current
  measured hand, K8. Future reference is allowed only in this oracle P0 arm.
- Healthy4371 and FM8500 use live measured H4, causal TaskController goal/events,
  dense CAD2048. Sparse hand11 populates corresponding joint21 mask, unobserved
  joints/skin remain invalid. First8 of full24 forecasts reach fixed executor.
  No future GT goal or hand enters predicted conditions/history.
- Recovery: natural slip/loss followed by re-established hold; no forced physical
  disturbance has been implemented, so perturbation robustness remains untested.
- Budget: <=1h per arm, <=4h entire P0, output <=2GiB, disk floor20GiB. No existing
  jobs stopped, restarted, pushed, or signalled; no credential access.

## Metrics and gates

Full original TaskController completion and path progress are distinct from a
single grasp-lift proxy. Proxy success: lift>=5cm, unsupported, palm-object
proximity<=20cm and at least2 measured finger contact bodies above0.1N for30
consecutive controls. Report lift-only>=5cm30 separately (ballistic motion is
not holding). Max continuous hold duration and any recovery count are per case.
Robot replay must qualify at>=80% proxy success and GT must attain>=80% of the
replay success rate; otherwise do not attribute failure primarily to generator.
A predicted deficit requires paired GT advantage>=20pp; confidence intervals
and small selected-panel limits must accompany any apparent deficit.

Failure labels are observable proxies and may overlap: no_lift, loss_after_lift,
no_finger_closure_contact, pushed_or_tipped (table-supported horizontal drift
>5cm or tilt>45deg), workspace/arrival failure. Early contact requires a timed
reference-contact comparison. Full mesh penetration is unmeasured by this host;
report UNKNOWN, never infer absence from keypoints. Contact count is net-force
body evidence, not a contact-pair proof that every force came from the object.

## Reproduction / environment

Entry: `tools/run/evaluate_executor.py` with `--panel .../panel.json --arm replay|gt|healthy|flow`,
`--stats outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json`,
`--gpu 2 --native-gpu 3 --controls 300`; prediction arms add `--wam` exact checkpoint.
Outputs: `outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/`.
Native host external output declared under `outputs/wam-rl-policy-v3/executor-qualification-*`.
Runner graspenv Python3.8/Torch2.4.1+cu121; native pinned dexplore_v120_train
Python3.8/Torch2.0.1+cu118, Isaac imported before Torch. Each manifest records actual
command, code commit, seed, packages, checkpoint, elapsed time and runtime calls.

## Results

Pending actual simulation. No success, bottleneck or sample-efficiency claim yet.
