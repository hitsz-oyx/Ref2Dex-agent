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
status: UNPROMISING
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
- Primary GT is robot reference_future minus reference_current in world coordinates, K8;
  the first3 controls use zero/invalid tau, identical to prediction startup.
  Extra `gt` corrective oracle uses reference_future minus live measured hand,
  but its legacy startup was valid from control0; it changes two factors and
  is not an isolated origin ablation. Future reference is allowed only in P0.
- Healthy4371 and FM8500 use live measured H4, causal TaskController goal/events,
  dense CAD2048. Sparse hand11 populates corresponding joint21 mask, unobserved
  joints/skin remain invalid. First8 of full24 forecasts reach fixed executor.
  No future GT goal or hand enters predicted conditions/history. The original
  online cache had only512 points; `prepare_executor_geometry.py` genuinely samples
  2048 from the same native OBJ, preserving its32 query IDs in a new output cache.
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

## Results — completed, independent physical measurements

All five runs completed300 controls/environment. The primary metrics censor at
per-sequence admitted reference_controls (16 cases,2316 controls total per arm).
Each arm has the same saved q/dq/physical/object/hand/tick/table initial arrays,
all exact. Native host checks: PD error0, action/episode mismatch0, indexed resets0.
Robot corpus TRAIN589 vs heldout71 sequence names and resolved source paths overlap0.
Reference screening selected16 objects/16 sequences from30 eligible/71heldout;
this is a reference-selected DEV Probe, not an unbiased universal success rate.

| Input / control | grasp-lift hold success | longest hold mean / median / max seconds | full TaskController success |
|---|---:|---|---|
| actual robot command replay |15/16=93.75%|2.090 /1.767 /5.567|0/14 uncensored;2 censored|
| **standard robot GT tau, matched startup** |**0/16=0%**|0.0542 /0 /0.8667|0/14;2 censored|
| extra live-centered GT, legacy startup |0/16|0 /0 /0|0/14;2 censored|
| healthy4371 predicted tau |0/16|0.0438 /0 /0.6667|0/14;2 censored|
| fixed FM8500 predicted tau |1/16=6.25%|0.0625 /0 /1.000|0/14;2 censored|

The full original object-path task differs from the grasp-lift proxy. No arm
completed that task;14 observations reach the declared reference endpoint and2
are right-censored. Do not replace these denominators with16 full-task failures.
GT minus healthy success0pp (paired sequence bootstrap95%CI[0,0]); GT minus FM
-6.25pp, CI[-18.75,0]. One selected panel and one execution seed do not certify
relative training performance. The bootstrap only varies sequences.

**Can this fixed executor stably execute executable robot GT tau? No.** Reference
command replay is15/16, whereas standard GT is0/16 (empirical required rate75%).
**Is predicted tau clearly worse than executable GT under this same qualified
executor? Not established: the executor never qualified, and GT does not outperform
these predictions.** This is an execution-bridge blocker; it does NOT establish
that trajectory decoding is the primary system bottleneck. The v4-tau-trained
executor and the v6 adaptation remain the scope; no universal claim against all
possible tau executors is justified.

### Failure and recovery evidence

Labels overlap; these are observable proxies, not visual causal diagnoses.

| arm | no lift | no multi-finger force contact | push/tip while supported | loss after lift | early contact |
|---|---:|---:|---:|---:|---:|
| replay |0/16|0/16|3/16|9/16|0/16|
| standardGT |15/16|15/16|3/16|1/16|0/1 known,15unknown|
| healthy |14/16|8/16|10/16|1/16|2/8 known,8unknown|
| FM |15/16|14/16|4/16|1/16|0/2 known,14unknown|

Natural qualified-hold losses/recoveries: replay2/0, standardGT0/0, healthy0/0,
FM1/0. Recovery is0/2 and0/1 where opportunities exist; otherwise UNKNOWN.
Intentional lowering may trigger loss-after-lift. Forced perturbation and full
mesh penetration are **not measured**, not0. Net body forces are not object-specific
contact pairs. Finger-closure failure is a force-contact proxy, not a verified
joint-limit or tendon diagnosis.

### Interruptions and repairs

Initial FM attempt (`flow-failed-workspace/`) aborted the entire batch after one
state left the WAM voxel domain. Preserved evidence; no process from another job
was touched. The repaired evaluator tracks per-environment workspace failures and
skips predictions beyond each admitted prefix, then completed the same panel.
Final FM full300 diagnostic has2/16 workspace failures (ticks253/291), both after
those cases' prefixes104/141, so primary-prefix workspace failures0/16. Healthy
workspace field was absent in its loaded runner and remains UNKNOWN.

A standard GT definition check was added before interpreting generator quality.
The valid-from0 reference run is retained in `gt_reference-startup-valid/`; primary
`gt_reference/` reruns with actual trace [0,0,0] valid counts, step3 first valid.
Both yield0/16, so the startup correction does not rescue qualification.

Original thread reused GPU2 during execution; it was not signalled. Healthy used
GPU2 with that resource contention, finalFM GPU3; latency is not a fair model-speed
comparison. Recorded native host outputs and commands are in each manifest.
User protected GPU4/5/6/7 and all ongoing training were untouched.

### Inspect / reproduce

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/summarize_executor_qualification.py \
  --root outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1
```

`summary.json` contains thresholds, CIs, per-case denominators, time series source
paths, initial-state checks and native-engineering audits; `summary.md` is the
reading entry. Exact launch commands: `{replay,gt_reference,gt,healthy,flow}-process.json`.
Fresh output is required when rerunning; do not overwrite existing evidence.
Robot DEV split is certified against executor TRAIN; WAM lifetime pretraining
exposure is unavailable. The older direct-full-action qualification plan remains
untrained as such; these measurements concern the existing Scheduled686 bridge
and do not fabricate a completed new executor.


### Path attainment and timing details

Final mean confirmed path progress over16 is replay0.28463, standardGT0, healthy0,
FM0. Endpoint position failure (>5cm) among14uncensored cases is4/14,14/14,14/14,
13/14 respectively; each has2UNKNOWN/censored. Arriving near an endpoint does not
imply grasp or TaskController completion.

P0 runtime inference amortized latency (full unused C head included) is healthy
27.109ms/sample over4768 forecasts and FM57.992ms/sample over2268 forecasts. These
use different active-batch schedules and GPU contention, so are workload records,
not a controlled speed ranking. P1 has an independent batched latency logger.
Trajectory-only path parity (`trajectory-only-parity.json`) is0 difference with
same encoder activations for healthy andFM; fresh encoder repeats differ0.0381mm
and0.000536mm. No bitwise end-to-end determinism or model-speed conclusion claimed.
