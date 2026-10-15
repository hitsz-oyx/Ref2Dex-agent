---
schema: ref2dex.probe.v2
probe_id: P-20261015-full-teacher-input
experiment_id: P-20261015-full-teacher-input
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 440f9c13
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 1
seed_pool: probe
seeds: [180, 181, 162]
decision_changed_if_positive: replace teacher future reference blocks incrementally with tau after qualifying the supervised full-input mapping
decision_changed_if_negative: diagnose fitting and same-execution labels before attributing failure to WAM or tau
status: RUNNING
run_id: full-teacher-input-20261015-r1
---

# Full-teacher-input supervised upper bound

Class: Decision. User explicitly requests actual DExplore1442D delta1+delta16
without tau, supervised on the same q_PD labels. Serves Mission autonomous
execution by distinguishing information mismatch from fitting/data problems.
This is an offline diagnostic, not a learned grasp success claim.

## Data and engineering gates

Use the exact native-current660 rollout, retained TRAIN110250/dev12049 windows,
original sequence split and observed-fixed-grid admission; no new simulation,
no joining contacts from other executions. Retain previous native-current
TRAIN/dev512 panels (seed162, half startup), plus source sequence/tick identity.
Only one current control q_PD[t] is predicted, unlike previous8-step Chunk-R.
Never directly compare new versus old loss as a single-factor information test.

The source stored delta0 only but also saved fullbody state, forces, current
reference428 at every tick. Reconstruct delta1+delta16 by compiling the two
original pure tensor observation functions; preserve measured current IG from
saveddelta0, recompute reference IG differences for each future reference.
Reference clamp is min(t+delta,T-1), matching same-capture terminal repeat-last.
Use pinned Torch2.0.1; metadata admission is exported separately from graspenv
because the pinned simulator runtime does not include spconv. No simulator is
created. Cache stored in full-teacher-cache-20261015-r1, raw source unchanged.

Fail closed per sequence: reconstructed delta0 all721 maxabs<=1e-4; frozen
original teacher (original ReLU MLP, external RMS epsilon1e-5/clamp[-5,5],
clamped deterministic mu) reconstructs same-tick applied action maxabs<=2e-4
and wristXYZ PD error max<=0.2mm. Teacher weights are used ONLY for this audit,
never to initialize or train any student. Capture called get_action(obs,True),
so this dataset's teacher actions are deterministic, not sampled policy noise.
PD-label/action/q same-tick reconstruction was already independently audited.

## Four bounded arms

All students are random-initialized MLP1442→1024→1024→1024→512→12, ReLU, no tau.
Same seed180, matched drawseed181, AdamW3e-4, decay1e-4, batch512,
clipgrad1, standardized single-step active12 PD L1, TRAIN-only input/output
statistics, scale floor1e-3. Current721 is padded with721 zeros to preserve
exact capacity; corresponding normalization leaves masked values zero.

- full_absolute: actual1442→absolute ACTIVE12 q_PD, literal user upper bound.
- full_relative: actual1442→q_PD minus currentq for first6, absolute independent
  fingers; fixed decoder adds measured currentq. q is not a network input.
- current_relative: captured delta0 721, same relativePD head and decode.
- memory_full_relative: same1442 relative model, only fixed TRAIN panel's unique
  windows, 4000 updates; tests fitting on a small subset, not generalization.

The relative arm is necessary because teacher input contains local geometry/
reference errors without a guaranteed world absolute root coordinate. Native
teacher itself computes q_PD=currentq+controller(action) for first6. Failure of
absolute but success of relative must not be mistaken for poor data or WAM.

Three main arms each12000 updates, one budget; evaluate TRAIN/dev512 every500,
select dev wristPD XYZ p95 then mean; memory selects training p95 then mean.
Report selected/final mean/p95, raw PD and executable clamped PD errors,
rotation/finger p95, clipping, curves and selected predictions/normalizers.
Matching observation-precision teacher replay is an engineering gate only.

Mapping fit is locally PROMISING if full_relative dev mean<=20mm; information
mismatch is only locally PROMISING if also >=50% lower than matched
current_relative mean (p95/tails reported separately). If both fit well, do not
attribute old Chunk-R failure to missing future information alone. If memory
fits but dev does not, prioritize generalization/distribution; if memory also
fails, optimization or model fitting remains unresolved. No formal causal or
information-theoretic claim follows from a single-seed Probe.

## Resources and stops

GPU0 cache reconstruction<=1200s, then GPU0/1/2/3 four students in parallel,
three main arms<=1200s each and memory<=1200s/4000 updates; <=4 simultaneous
GPUs, global cap6. Cache<=2GB, each arm<=768MB, aggregate<=6GB;
filesystem>=20GiB. Stop on nonfinite, teacher replay failure, foreign assigned
GPU consumer, budget/time violation. Source/checkpoints unchanged; no official
student initialization and no WAM inference. Record actual code, PIDs,
utilization, memory, elapsed and ETA in each output manifest/progress.

## Engineering smoke

Original metadata loader in pinned runtime failed before computation due to
missing spconv; split CPU admission export from pure native observation cache.
Two train+two dev sequences passed: delta0 max2.38e-6, teacher action max5.66e-7,
PDXYZ max0.000224mm. This demonstrates correct reconstruction for these sequences
only; full corpus must pass per-sequence before scientific fitting.

## Limitations / future evidence

Repeated dev use, one seed, coarse source quality gate, no physical execution.
A1442 win over the old tau model also changes architecture/output horizon;
only matched721vs1442 relativePD isolates the bundled future reference change.
It does not identify t+1 versus t+16 contributions or contact specifically.
Substitution of teacher reference blocks with tau and physical deployment are
follow-ups only after interpreting this upper bound; not included here.
