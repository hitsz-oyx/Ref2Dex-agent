---
schema: ref2dex.probe.v2
probe_id: P-20261020-inspire-measured-official-ppo
experiment_id: P-20261020-inspire-measured-official-ppo
date: 2026-10-20
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 14
seed_pool: probe
seeds: [246, 247, 248]
decision_changed_if_positive: continue the measured goal WAM policy improvement route
decision_changed_if_negative: diagnose measured training dynamics before adding scale or changing the method
status: RUNNING
run_id: inspire-measured-official-20261020-r2
---

User requests starting PPO now onGPU4/5/6 concurrently with independent
trajectory visualization onGPU7. This is an explicit resource allocation for
this run, not a general change to CAMPAIGN. Main branch, frozen code manifests.

Decision: can the actually measured goal+WAM observation support policy
improvement after latest BC? The user requests enough training scale, so a
two-env wiring check is only a Blocker check, not a substitute for training.
Then three ranks with512 environments each,1536 total, matching the previous
Shadow six-rank256 total and global12288 transitions per eight-step rollout.
Global minibatch3072 after four minibatches; five epochs. Use pinned official
UniDexGrasp2 PPO collector/storage/GAE/update, actual-distribution KL correction,
global advantage statistics and gradient averaging already used by Shadow.
No DExplore PPO, reward or hand-reference actor input is used.

Source: corrected r3 teacher-rollout admission, all501 TRAIN sequences/50 objects.
Tasks/events derive from teacher object rollouts. BC source is corrected-data
Scheduled686 DEV-best(step0), retaining learned Current421 goal columns; this
is the declared DEV selection, not post-hoc endpoint selection. Added tau
weights are zero initially but trainable during PPO, with actual WAM80000 input.
CAD encoder is frozen at the BC parent so rollout features do not drift between
sampling and update. Actor/critic MLP1024/1024/512/512 ELU; actor BC initialized,
critic fresh as Shadow; native12 actions/30Hz remain Inspire-specific.

Official optimizer settings: lr3e-4 adaptive desiredKL.016, clip.2, gamma.96,
lambda.95, entropy0, gradient norm1, initial noise_std.8 (preserve upstream
squared-scale sampling convention). Reward is existing causal MeasuredRewardTracker
with the same gamma. Full TaskController task, not imitation of hand references.

Reset contract: execute each action in its observed episode, collect terminal
reward/done, explicitly reset only completed native environments, clear only
their task/reward/history/previous-action state, and return the new observation
before sampling the next action. Actual native clock/episode IDs are checked.
Two-env probe forces different environments to terminate atsteps2/4, then real
official PPO updates; training never uses this engineering forced termination.
Terminal GAE does not bootstrap from a reset state.

Run limits:10k PPO iterations or48h, own STOP at update boundary,20GiB disk
floor, bounded checkpoint retention (last3 plus500 milestones). Rank0 only
saves checkpoint; other ranks log progress. Physics and policy subprocesses
share their assigned GPU; no unknown processes killed. All native motion
sampling is restricted to the admitted TRAIN membership. Runtime graspenv
Torch2.4.1+cu121 policy and pinned dexplore_v120_train Torch2.0.1+cu118 physics.
Log first iteration throughput/memory/ETA and actual multi-episode reset results.
Training success is full TaskController completion, kept distinct from lift
proxies; no WAM gain or held-out generalization claim from this run alone.

## Startup failure and backend isolation

The r1 multi-rank GPU PhysX run failed before its first PPO update with
`convexPlaneNphase_Kernel` illegal GPU access; its logs remain in the r1 output.
The two-env asynchronous-reset engineering test passed2 iterations/40 optimizer
steps, zero source-episode mismatches and zero behavior log-prob replay error.
Native screens r1/r3 include successful full50-object512-env single-rank two-update
runs, but also failed256-env and partitioned512-env cases. This is not evidence
of a simple environment-count/OOM boundary. Explicit synchronization did not
eliminate the error; do not claim the GPU backend fixed.

r2 tests CPU PhysX (8 physics threads/rank), with the same1536 environments,
TRAIN membership, physical parameters, native actions, reward and official PPO
settings. WAM and PPO stay on the requested GPUs4/5/6. Upstream hardcoded CUDA
motion caches are moved to the CPU sim tensor device after creation. This is an
explicit physical-backend change, not a strict GPU-PhysX reproduction. Test the
same asymmetric indexed reset before beginning its three-rank training.

WAM batch256/shared TF32 is checked against FP32 for every full trajectory in
the first inference batch; failure above1mm aborts. A full512-env single-rank
screen measured0.377mm maximum,30.38s collection and0.17s optimization per
iteration, so physics-only timing is not total training throughput. Backend and
inference settings now form part of the strict resume identity. r2 manifests
record the frozen code revision and actual progress; startup or environment
creation alone is never counted as a PPO update.

CPU PhysX full50-object512-env screen r5 also completed2 iterations/40 optimizer
steps,31.31s collection and0.20s update on its second iteration, source mismatches0.
CPU two-env r2 smoke completed the same asynchronous reset fixture with0 source
mismatches and0 log-prob replay error. These verify startup and handoff, not policy
improvement or physical equivalence between CPU/GPU solvers.

## Three-rank startup accepted

r2 frozen code ed62f767 passed the CPU asynchronous-reset smoke and began actual
three-rank updates. At acceptance each rank completed at least2 PPO iterations /
40 optimizer steps; parameter synchronization error0, source-episode mismatch0,
behavior log-prob replay error0. All ranks bind501 TRAIN motions/50 objects.
One natural workspace terminal was reset in this initial interval; this is not
a task-success result. WAM first-batch full-trajectory TF32 errors were
0.330/0.285/0.548mm (ranks0/1/2), below1mm.

Rank0 saved official-iteration-1.pt (61,158,108 bytes), completely loaded on CPU
with finite model/CAD tensors and optimizer state. Second-iteration throughput
was approximately32s per global12288-transition iteration.10k would take about
89h at that rate, exceeding the explicit48h run budget; the budget stops/saves
first, and a subsequent run may resume. No convergence or gain claim yet.

Startup resources are recorded separately in the output's startup-resources.json;
progress/curve files belong to each rank, and bounded checkpoint writing is
rank0 only. All154 Task tests pass. Global tools/verify.py still reports only
pre-existing missing metadata in six trajectory-policy cards; do not describe
the whole repository verification as passing.

Thirty1Hz startup samples measured mean GPU utilization32.6/36.9/46.9% on
physical GPUs4/5/6, peak total allocation11,480MiB per card. Utilization remains
intermittent; the measured causal state/task packing and WAM calls dominate the
collector, so high GPU utilization is not yet achieved. Four synchronized
iterations/80 optimizer steps per rank were reached during this sampling.
