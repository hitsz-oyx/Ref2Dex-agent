---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ddp-scale
experiment_id: P-20261012-goal-point-wam-ddp-scale
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 04689d66
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 14
seed_pool: probe
seeds: [275, 276, 277, 278, 279]
decision_changed_if_positive: retain scaled hand-memory candidate for action-world reliability and downstream policy work
decision_changed_if_negative: inspect learning curves and forward-action/input semantics before additional scaling
status: UNCLEAR
run_id: goal-memory-ddp-20261012-r1
---

# Single-model four-GPU Goal-Point-WAM scale-up

Class: Decision. Serves the Task's learned goal-conditioned hand representation
upstream of Mission's robot/Cm/RL claim. Prior hand+memory goal reduced trajectory
EPE and goal ablations on GRAB/ARCTIC. This run asks whether that candidate keeps
learning with broader exposure and yields a useful forward model. User explicitly
replaces the proposed six matched seed/control launches with ONE model on FOUR
synchronously distributed GPUs, maximizing useful GPU utilization. No six-arm
training started; their unused configs were removed. The cancelled panel449 was
generated but no outcome was observed, and is not used by this Probe.

Result: UNCLEAR; engineering preflight passed, scientific launch pending. No formal causal or multi-seed claim is possible.
Decision: Run the selected architecture at larger global batch and fixed budget;
use final real/static goal and world/action diagnostics to choose the next step.

## Fixed method and budget

- Fresh strict PW50000 scene/projection transfer; current WAM heads, object
  projection and observed-hand adapter freshly initialized with seed275.
  Current-hand XYZ/history + memory object goal32 paired points, coordinate
  scale.05m, world_weight1. No robot, video, tactile auxiliary or L_goal.
- NCCL DistributedDataParallel,4 ranks on idle physical GPUs0/1/4/5, all on
  the same NUMA node. GPU topology avoids GPU7's cross-socket path. Each rank
  microbatch128; global batch512, accumulation1,10000 optimizer steps,
  5,120,000 sample occurrences from the existing4,378,478 train-window corpus.
  Source weights5/9,2/9,2/9, existing category stratification, draw seed276.
  Occurrences are partitioned by update/rank; there is no separate model/seed.
- Same initial model on all ranks, confirmed by parameter SHA; rank-specific
  stochastic serialization stream after initialization. DDP averages gradients
  of locally masked WAM losses; this is a mean of rank-local objectives rather
  than a monolithic global-valid-count mean. No old/new matched-batch claim.
  broadcast_buffers=false follows the existing PW distributed substrate;
  find_unused_parameters=true covers dormant query adapter parameters.
- BF16 forward/FP32 physical loss, TF32, AdamW lr.001,500 warmup, cosine to10%,
  decay.01, gradient norm cap1 after synchronized backward.4 loader workers/rank.
  GPU queries every100 steps, per-rank progress every10, all-step local logs.
  No synthetic initialization, adaptive LR, extra seed, control or silent batch
  reduction. OOM requires a recorded resource-only adjustment and fresh launch.
- Evaluation1536 unique windows,512/source, seed277. Freeze actual index/hash
  before launch; exclude same-sequence temporal frame overlap with previously
  viewed compatible Task panels, regardless of anchor (abs(tick difference)>=28).
  Windows may overlap each other and reuse val sequences/PW selection data;
  these are fresh temporal windows, not untouched subjects or a final benchmark.
  Evaluate every1000 steps, all4 ranks, aggregate by actual valid-point/object
  counts. Fixed final10000 for conclusions; best.pt is descriptive only.
- The learning decision is exploratory: at final, goal beats its own static-goal
  ablation and zero hand by>=5% on>=2 sources => PROMISING for this candidate;
  otherwise UNCLEAR. No no-goal control/multiseed route superiority claim.
  World moving/static and action sensitivity need separate audit before L_goal;
  no amount of training loss alone proves world or robot usefulness.

## Resources / stopping

The latest user explicitly authorizes exactly four distributed GPUs, consistent
with Campaign's concurrent cap. Foreign tasks are never modified. Memory128
single-rank preflight measured15.65GB allocated/18.62GB reserved; synchronized
compute.882s/update. Four-rank small smoke and full-batch smoke must both pass
before scientific launch, including finite losses and final parameter equality.
Small smoke4 updates completed with all four final parameter SHAs identical.

Max6h wall/1440GPUmin,6GiB new artifacts including smoke and rolling checkpoints;
aggregate outputs+tmp must remain<300GB (prelaunch279.9GB). Stop on nonfinite,
OOM, input/config/source drift, foreign conflict or budget/deadline. Owned ranks
accept SIGUSR1 at the next100-step synchronized stop boundary and save their
state; torchrun owns its four workers. Never kill unknown processes. Checkpoints
store all-rank RNG and optimizer for same-config/world-size resume. Only own
rolling latest/best names are atomically replaced; final/best use hardlinks to
checkpoint snapshots to bound disk. Earlier runs/checkpoints remain untouched.

## Artifacts / limitations

Output root: `outputs/goal-point-wam/goal-memory-ddp-20261012-r1/`.
`evaluation/panel.json` binds the frozen panel and excluded history; training
subdirectory will hold global/per-rank indices, config/input identity,
per-rank logs/progress, validation curve, optimizer checkpoints and result.
Fixed code commit and launcher/PIDs are recorded before/at actual launch.
Raw sparse forward-repeat noise remains a separate engineering issue; final
fixed-input world independence audit cannot prove physical accuracy. Any
positive stays single-seed local evidence; multi-seed and matched causal tests
are explicitly absent per user instruction. Scaling alone does not complete
Mission's Cm-on/off RL requirements.

## Prelaunch engineering evidence

Four-rank full-batch128 smoke also completed4 updates, global batch512, with
identical final parameter SHA on all4 ranks. Warm steady update time.95--.96s;
peak observed reserved memory about17.4GiB. Small and full-batch smoke are
engineering-only, not candidate accuracy evidence or scientific initialization.
A read-only independent review confirmed gradient synchronization, occurrence
sharding and count-weighted evaluation. It found resume RNG restoration before
DataLoader iterator creation consumes an extra CPU base seed; restoration was
moved after iterator creation before the fresh scientific launch.

The supervised launcher reuses the existing PointWorld ownership checks,
records exact source hashes/PIDs and polls every30s. It requests only owned
workers save/stop on source drift, foreign context,298GB artifact threshold or
6h deadline. Max180s cleanup grace; no foreign PID is signaled.
