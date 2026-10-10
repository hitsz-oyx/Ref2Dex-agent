---
schema: ref2dex.probe.v2
probe_id: P-20261010-startup-balanced-actor
experiment_id: P-20261010-startup-balanced-actor
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: c7d991b
claim_id: C3
hypothesis_family: HF-trajectory-policy-learning
probe_index_in_family: 2
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: use startup-accurate pure-H initializer for trajectory PPO
decision_changed_if_negative: stop BC weighting search and reassess RL initialization
status: UNPROMISING
run_id: startup-balanced-actor-20261010-r1
---

# Does startup-balanced BC repair the independent actor initializer?

## Motivation / Decision Note

First H-only initializer failed grasping after a valid H/c/D/R audit. Initial
H equals its training input, with consistent labels, but prefix handRMS56.36mm;
feedback/distribution shift cannot explain all of this startup error. Tick0 is
only0.1845% of uniform6504training samples; global val selection improving80%
missed startup. Cheapest next distinction: one fixed offline sampling/selection
rebalance, keeping H/D/R/network/dataset/normalization/optimizer/budget unchanged.
This serves Mission A initialization, not evidence that WM or PPO works.

## Protocol

Same full metric-native source wave and hand-only-derived geometry as
[P-20261010-history-trajectory-actor](P-20261010-history-trajectory-actor.md):
all16rows and failed states retained, same12/4 split,6504train/2168val samples,
H328 past/current only, standalone c288, no online reference-action base.
Same seed295, MLP512x2/Tanh, train-only normalization,2500Adam updates/batch512/
lr3e-4 and pose/nominalFF objective. Invoke fit_history_actor.py --startup-balanced.

Each batch draws128uniform samples from source ticks0:8,128from8:64 and256from
all542ticks (12train rows each). No stage/clock is actor input; source tick
indexes only offline sampling and supervised evaluation. Selection criterion:
.25 val loss0:8 +.25 val loss8:64 +.5 val lossall. Same checkpoint candidates
1/250/.../2500, save selected checkpoint once. This changes both training
sampling and selection; do not attribute differences to one alone.

Before physics, require first-query (tick0, allfour val rows) first8 decoded
hand3D RMS<=5mm, palmmaximum<=10mm and worst-row nominal wristXYZ FF3D RMS<=10mm.
These are initialization coverage, not grasping or generalization. If missed,
no new physics and local UNPROMISING for this fitting variant. If passed,
exactly one native16env/542controls/seed296 wave using the same H actor/GT/dense
roles and frozen D/R. Unchanged >=433held+terminal and<1%actor clipping screen
from previous card. Replay H/actor c/D/native chain/outcomes before any positive
claim. Runtime actor matrix precision is native rl_games high/TF32-on; audit
matches actual runtime, training remains strict FP32. No precision intervention.

## Resources / stop

Single idleGPU4, bounded new10GPUmin/512MiB: fit<=240s, conditional native<=300s,
audit<=60s. GPU training/FK/PhysX/inference, CPU files/tiny contracts/statistics.
Memory/util/ETA monitored in fit/native. Stop on nonfinite/drift/foreign compute/
unsafe occupancy/reset/timeout. Preserve all previous data/checkpoints, no retry,
extra seed/motion/WM/PPO or coefficient sweep. OldTask paused; main/no branch/push.

If this also fails startup coverage, stop BC weighting search. Next route needs
an explicit bounded decision about practical RL initialization/curriculum,
not indefinite BC tuning or a formal refutation of H-only trajectory policies.
No Mission/claim or external authority change.

## Results

Code `c7d991b`, seed295,2500updates completed13.143s onGPU4. Later utilization
11--23%,399MiB, Torch peak59.10MiB; no owned process remains. Selected update500
per the frozen combined val criterion. Train6504/val2168 and allsource rows
unchanged. Weighted val objective0.81043 vs mean19.68379 (95.883% lower); its
weighting differs from previous card, so raw objective values are not matched.

First-query four-val-row prefix hand3D RMS7.39460mm, palm maximum15.60908mm,
worst-row wristXYZ nominalFF3D RMS5.46399mm. First two miss the fixed5/10mm
limits; FF meets10mm. Across all68x4 row-val replans, prefix hand RMS35.13388mm
and palm maximum204.72589mm. Startup improved relative to prior same-H FP32
coverage56.369mm, but did not meet the declared initialization gate. Local
**UNPROMISING** for promoting this fitted initializer under this protocol.
No native simulation, no new grasp-failure evidence, no PPO or WM in this run.

Artifacts: `outputs/trajectory-policy/startup-balanced-actor-20261010-r1/`
contains manifest/result/best.pt/coverage.npz/monitor.jsonl. Checkpoint was
written once; prior uniform-fit and native/audit outputs unchanged.

## Follow-up Decision Note

Honor the stopping rule: no more BC weighting/epoch/width sweep. Evidence now
separates representational coverage (D288 exact) and initializer learning:
independent H-only BC retains substantially more startup geometry after one
rebalance, yet startup remains imperfect and rolling execution is unproven.
Neither this offline gate nor the first BC zero-grasp run tests task RL.

Next bounded research decision: can real-reward closed-loop trajectory PPO
improve this independent actor, with D/R fixed and no WM? Use balanced500 only
as a declared supervised warm start, never as an already successful policy.
Compare to its frozen mean under the same reset/task reward/evaluation contract;
collect from true frame0, no teacher action substitution or reference reward.
First freeze reward from current measured proximity/lift/held/loss and native
control cost, plus chunk transition discount/GAE/logprob semantics; separate
V(H) from any action-Q/WM. A new PPO card must declare concrete GPU/interactions/
stop bounds before training. Small mean/std updates and audited actual chunks
will distinguish learning signal from execution/credit-assignment errors.
If early PPO has no useful signal, diagnose which physical events/reward paths
occur; do not claim PPO or trajectory RL refuted. This changes implementation
phase, not Mission/Cm claim or external permissions.

## Limitations / future evidence

Same-motion row split and oracle hand-only offline labels, frozen warmstarted R;
not independent demonstrations. Startup geometric pass does not ensure rolling
execution or longhold. Positive still needs actual trajectory PPO and matched
Cm-on/off strategy-training utility; multi-motion generalization deferred.
