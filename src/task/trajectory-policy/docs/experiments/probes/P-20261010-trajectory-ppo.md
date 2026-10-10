---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-ppo
experiment_id: P-20261010-trajectory-ppo
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-learning
probe_index_in_family: 3
seed_pool: probe
seeds: [293, 294]
decision_changed_if_positive: retain actual task trajectory RL baseline and evaluate WM training intervention
decision_changed_if_negative: audit reward events and actual PPO credit/execution before further training
status: UNCLEAR
run_id: trajectory-ppo-20261010-r1
---

# Can real-task PPO improve the independent trajectory actor?

## Motivation / Decision Note

Mission A needs a self-trained physical manipulation policy, then Mission B
matched Cm training benefit. Stop48D compression and BC weight sweeps. D288
coverage is exact, H-only uniform BC failed grasp, startup-balanced BC improved
initial geometry but missed its coverage gate. Neither tests task RL. Use
balanced500 as a declared imperfect warm start, not a successful initializer.
Keep D/R/H fixed, no WM/reference-action base, and test actual reward learning.
Small on-policy Probe is cheaper than more BC/architecture/data variants.

## Fixed protocol

Initializer `outputs/trajectory-policy/startup-balanced-actor-20261010-r1/best.pt`,
frozen R SHA79ee282e6b2287375c6480eb8f793864c858311b968ffa97b95c65842502fa45.
Same single airplane motion/reset0 and native gains. Actor reads H328 only
four measured states (pad at reset), emits independent Gaussian c288 once every
8controls; D decodes24future hand frames and R uses actual feedback everystep.
No true future q/object/hand, actor obs, reference reward, clock/phase/force/WM.
Future references only exist as environment data; online training extracts
only initial q from the old packet to audit reset. Loader-owned actor is never
queried for actions. D/R weights/normalization are frozen. Actor mean and
logstd are trainable; Gaussian raw c density is used before deterministic D
bounds, one joint logprob per high-level sample, no low-step probability copies.

Task reward from actual after-step geometry: .2 exp(-2*unsignedsurfacegap),
+.5 clip(lift/.2,0,1) ifgap<=.01, +1 ifnear andunsupported andlift>=.03,
-.5 on held->lost, -.1 native clipping, -.01 mean squared native action change.
Lift is current object center z minus its measured reset z. Supported means
actual table footprint andabs supportgap<=.02. No tactile/force measurement.
Native reward discarded. Evaluate long-held/terminal separately from shaped
return; proximity or lift reward alone does not prove stable grasping.

Episode horizon542controls, always genuine frame0. Treat this declared task
end as terminal; disable native earlytermination/adaptivekappa. Last prefix is
6controls, otherwise8. R previous residual/action cost state and all four actor
history states reset correctly. High reward=sum gamma^j r_j, bootstrap gamma^k,
GAE factor(gamma*lambda)^k, gamma.99/lambda.95 per native control. Rollout boundary
bootstraps V(Hnext), task terminal masks it. V(H) separate256x2 Tanh, no action
orphysical token. Actual native precision forced highest/TF32-off after Runner
initialization, recorded; future frozen comparison uses same precision.

Train seed293,16env,24updates x16high transitions; <=49152actual environment
interactions (shortterminalprefixes reduce count). Full-batch4epochs PPO clip.2,
actor Adam lr1e-6, value Adam lr3e-4, maxgrad.5/1, backtrack actor proposals until mean joint
Gaussian KL<=.02 (factor.25, atmost4trials; restore actor/Adam moments on rejection,
retain reduced LR; stop epochs if no feasible proposal); logstd clamp log(.01):log(.2). No entropy/behavior-prior/WM terms.
Before updating, behavior logprob replay max must<=.02 to catch precision/layout
errors. Save every rollout's pre-update actor/value snapshot and final checkpoint
once; all high H/c/distribution/logprob/duration/value/reward/GAE/returns plus
actual low measuredstates/Rinputs/requested/applied/PD/reward components preserved.
Only final24 checkpoint evaluated, no best-rollout/cherrypick/seed sweep.

One debug9 engineering smoke uses2updates x2chunks and artificial14control
horizon (8+6) solely to check terminal/reset/GAE/optimization/wiring. No grasp
or science claim from this smoke. It cannot substitute the full542control task.

After successful engineering and finite full training, one16env seed294 frozen
542control wave: GT tau,dense FK,frozen balanced mean,final PPO mean4each,
randomizedroles; both learned roles observe only their own pure H, identical D/R,
reset/replan/native precision. Calibrated ifGT/dense each>=3/4 >=433held+terminal.
PPO learning signal PROMISING ifPPO>=3/4, clipping<1%, and at least2more longheld
than warm start. Bothlearnedroles0/4 withcalibration UNPROMISING; otherwise
UNCLEAR for improvement. IfPPO>=3/4 regardlesswarm performance, report baseline
usefulness signal separately; no causal PPO/Cm claim from a saturated comparison.
Actual trace/reward/GAE/logprob/parameter-change/execution/outcome audit required.

## Resources / stop

Single idleGPU4, new14GPUmin/512MiB: debug smoke<=120s, training<=480s,
one frozen evaluation<=180s and audit<=60s. GPU neural optimization/FK/PhysX,
CPU file/statistics/tiny contract tests. Monitor perupdate GPU utilization/memory/
ETA, foreignprocess guard; control finiteness/requested-applied checks everystep.
Abort unexpected reset/drift/nonfinite/logprob failure/resource conflict/timeout;
preserve outputs, no automatic restart or extra policy/seed/curriculum run.
OldTask stayspaused, main/no branch/push, no external writes/process changes.

## Results

Formal Probe not run yet. Engineering smoke r1 (`57d9390`, debug9) completed
13.398s,448interactions on an artificial14control horizon,8/6duration/reset masks
correct; actor parameters changed and behavior-logprob replay<=.00055. It found
vanishing exp(-50gap) reward (gap1.03--1.37m, proximity<=8.76e-24) and first-step
joint KL1.67/1.07 despite early stopping. This is engineering evidence for fixing
reward scale and trust control before scientific training, not policy failure.
Preserve `outputs/trajectory-policy/trajectory-ppo-smoke-20261010-r1/`.

Pre-training repair: physical distance kernel exp(-2gap), no reference terms;
finite KL proposal backtracking, fixed no-accept rollback. Same Gaussian action,
D/R/H, horizon and task metrics. Smoke r2 must verify these changes before the
single formal training run; totalengineering cost remains within120s cap.

## Limitations / future evidence

Single motion/seed, small interaction budget, imperfect BC warm start and frozen
executor; stochastic training and deterministic evaluation differ. Existing
hand-only offline supervision cost is separate from on-policy samples. This
Probe is not RL superiority or stable-grasp Validation. Need matched native PPO,
repeat held/drop checks and finally matched Cm-on/off policy-training benefit.
Reward proxy can be exploited; held and terminal checks remain independent.
