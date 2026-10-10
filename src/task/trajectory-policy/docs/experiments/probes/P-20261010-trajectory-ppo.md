---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-ppo
experiment_id: P-20261010-trajectory-ppo
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 4da53a9
claim_id: C3
hypothesis_family: HF-trajectory-policy-learning
probe_index_in_family: 3
seed_pool: probe
seeds: [293, 294]
decision_changed_if_positive: retain actual task trajectory RL baseline and evaluate WM training intervention
decision_changed_if_negative: audit reward events and actual PPO credit/execution before further training
status: UNPROMISING
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

Engineering smoke r1 (`57d9390`, debug9) completed
13.398s,448interactions on an artificial14control horizon,8/6duration/reset masks
correct; actor parameters changed and behavior-logprob replay<=.00055. It found
vanishing exp(-50gap) reward (gap1.03--1.37m, proximity<=8.76e-24) and first-step
joint KL1.67/1.07 despite early stopping. This is engineering evidence for fixing
reward scale and trust control before scientific training, not policy failure.
Preserve `outputs/trajectory-policy/trajectory-ppo-smoke-20261010-r1/`.

Pre-training repair: physical distance kernel exp(-2gap), no reference terms;
finite KL proposal backtracking, fixed no-accept rollback. Same Gaussian action,
D/R/H, horizon and task metrics. Smoke r2 must verify these changes before the
single bounded Probe training run; totalengineering cost remains within120s cap.

Repair smoke r2 (`b1bdb20`) completed13.423s/448interactions: jointKL .013974
and .001818, four accepted actor steps each, replaylogprob max .000336,
actor parameter L2 change9.68e-5. Proximity .0130--.0255; duration8/6 and
terminal/reset masks correct. Engineering passed; no grasp inference from14steps.
Before formal training, preserve actual rollout-end H alongside V bootstrap to
permit independent snapshot/value replay; this adds trace data only.

Full training `4da53a9`, seed293 completed264.032s:24updates,96accepted
actor steps,3062controls x16=48992interactions, clipping1.0267%, actor mean
parameter L2 change .00094752. JointKL .001723--.017811; mean Adam LR after
initial backtracking remained6.25e-8. Five complete542control waves plus352
controls. 2416held steps; one completed row held486consecutive and terminal,
one held471 then dropped. These exploratory sampled trajectories do not
establish a stable policy or improvement over the warm start.

Frozen `054a78c`, seed294 completed56.411s,542controls: GT3/4 longheld+terminal,
dense3/4, warm_start0/4 andPPO0/4 (both zero held frames). Warm/PPO clipping
.3229%/1.1531%. All learned rows reached gap<=.01 atcontrol42, then lost proximity
betweencontrol52--62; thus this is failure to acquire stable grasp, not successful
reference placing. Final deterministic mean does not preserve occasional long
holding from stochastic training. Local protocol conclusion **UNPROMISING**;
not a refutation of trajectory PPO in general or a Cm conclusion.

Training replay `054a78c` r1 exact H/behavior/value/bootstrap and then `c7eab30`
r2 expanded all6144 sampled c: independent XYZ/SO3/native Euler/finger coupling/FK
q/hand exact, velocity5.72e-6. GAE max1.22e-5, reward1.19e-7, executor inputs
5.72e-6, latent2.91e-7, command/PD exact. Each training audit<=6.36s.
Frozen execution audit `054a78c` H/c/D/FK exact, Rinput3.81e-6, command2.38e-7,
PD0, outcomes and predeclared classification agree. No physical rerun.
11Task tests passed; the first test command omitted task PYTHONPATH and failed
collection, corrected command passed (no code change or scientific impact).

Artifacts (preserved, no overwritten checkpoints):
- `outputs/trajectory-policy/trajectory-ppo-20261010-r1/`: full training and24 snapshots.
- `outputs/trajectory-policy/trajectory-ppo-training-audit-20261010-r1/`: first credit/control replay.
- `outputs/trajectory-policy/trajectory-ppo-training-audit-20261010-r2/`: expanded sampled-decode replay.
- `outputs/trajectory-policy/trajectory-ppo-execution-20261010-r1/`: frozen comparison.
- `outputs/trajectory-policy/trajectory-ppo-execution-audit-20261010-r1/`: outcomes/control replay andbehavior.png.
Training artifacts282.79MiB; remaining artifacts within512MiB experiment cap.

## Next decision

Independent read-only review under AGENTS14 found no terminal, input, precision
or reward/GAE implementation error that invalidates this negative Probe. Initial same-H actor XYZ mean change from warm tofinal is only .04964mm
RMS across24frames; startup geometric discrepancy was7.395mm. This makes
insufficient practical policy movement plausible, without establishing cause.
Also stochastic versus mean deployment differs. Before any further simulation
or training, distinguish sampled exploration and update scale using existing
rollouts and checkpoint snapshots; no more BC weight/48D decoder sweep.
Seven sampled rows with>=45consecutiveheld all had negative batch-normalized
advantage on their episode-start chunk. The documented per-control lambda.95
propagates credit by(.99*.95)^60=.0252. Keeping all stored rewards/value/masks/
bootstrap fixed, lambda1 makes sixofseven startup advantages positive. This
suggests a trace-time-scale problem; sample-local surrogate sign cannot prove
the shared-network update decreases each sample's actual probability.

Post-hoc **Decision diagnostic** (no new physics/training budget): replay the
four recorded rollout snapshots containing allseven startup rows, matched
lambda.95/1 arms with same H/c/V/reward/done/bootstrap/learningrate/KL budget.
Use fresh Adam in both arms because optimizer moments were not saved; this is
not exact historical optimizer reconstruction. Each arm calls the same production
PPO function with only GAE lambda overridden locally; no edits to frozen source.
Measure actual startup sampled-action logprob changes and prefixXYZmean movement.
GPU4,<=30s within remaining audit allocation,<=16MiB; files/statsCPU.
If lambda1 raises successful-startup probability more consistently while KL<=.02,
next fixed-budget task Probe changes lambda only; otherwise reassess before
spending simulation. Allseven rows retained, no best-checkpoint/seed selection.
Keep Mission claim andoldTask pause; baseline andWM training benefit incomplete.

## Limitations / future evidence

Single motion/seed, small interaction budget, imperfect BC warm start and frozen
executor; stochastic training and deterministic evaluation differ. Existing
hand-only offline supervision cost is separate from on-policy samples. This
Probe is not RL superiority or stable-grasp Validation. Need matched native PPO,
repeat held/drop checks and finally matched Cm-on/off policy-training benefit.
Reward proxy can be exploited; held and terminal checks remain independent.
