---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-ppo-long-credit
experiment_id: P-20261010-trajectory-ppo-long-credit
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 0487233
claim_id: C3
hypothesis_family: HF-trajectory-policy-learning
probe_index_in_family: 4
seed_pool: probe
seeds: [293, 294]
decision_changed_if_positive: retain longer-credit trajectory PPO baseline and verify the physical mechanism before WM
decision_changed_if_negative: inspect actual policy movement and stochastic exploration before further training
status: UNCLEAR
run_id: trajectory-ppo-long-credit-20261010-r1
---

# Does longer credit improve actual trajectory policy learning?

## Motivation / Decision Note

Mission A still lacks a usable independent H->trajectory policy; Mission B
still requires matched Cm policy-training gain. Uniform BC, startup-balanced BC,
and first short task PPO did not improve the stable-grasp scoreboard. Stop BC
weight/48D sweeps. First PPO source/credit/physical execution audits passed,
so do not relabel its negative result as an implementation bug. Training sampled
one486frame terminal-held row, but final deterministic policy remained0/4.
Read-only independent review found allseven >=45held sampled episode startups
had negative normalized advantage with per-control lambda.95.

Existing snapshot replay (12a0f7c) changed only lambda.95->1 under the same
KL budget: startup sample probability increases1/7->5/7 andmeanlogprobchange
-.086->+.194. This changes the cheapest next decision: one same-init/seed/budget
real task Probe of lambda1, not more architecture or BC searches. Root chooses
this action within the current Mission/user authorization. Offline probability
changes are not physical improvement, and optimizer replay used fresh moments.
Estimated oneGPU~5--6minutes; hard caps below. Positive held signal leads to
mechanism/fixed-policy verification; negative leads to saved-trace movement and
exploration diagnosis, not an automatic longer run. No external authorization
boundary or new branch needed; oldTask experiments remain paused.

## Fixed protocol

Same initializer startup-balanced-actor-20261010-r1/best.pt, fixed D288/R,
H328 measured t-3:t, standalone absolute Gaussian c, replan8/plan24, native
PD gains andframe0 single-airplane task as
[PPO baseline](P-20261010-trajectory-ppo.md). Only scientific intervention is
**per-control GAE lambda1 instead of.95**. Gamma.99, actual chunk durations8/6,
terminal542 androllout bootstrap unchanged. Same current-state task reward,
Adam actor1e-6 withbounded KL.02 backtracking/persistent accepted LR, critic3e-4,
full-batch4epochs, stdbounds, strict highest precision. No future/reference
reward or policy inputs, tactile, phase/clock, WM, entropy/prior, orresidual base.
Public explicit GAE parameter preserves default.95 behavior; recorder/auditor
read actual lambda. Existing runs/source SHAs remain attached to their original
Git identities rather than being overwritten by current code.

Matched exploratory seeds293train/294eval intentionally reused from the first
Probe; not new statistical replicates. One24updates x16high x16env run, <=49152
interactions. Reuse original lambda.95 run as the observational control; no new
lambda.95 simulation. Independent full frozen evaluation uses rolesGT/dense/
warm_start/finalPPO4each withsame random role permutation seed294 andpipeline.
Bothlearned arms pureownH anddeterministicmeans, no privileged handoff.
Only final24 checkpoint evaluated, no bestcheckpoint/seed selection.

Engineering smoke debug9/two updates/two chunks14controls checks lambda1
API/GAE/actual optimizer andnative resets; <=120s, cannot prove grasp.
After finite full training, replay allH/behavior/value/bootstrap, lambda-aware
GAE/returns, current physical reward, every sampled c independent native
XYZ/SO3/Euler/fingers/FK/velocity, fixed Rinputs/latent andrequested/applied/PD.
Frozen eval calibration GT/dense each>=3/4 held>=433 andterminalheld. Local
PROMISING iffinalPPO>=3/4 withclip<1% and>=2more successful thanwarm;
UNPROMISING ifbothlearned0/4 withcalibration; otherwiseUNCLEAR. Also report
baseline usefulness separately ifPPO>=3/4 without warm-start improvement.
Compare held acquisition/loss/clip andinitial mean movement toold run; cannot
claim formal lambda superiority from one seed or changing trajectories.

## Resources / stop

Single idleGPU4, new14GPUmin/512MiB: smoke<=120s, fulltrain<=480s,
one542control frozen evaluation<=180s, allreplay/statistics<=60s.
GPU fornetwork/FK/PhysX, CPUfiles/stats/tiny contract tests. Observeperupdate
GPUutilization/memory/ETA, guardforeigncompute. Abort nonfinite/inputsource
orweights drift/unsafeGPU/unexpectedreset/logprobfailure/timeout. Preserve
original outputs; no automatic extra training, extra seed orcurriculum run.
Do not touch other processes/checkpoints orread-onlyexternal project.

## Results

Engineering smoke `0487233` debug9 completed13.651s/448interactions,
actual lambda1 recorded. Duration8/6/8/6 andterminal masks correct; independent
MonteCarlo prefixreturn max error8.94e-8, actor changed9.68e-5, jointKL .01395/
.001835, behaviorlogprob replay max.000535. This only validates wiring.
Full task train run launched fromsame0487233, seed293, fixed24updates.
No final policy/held conclusion until completed frozen comparison andaudits.

## Limitations / future evidence

Same imperfect warm start, one motion/seed andtiny Gaussian trajectory noise.
Longer traces increase target variance; rollout truncation may still miss late
reward. Smaller mean movement andstochastic/mean deployment difference remain
possible bottlenecks. Ifcredit improves offline butgrasp doesnot, do not call
lambda1 ineffective generally. Need matched native-action RL andmulti-seed
stable-held/drop validation before a baseline claim, andfinally Cm-on/off
training benefit; WM remains deferred until a usable trajectory baseline.
