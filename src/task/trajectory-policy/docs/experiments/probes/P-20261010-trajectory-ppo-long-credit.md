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
status: UNPROMISING
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
Full training completed265.022s,48992interactions,96accepted actor epochs.
Actor mean parameter L2change .00097549; sampleheld2825steps, clipping1.2349%.
Two completed rows were terminalheld252/424frames, neither>=433;0fullstable
successes duringtraining (oldlambda.95 hadone486frame terminalheld row).
Valuefit lasttwo losses1279.7/1260.6; longer credit increases target magnitude/
variance. No nonfinite/KL/logprob guard failure, no model/physics restart.

`6d0d532` CPU fixed-condition comparison passed: first16high queries and128native
controls match original lambda.95 run **exactly**, including H/c/mean/std/value/
reward/state/action/PD. Fixeddata/checkpoints, seed/horizon/optimizer/gains match.
After updates, lambda1 observed12sampled rows>=45held,9startup normalized Apositive;
oldlambda.95 7rows,0positive. Different subsequent trajectories are not matched
individual outcomes. Credit improvement alone is not stable-grasp learning.

Training audit r1 (`6d0d532`) stopped on FP64 versus storedFP32 GAE error4.79e-5
above original3e-5 fixedabsolute tolerance; failed manifest preserved. Independent
NumPy FP32 replay isexact for all24rollouts. `f5a965d` audit r2 reports FP64maximum
4.938e-5 separately andpreserves fixedFP32 tolerance; actual GAE/returns exact,
H/distribution/logprob/V/bootstrap/cdecode/FK exact, velocity7.63e-6, reward1.19e-7,
Rinputs5.72e-6, command/PD exact. Audit6.133s; only validator precision fixed,
nottraining data/algorithm, no physical rerun.

Frozen f5a965d seed294 completed55.423s:GT/dense each3/4>=433held+terminal,
warm_start/finalPPO each0/4 withzeroheld. PPOclipping .6919% (oldlambda.95
1.1531%), warm .3229%; betterclipping alone is notgrasp gain. LocalUNPROMISING.
Independent execution/outcome audit launched; no repeatphysical wave.

Next post-hoc **Decision diagnostic**, withinremaining audit60s cap: on all68x4
saved warm-start measured H/currentq/object, frozenwarm/lambda.95final/lambda1final
means throughsameD. QueryeachactoronidenticalH, measure decodedfirst8 hand/XYZ/
fingermean movement andactualstd, especiallyprecontactticks0--40. Allqueries
retained; interpretthis as updateamplitude, notsuccessfulcounterfactual physics.
SingleidleGPU4 forbatchinference/FK,<=30s, smallJSONonly; no training/sampling/
physics/extraepoch. Ifmovement remains tiny relativedeclared Gaussiannoise,
nextdecision examines trust-region/updatescale; ifsubstantial, investigate
closed-loop/contact execution before changingoptimization. Read-only independent
negative-result review requested underAGENTS14, limitedsource/trace scope.

Independent frozen execution audit `f5a965d` passed:H/c/D/nativeEuler/FKexact,
velocity3.81e-6, Rfeatures3.81e-6, command2.38e-7, PD0, outcomes/screen agree.
No simulation rerun. Allcheckpoints/originalfailedvalidator retained.

`77b0e9e` fixed-H movement diagnostic completed in 5.800s onGPU4,
all68x4 warm-history queries retained. Metrics here are **per-coordinate RMS**
(not Euclidean per-point RMS). Initial handmean movement:warm->lambda.95
.01747mm, warm->lambda1 .02569mm. Before contact(ticks0--40), lambda1
XYZmean change<=.03893mm, handmean<=.04205mm, fingermax8.12e-5rad; actualXYZ
samplingstd1mm. Thus update amplitude is small relative to exploration at these
fixed measured inputs; not proof that optimization is locked or causal failure
attribution. Failed warm-H aftercontact may also be OOD; no physicalcounterfactual.

Read-only independent review confirmed actuallambda intervention: firstrollout
FP32 A replaylambda1exact; reusinglambda.95 woulderr11.7264. All96epochs accepted,
KLmedian .003782, LR6.25e-8, stdalmostunchanged. Longercredit improves sampled
startup signal butdoesnot establish frozenmeangrasp. Root chooses the next
cheapest **frozen stochastic warm/final comparison withdeclared paired noise**
beforeanothertraining change, to distinguish learning instochastic deployment
from occasionalexploration successes. Newprotocol/resourcecard must precede
thatwave; none launched here. If no usable stochastic gain, then investigate
actualKL/step movement onretainedbatches before any optimizer tweak; packetslack
Adam moments, so freshoptimizer replay mustnot be called an exactcounterfactual.
No morelambda/epoch/BC/48D sweeps, noWMactivation orMission claimchange.

Artifacts:
- `outputs/trajectory-policy/trajectory-ppo-long-credit-20261010-r1/`
- `outputs/trajectory-policy/trajectory-ppo-long-credit-training-audit-20261010-r1/` (FP64validatorfailure)
- `outputs/trajectory-policy/trajectory-ppo-long-credit-training-audit-20261010-r2/`
- `outputs/trajectory-policy/trajectory-ppo-long-credit-execution-20261010-r1/`
- `outputs/trajectory-policy/trajectory-ppo-long-credit-execution-audit-20261010-r1/`
- `outputs/trajectory-policy/trajectory-ppo-credit-comparison-20261010-r1/`
- `outputs/trajectory-policy/trajectory-mean-movement-20261010-r1/`
12Task contracts passed andrepositoryverification passed. Currentoutputs119GiB,
wellbelow300GiB global cap; thisProbe artifacts remainwithin512MiB and14GPUmin.

## Limitations / future evidence

Same imperfect warm start, one motion/seed andtiny Gaussian trajectory noise.
Longer traces increase target variance; rollout truncation may still miss late
reward. Smaller mean movement andstochastic/mean deployment difference remain
possible bottlenecks. Ifcredit improves offline butgrasp doesnot, do not call
lambda1 ineffective generally. Need matched native-action RL andmulti-seed
stable-held/drop validation before a baseline claim, andfinally Cm-on/off
training benefit; WM remains deferred until a usable trajectory baseline.
