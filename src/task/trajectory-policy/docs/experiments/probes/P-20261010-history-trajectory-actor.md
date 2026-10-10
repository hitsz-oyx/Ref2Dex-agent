---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-trajectory-actor
experiment_id: P-20261010-history-trajectory-actor
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-learning
probe_index_in_family: 1
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: keep independent actor initializer and enter real-reward trajectory PPO
decision_changed_if_negative: audit measured-history and prefix prediction execution gap before RL initialization
status: UNCLEAR
run_id: history-trajectory-actor-20261010-r1
---

# Can measured history initialize a standalone trajectory actor?

## Motivation / Decision

Mission A needs self-trained manipulation; ref1 wants an absolute trajectory
actor, not an online reference-action residual. D288 retains all24 wrist/finger
frames and its68x4 dense-plan native-q/FK/future-velocity replay passed. Stop
compression searching. Cheapest decision is one supervised initializer on the
existing full native wave, then one closed-loop physics screen; no new dataset,
WM, architecture/seed sweep or oldTask experiment. Positive opens PPO, negative
requires checking actual H->c->D/R behavior, not concluding RL cannot work.

## Inputs / fitting protocol

Source: `outputs/trajectory-policy/metric-trajectory-decoder-execution-20261010-r1/`.
Keep all543states x16rows, including every failed metric row. Extract only
measured q/dq/11hand-points/object pose/object velocity. H328 consists of the
existing300D four-state query-object-frame encoding, four native wrist dq
states24D (rotate XYZ rates, retain Euler joint-rate convention), current
absolute object height1D and query-frame gravity3D. Repeat earliest state to
pad history. No actor obs/reference/contact/action/phase/clock. Current height
and gravity are measurements, not future object information.

Labels at all542control ticks are24future geometric frames obtained solely
from hand tau, initial reset and URDF; encode against each row's actual current
q/object. Never use actual future robot q, future object or commands as labels
or inputs. Future hand is privileged offline supervision only. D288 deployment
decodes independent c with current state and never calls encode.

Fixed row split: last row of each source role is validation (4rows), remaining
12train; allticks retained. Same motion/wave/noise domain, not held-out motion
generalization or independent demonstrations. MLP328->512->512->288,Tanh;
input train-only mean/std floor.001, output train-only mean/std floor.1 in
physical c units. Standalone c, no base actor/chunk. Gaussian logstd log(.1)
fixed in BC; deterministic mean in execution. Old R frozen and separate.

Seed295, Adam lr3e-4, batch512,2500updates. Objective: physical c error weighted
1 forfirst8 and.25 fortail16, plus.1 times first8 wrist nominalFF error under
future-only central differences at1/30. Rotation-vector FF is only an Euler FF
surrogate. Select lowest row-val objective atupdates1/250/500/.../2500. Save
selected checkpoint once; no overwriting prior checkpoint. Report train-mean
baseline, prefix hand FK RMS and palm maximum; no loss gate substitutes grasp.

## Native screen

After finite training/identity passes, exactly one16env/542control/seed296 wave:
GT tau,dense FK,H actor,H actor repeat4each randomized. Actor observes only
its own four measured states, emits c288 every8controls. D/R unchanged, R reads
feedback eachcontrol. No future input/reward to H actor, no execution-time
mother actor, no updates/WM. GT/dense are privileged calibration roles; loaded
owned native actor only boots environment, never supplies evaluated actions.

Unchanged grasp screen: GT anddense each>=3/4 >=433held+terminal;
each H arm>=3/4 longheld+terminal with<1%clipping. Both pass withcalibration
PROMISING; bothfail withcalibration UNPROMISING; otherwise UNCLEAR.
Save all states,H/c/plans,actual897Rinputs/requested/applied/nativePD. Audit
H from past measurements, actor outputs, D/FK/future-only velocity, R input,
native commands/PD/outcomes. No positive claim until actual chain audit passes.

## Resources / stop

One idleGPU4, new bounded12GPUmin/512MiB: fit<=240s, one native<=300s,
audit<=120s. GPU neural training/inference/FK/PhysX; CPU file/statistics and tiny
contracts only. Monitor fit at1/every250updates, native at1/every128controls;
memory/util/ETA reported. Stop on source/input drift, nonfinite, unsafe occupancy,
unexpected reset or timeout; preserve failures, no automatic rerun. Staymain,
no newbranch/push or external process/data changes; oldTask stayspaused.

## Results

Not run yet.

## Limitations / future evidence

Single-motion row split, oracle offline hand labels, frozen warmstarted executor,
small BC initializer; no RL or causal Cm utility yet. Positive needs real-reward
PPO, native-action comparison and matched Cm-on/off training. Failure doesn't
refute all H/trajectory policies or288D RL. Covariate shift, observation aliasing,
exploration and task generalization remain distinct questions; investigate only
when they change the next decision.
