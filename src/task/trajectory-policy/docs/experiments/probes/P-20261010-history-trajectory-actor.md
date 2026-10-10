---
schema: ref2dex.probe.v2
probe_id: P-20261010-history-trajectory-actor
experiment_id: P-20261010-history-trajectory-actor
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 68fccf9
claim_id: C3
hypothesis_family: HF-trajectory-policy-learning
probe_index_in_family: 1
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: keep independent actor initializer and enter real-reward trajectory PPO
decision_changed_if_negative: audit measured-history and prefix prediction execution gap before RL initialization
status: UNPROMISING
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

Code `68fccf9`; fit seed295 completed13.980s onGPU4,2500updates. GPU utilization
16--19%/399MiB at later monitors, Torch peak59.10MiB; tiny dataset throughput
reasonable. Selected update250 per frozen val criterion. Train6504/val2168
samples, train rows[0,1,2,3,4,5,6,7,9,10,11,12], val[13,15,14,8].
Val objective1.46352 vs train-mean7.31600 (79.996% lower), but val replan first8
hand RMS34.979mm and palm maximum189.530mm. No execution claim from that loss.

Native code `68fccf9`, seed296,54.963s,16env/542controls/68replans onGPU4.
GPU utilization26--43%/7543MiB at later monitors; owned PID294788 exited.

| Role | >=433held+terminal | Terminal held | Median max held | Clipping |
| --- | --- | --- | --- | --- |
| Original GT tau | 4/4 | 4/4 | 478 | 0% |
| Dense hand-derived FK | 3/4 | 3/4 | 480.5 | 6.8266% |
| Independent history actor | 0/4 | 0/4 | 0 | 0% |
| History actor repeat rows | 0/4 | 0/4 | 0 | 0% |

Calibration met the predeclared >=3/4 grasp gates (one dense row lost after
139heldframes; do not hide that row or its clipping). Neither actor arm formed
stable grasp. Local **UNPROMISING** for this particular BC initialization/D/R,
not a test of PPO or all H-only policies. Repeat rows are not independent seeds.

Independent audit r2 `4af8345`: initial state/H/actor c/decoded native q/FK all
exact; future velocity4.29e-6, actual897Rfeatures3.81e-6, intended/command2.38e-7,
appliedPD0. Audit r1 used strict FP32 whereas rl_games Runner.__init__ changes
native runtime to high/TF32-on; c error0.008327484 exceeded fixed tolerance.
Preserved FAILED r1 record. Matching actual high precision made all68c exact
without relaxing thresholds, changing checkpoints or rerunning simulation.
Pure FP32/TF32 c differences at startup0.001621246 are far below initial physical
error; no claim that precision caused failure. Runtime setting was reconstructed
from installed Runner and exact saved-output replay, not captured in native
manifest; record it explicitly in future runs.

Read-only independent AGENTS14 review found no H/label/split/normalization or
actor->D/R wiring error. Initial H equals source/val H exactly, labels agree
across rows, no startup label conflict. Native first-query prefix handRMS56.363mm,
offline same-H56.369mm; initial action XYZ RMS35.31mm, rotvec RMS0.3033rad,
finger maximum0.1914rad. Thus inaccurate initialization precedes feedback:
not all failure can be assigned to covariate shift. Native prefix RMS grows
59.51mm at tick8 and146.12mm at tick16; correlation alone isn't causal proof.
At ticks0/8/16 all H columns stay within training min/max; tick64 about0.9%
outside. Tick0 occupies12/6504=0.1845% of uniform training samples. Global val
selection improving80% did not guarantee the startup portion.

Artifacts:

- Fit: `outputs/trajectory-policy/history-trajectory-actor-20261010-r1/{manifest.json,result.json,best.pt,coverage.npz,monitor.jsonl}`.
- Native: `outputs/trajectory-policy/history-trajectory-actor-execution-20261010-r1/{manifest.json,result.json,trajectory.npz,plans.npz,monitor.jsonl}`.
- Audit: `outputs/trajectory-policy/history-trajectory-actor-execution-audit-20261010-r2/{manifest.json,audit.json,behavior.png}`.
- Preserved failed audit: `outputs/trajectory-policy/history-trajectory-actor-execution-audit-20261010-r1/manifest.json`.

## Follow-up Decision Note

Do not initialize a PPO benefit claim from this non-grasping mean. The next
cheapest distinction is whether rare startup/approach states are underfit under
uniform BC/global-val selection, rather than changing H/D/R or adding WM.
One fixed startup-balanced fitting/selection Probe on the same all-row dataset,
with explicit first-query geometric screening before at most one native wave,
is warranted. Same pure H and independent actor; no phase/clock policy input.
If that remains poor, stop BC weighting search and reassess initialization/RL
curriculum with a bounded decision note. No core claim or external authority
changed; preserve this valid negative initializer evidence.

## Limitations / future evidence

Single-motion row split, oracle offline hand labels, frozen warmstarted executor,
small BC initializer; no RL or causal Cm utility yet. Positive needs real-reward
PPO, native-action comparison and matched Cm-on/off training. Failure doesn't
refute all H/trajectory policies or288D RL. Covariate shift, observation aliasing,
exploration and task generalization remain distinct questions; investigate only
when they change the next decision.
