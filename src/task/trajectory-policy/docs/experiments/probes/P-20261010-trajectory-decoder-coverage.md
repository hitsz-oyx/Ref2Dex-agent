---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-decoder-coverage
experiment_id: P-20261010-trajectory-decoder-coverage
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-action-space
probe_index_in_family: 1
seed_pool: probe
seeds: [297]
decision_changed_if_positive: initialize an independent measured-history latent actor then train high-level PPO
decision_changed_if_negative: repair trajectory representation or executor interface before policy training
status: UNCLEAR
run_id: trajectory-decoder-coverage-20261010-r1
---

# Can a fixed48D trajectory decoder retain executable grasp behavior?

## Motivation / Decision Note

Decision serving ref1 independent trajectory policy and eventually Mission Cm
policy-training utility. Previous goal turn made progress: retained actual
future-velocity fix, established generated proposal still fails and organized
new architecture. Current f852b77 has no newTask implementation/live process.
User stopped oldTask experiments; all new tools/output and this Probe belong
to trajectory-policy. Reuse old stable controller/geometry as read-only inputs,
not resume its proposal/T/selector experiments.

Hypothesis: fixed four wrist+finger trajectory nodes can preserve the contact
behavior of existing executable hand tau, so PPO can act in this simpler space.
Cheapest informative route: one geometric coverage audit, then one single-motion
native wave, frozen executor and no policy training. Positive lets us initialize
H->c; negative requires fixing D/R before adding WM or RL complexity.

## Protocol

One fixed48D decoder, knots[1,8,16,24]. Each node has query-object-frame wrist
translation offset, relative SO(3) rotation vector and six independent finger
shapes. Translation is tanh-bounded at1m per coordinate; rotation uses a smooth
radial pi-angle bound; fingers use bounded sigmoid times native limits/coupling.
Linear translation/finger interpolation and shortest-path SO(3) interpolation;
actual current wrist/fingers anchor time0. FK generates24x11x3 tau; its geometric
q is a derived decoder quantity, not future measured q_ref or PD commands.
Latent actor would emit standalone c; no frozen base-policy addition.

Coverage uses verified old hand-derived geometry q, generated only from GT
hand tau/reset/static URDF, not recorded future joints. It is an explicit oracle
label upper bound to test decoder, not H->c inference or final policy success.
Check all68 eight-step windows, first8/full24 errors, palm/fingertips and actual
limits/coupling. Do not select knots/dimension/seed from these results. Tiny
shape/rotation tests can use CPU; batched FK uses one idleGPU2.

If finite valid geometry, exactly one16env/542control native wave, four roles
of4rows each randomized seed297: original GT-tau controller, dense hand-derived
geometry+FK tau with future-only FF,48D decoded tau, identical48D repeat. Replan
every8steps using own current calibration/object frame, oracle GT hand-derived
future knots only in this explicit diagnostic. R reads live feedback every step,
consumes24-step shifted/padded tau, uses its own learned residual/native clamp.
No true future object/q/force/reference commands as controller inputs; original
GT baseline is independent calibration. No reset/fork/same-state causal claim.
All48D data and commands must be actually recorded/audited.

Calibration original GT>=3/4 >=433held and terminal. Dense FK failure makes
compression attribution UNCLEAR. Each48D arm >=3/4 longheld+terminal and clip<1%
is local PROMISING; both fail with calibrated dense path yields UNPROMISING for
this fixed decoder/executor combination. Other cases UNCLEAR. Report geometry
errors without treating them as proof of physical grasp. No threshold relaxation
or knot search. Baseline passes are required before entering PPO initialization.

## Resources / stop

One idleGPU2; <=8GPUmin/512MiB, offline audit<=120s, one native run<=300s.
No training, extra seed, automatic retry or WM inference. CPU only tiny contract
checks/file-label audit; GPU batched FK/native simulation. Check memory/util/ETA
at start and128controls. Source/input hashes, finite checks and actual requested/
applied commands required. Stop on identity drift, foreign GPU work, invalid
bounds/coupling/nonfinite or unexpected early reset. Preserve failed outputs.
New branch prohibited without user permission; remain main; no push.

## Results

Not run yet.

## Limitations / future evidence

Single motion/seed, oracle tau compression and frozen oracle-warmstarted low
controller. Positive does not prove a learned H->c policy, superiority to native
PPO, general trajectory executability, original placing success or Cm benefit.
Need standalone actor and actual RL learning, then matched Cm-on/off training.
