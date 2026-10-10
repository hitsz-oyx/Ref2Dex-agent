---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-decoder-coverage
experiment_id: P-20261010-trajectory-decoder-coverage
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: ce503afd93d717c0f6868c00e9dc7d9273201234
claim_id: C3
hypothesis_family: HF-trajectory-policy-action-space
probe_index_in_family: 1
seed_pool: probe
seeds: [297]
decision_changed_if_positive: initialize an independent measured-history latent actor then train high-level PPO
decision_changed_if_negative: repair trajectory representation or executor interface before policy training
status: UNPROMISING
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
label reconstruction diagnostic, not H->c inference or final policy success.
Copying reference knot values is not optimization over all c, so this is not
an upper bound on the decoder's attainable control performance.
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

Code `ce503af`; seed297; offline audit2.09s onGPU2, native wave57.29s onGPU2.
Native GPU memory~7.5GiB (Torch peak143.66MiB), utilization16--32% at later
monitors; speed reasonable for small CPU-exchange16env/GPUPhysX evaluation.
GPU2 released to its pre-existing passive130MiB process; no own live process.
Saved newTask artifacts~30MiB. Two tiny CPU decoder contract tests passed.

| Role | >=433held and terminal | Terminal held | Median max held | Clipping |
| --- | --- | --- | --- | --- |
| Original GT tau | 3/4 | 3/4 | 479 | 0.1845% |
| Dense hand-derived FK tau | 4/4 | 4/4 | 482.5 | 0% |
| Four-node48D | 0/4 | 0/4 | 1.5 | 1.1531% |
| Four-node48D repeat rows | 0/4 | 0/4 | 0 | 0.2768% |

One48D row did acquire and hold397frames before loss; do not describe all rows
as never grasping. The repeated arm has identical design but different native
rows, not a second seed. Original GT3/4 and dense4/4 meet predeclared calibration.
Both48D arms miss the unchanged long-held/terminal screen. Local **UNPROMISING**
for this node-copy reconstruction plus frozen executor; no actor was trained.

Geometry coverage relative dense FK: full24 point3D RMS13.94mm, first8 RMS13.97mm;
first8 palm maximum84.25mm, all-point maximum116.85mm. The largest prefix error
is the first planning window. Actual source wrist moves nonlinearly during its
first8steps, while D interpolates a straight segment between frames1 and8.
Independent read-only review reconstructed all68plans: translation4.17e-7m,
finger1.15e-6rad, Slerp matrix7.09e-7, coupling error0. It found no SO3, timing,
padding or future-only FF wiring error. In the first window, planned wrist
FF base position difference `delta_q + .1*delta_velocity` reaches141.69mm;
rotation difference-vector norm0.3656rad. Compression changes immediate
geometry and feedforward, not merely an unexecuted far suffix. Their separate
causal contributions have not been isolated.

Independent saved-input/command audit: actual897features maxerror2.86e-6,
velocity9.54e-7, command2.38e-7, applied PD0; requested/applied commands exact,
initial states exact, all outcomes/screens reconstructed. Audit r1 failed only
on JSON serialization of a NumPy boolean after checks; preserved FAILED record.
Fix `5fe3d05` casts counts to Python integers; audit r2 completed without rerunning
simulation or modifying source results. This serialization failure does not
invalidate execution data or support changing physical thresholds.

Artifacts (read-only oldTask inputs remain in manifests):

- Coverage: `outputs/trajectory-policy/trajectory-decoder-coverage-20261010-r1/{manifest.json,result.json,coverage.npz}`.
- Native: `outputs/trajectory-policy/trajectory-decoder-execution-20261010-r1/{manifest.json,result.json,trajectory.npz,plans.npz,monitor.jsonl}`.
- Audit: `outputs/trajectory-policy/trajectory-decoder-execution-audit-20261010-r2/{manifest.json,audit.json,behavior.png}`.
- Retained failed audit: `outputs/trajectory-policy/trajectory-decoder-execution-audit-20261010-r1/manifest.json`.

## Follow-up Decision Note

Do not initialize PPO with this unverified reconstruction or add WM yet. Keep
the independent actor->D->tau->R architecture; this Probe only rejects promoting
the current node-copy initializer/frozen-R combination. Next cheapest question:
can better fitting of the same c preserve executed-prefix positions AND nominal
velocities, or is a more expressive D needed? This avoids confusing a poor
encoding with a proof that no feasible c exists. Any follow-up gets a new bounded
protocol before execution; no additional training/simulation in this card.

## Limitations / future evidence

Single motion/seed, oracle-label tau reconstruction and frozen oracle-warmstarted low
controller. Positive does not prove a learned H->c policy, superiority to native
PPO, general trajectory executability, original placing success or Cm benefit.
Need standalone actor and actual RL learning, then matched Cm-on/off training.
No optimization over c or separately matched geometry-vs-velocity intervention;
this failure does not refute all48D representations or all policies within D.
