---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-execution
experiment_id: P-20261012-goal-point-wam-ref2-execution
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 07bf1e6b
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 18
seed_pool: probe
seeds: [292]
decision_changed_if_positive: retain live WAM-R interface and review a separately bounded robot-domain continuation
decision_changed_if_negative: use live state/trajectory/PD trace to locate startup or retargeter-distribution failure before adapting again
status: UNCLEAR
run_id: wam-ref2-execution-20261012-r2
---

# First live WAM -> frozen R -> native PD simulated closure

Decision: user ref2's hard engineering gates are named semantic mapping and
current-object frame numeric equivalence. They passed; its goal is to use
minimal simulated closure to learn whether forecast EPE blocks real control.
Root explicitly allows this simulation despite the unchanged conservative
shadow PD distance screen failing. This does not relabel shadow/calibration
results or introduce real robot execution. No oracle prefix repairs WAM.

## Fixed protocol

One seed292, one idle physical GPU2,3 nominal frame0 environments: official
teacher, GT-R and WAM-R. Same pinned native env/assets/motion/pipeline/controller
as qualified teacher and prior R physical calibration, 542 controls at30Hz,
actor query every control. Measured q/dq/hand/object/velocity reset equality
must hold<=1e-5. Three-row physics is newly calibrated by teacher and GT-R;
if either fails hold45/controlled-placement, this run is UNCLEAR even if WAM
fails. One episode per row is a route Probe, not success-rate evidence.
With calibration passing, WAM hold>=45 frames and clipping<1% is PROMISING
for student initialization; zero held frames is UNPROMISING for this exact
initialization; intermediate holds remain UNCLEAR. Report controlled-placement
success separately. No checkpoint or threshold selection after seeing physics.

GT-R takes original actual future-hand source donor24 in world coordinates
and its own live measured state. Only this row may read privileged future-hand
labels. WAM-R takes its own genuine measured H4 at every tick, repeating actual
frame0 measurements for the missing startup history. Known object endpoint is
planned native reference pose at min(t+24,542),32 frozen canonical mesh point
pairs. No teacher intervention, future q/contact, measured future object or
GT prefix enters WAM. Original WAM step10000 plus the fixed300 head-only
robot adapter are frozen, along with original R; no new model selection or fit.

Pinned native Torch2.0.1 cannot import the WAM spconv/torch_scatter runtime.
Use a separate graspenv Torch2.4 server on the same physical GPU with bounded
Unix socket JSON (<=64KiB/frame,30s per socket operation). Both exact owned
PIDs are whitelisted; no arbitrary GPU process whitelist. Server accepts only
sequential tick/H4 object/H4 hand/known endpoint fields, returns native-ordered
relative tau and the exact current object anchor. Native R receives that tau
directly with the same current measured state; no world/anchor roundtrip.
Server and native runtime record input/source/checkpoint/stats/architecture
SHA, latency, all live observations, requested/applied action, requested/actual
PD, clipping and measured contact geometry. Native verifies actual PD equals
the independent inverse, and the reference matches its actual HOI loader.

Source teacher manifest's native/assets/motion/config hashes must remain exact;
only a changed producer entrypoint may authenticate its recorded Git blob.
Reference is checked against its export manifest and head calibration identity.
Original model source and head/base/stats SHA must agree before inference.

## Resource and stop conditions

Entire process group<=600s from launch; combined new ref2 output/private
scratch<=100MB including previous shadow and head calibration. Socket lives
only under projecttmp, byte path<108 and directory0700/socket0600. Own process
groups only may be terminated on timeout/source/foreign-PID failure. Stop on
nonfinite, mismatched peer/frame/reference, native PD mismatch, unexpected
reset, foreign GPU PID or source drift. No checkpoint or simulation video.
Group launcher: `src/task/trajectory-policy/tools/run/launch_wam_native_bridge.py`;
server and native entrypoints in the same directory. Independent interface tests
cover finite JSON, exact peer PID, maximum frame size and malformed length.

## Results

First execution code2fb7a8b9, r1, ran76.90s and returned172 live queries.
Frame0 role reset, native reference and first-query full-model/bridge differences
were exact0. Query172 stopped at the original fixed PTv3 workspace bound
(grid y_min=-4). Both owned runtimes stopped; no full behavior/calibration
verdict is available. The exception path omitted partial raw trace persistence,
a recording implementation defect; preserve r1 and its FAILED logs as-is.
Root permits one same-model/same-seed r2 after logging-only repair, within
unchanged600s/100MB bounds. Save executed-prefix states/actions/PD/H4 plus
failed server request, then audit earliest visited-state divergence. No actor,
R, sampling, serialization origin or precision threshold changes. The original
10mm/.05rad shadow screens remain FAILED; physics is a separately authorized
Decision, not a claim that those screens have passed.

Logging-only rerun r2, code07bf1e6b, stopped at the same query172 in62.32s.
It saved172 executed controls and173 measured states in
`outputs/trajectory-policy/wam-ref2-execution-20261012-r2/native/partial.npz`
(SHA d0392772a84c219bd4e2d28791f771f39bed54b1e553c9bd2925c214ecc6b19b),
plus the exact failed live request. Scene minimum y=-2.03637m crosses the
unchanged -2m workspace origin (1cm voxel, grid y_min=-4). Model/head/R,
seed292, reference and controller remained fixed; no guard relaxation.
All owned processes exited. GPU2 peaked around18776/24576MiB with
native18156MiB/server594MiB; sampled utilization14–40%, server median
49.88ms/query and native median124.42ms/control. Combined ref2 outputs/private
scratch approximately11.26MB; the two groups together139.22s.

Two independent raw audits recovered exact0 for requested/applied action,
actual native PD versus pure inverse, current-state87, GT-R source tau,
WAM's own live H4 and planned goal; all raw arrays are finite and authenticated.
Pinned Torch2.0.1 GPU replay of frozen R differs by at most2.98e-7 in encoded
output and4.77e-7 in requested PD; original first-query/full-model server
comparison was exact0. No recorded source/input drift. Reproduction/audits:
`outputs/trajectory-policy/wam-ref2-execution-20261012-r2/prefix_audit.json`,
`root_prefix_audit.json`, `R_dependency_audit.json` and their artifact scripts.

Observed prefix geometry, **not a full episode verdict**: teacher and GT-R
achieved118 and119 consecutive held frames, while WAM achieved0. Placement
was not reached by the truncated run, so three-row calibration is incomplete
and the predeclared full-episode label remains **UNCLEAR**. WAM wrist/object
distance grows from0.224m at tick64 to1.837m at172; teacher/GT distances
remain about0.169/0.174m at172. WAM object stays on the table (height~0.780m)
while teacher/GT have lifted it to~0.881/0.888m. All rows have0 clipping;
the large native XYZ scale makes clipping a weak safety/quality proxy.

Earliest failure is measurable before contact: identical frame0 state yields
WAM versus actual teacher PD XYZ error225.27mm, GT-R2.28mm; measured WAM
wrist differs from teacher by25.32mm after the first control. GT-R wrist first
exceeds10mm only at measured tick53. WAM prefix PD error versus teacher has
p95=1529.29mm as its own visited state drifts.

Privileged offline substitutions at **the same measured WAM state** locate
input dependence, not an executable oracle repair: at tick0 replacing the
first1/4/8/16 predicted tau rows with GT reduces immediate PD discrepancy
versus full GT-R from225.80mm to153.04/38.53/14.59/4.61mm. The short trajectory
prefix materially drives initial control error. Later state divergence is
also consequential: at WAM tick64 even full source-GT tau yields PD XYZ
298.45mm from nominal teacher, and at171 this is1447.00mm. Source future hand
becomes an out-of-distribution recovery request in the diverged live state.
This does not establish that repeated startup H4 alone causes failure, nor
that a lower nominal trajectory EPE alone guarantees closed-loop control.
The next useful decision is whether targeted startup/prefix supervision can
prevent initial drift while retaining nominal real-H4 behavior; any new fit
requires its own fixed protocol, and no fit or further actions occurred here.

## Limitations / future evidence

Native landmarks match semantic roles, not human anatomical offsets. Teacher
and oracle R calibration can fail when physical env count changes. Distinct
live rows are not exact hidden-state forks. One nominal frame0 motion cannot
establish perturbed/multi-motion control, human retention, real robot transfer,
actual-goal precision, RL gains or Cm causal utility. Preserve every failure
and its source rather than closing the whole WAM route from a single episode.
