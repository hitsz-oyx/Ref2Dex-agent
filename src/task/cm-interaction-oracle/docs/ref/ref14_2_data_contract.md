# Ref14_2 P0: immutable rolling-oracle data asset

This is CPU file/label packaging of completed ref14_1 groups, not a new Probe
or permission to change Y, fit models, or collect additional simulation.
It serves ref14's later `F_actual→Y` and `E/I→Y` comparisons by preserving
physical observations and their provenance without repeating expensive replay.

Entry: `tools/audit/package_rolling_oracle_asset.py --source-run <rolling run>
--run-dir <new unique outputs/cm-interaction-oracle/ref14-assets-...>`.
Use repository `tmp` for TMPDIR. The tool refuses an existing output folder,
reads only groups with `s*-g*-result.json`, checks all12 completed decisions,
and leaves the active source run untouched. Admission is frozen at invocation;
groups finishing later require a new asset folder. Cap: CPU, two threads,
540seconds checked between groups, <=1GiB new output. This is file processing,
so GPU startup has no benefit. A failed run keeps its partial outputs and has
no COMPLETED manifest; downstream consumers must require that manifest.

## Schema `ref2dex.rolling_oracle_asset.v1`

One `s<seed>-g<group>-asset.pt` per completed synchronous solver group.
`manifest.json` records group coverage, source and output SHA256, tool hash,
Git identity, device, elapsed time and limitations. These snapshots may be
partial cohort assets; never infer a full experiment gate from their counts.

For N fixed rows, each of12 `rounds` at offsets0,8,...,88 contains:

| Field | Meaning / shape |
| --- | --- |
| `choices` | N selected native candidate indices |
| `candidates` | Seven slots, dict if physically observed, `None` if pruned |
| `candidate_observed`, `missing_reason` | Observed mask and explicit max-U certificate missingness |
| `actual` | Separate mixed execution observation; first8steps executed, later suffix speculative |
| `actual.executed_mask32` | Local steps1..8 true; never treats speculative continuation as actual |
| `actual.in_Z90_mask32` | Executed observations belonging to final bounded90; only first2 at offset88 |

Every observed candidate/actual record contains:

| Field | Meaning / shape |
| --- | --- |
| `H.before/history/actor_obs/hand_root` | Original current physical72/history/actor input/root13 |
| `H.native_q0` | N×18 pre-action native DOF positions from final factual trace |
| `H.before_fingertip_positions/before_hand_base_pose` | Measured current tips / base pose |
| `a_base/native_actions/pd_targets` | N×8×18 feedback actions and PD targets |
| `delta_a` | Candidate18 or actual N×18 native residual; action clipping retained |
| `F_actual_tip_displacement` | N×8×5×3 measured WORLD tip displacement from current tips |
| `F_actual_surface` | `None`: corresponding sampled hand-surface positions were not recorded |
| `native_q32/tip_positions32/hand_base_pose32` | Measured32step physical execution trajectories |
| `physical32` | N×32×72 raw physical trajectory: object13, body poses35, hand forces15, object force3, proximity5, pair proxy1 |
| `E8/I8` | N×12 / N×14 inherited `physical_targets(before, physical32[:,7])` |
| `height/pair/valid_steps` | Original full recorded outcome windows |
| `Y32/current_risk/U` | Exact inherited short-Y8 and utility at this actual-current-state query |
| `Z_candidate` | `None`:32step candidate forks cannot provide90step Z |

E12 uses position displacement (metres), quaternion short rotation (radians),
and linear/angular velocity differences in world axes. I14 uses five log1p
hand-force norms, three log1p absolute object-force components, five physical
body-to-object distances (metres) and aggregate pair proxy. It does not certify
paired contacts, slip or friction. Native q0:3 are metres, q3:18 radians.
Do not call five-tip displacement corresponding dense hand-surface flow.
Raw q/root/base observations are retained for a separately specified FK audit;
surface flow is not silently reconstructed or replaced by body-centre flow here.

`factual90` contains N×90 pre-action physical/dof/root, applied action/base
action/done and post-action `after_physical`. `Z_rolling`, `Z_baseline`,
qualification and `failure_events` preserve original cohort pairing. Failure
events explicitly separate first raw threshold event, qualification and first
post-qualification drop; a raw event is not automatically a task failure.

## Guardrails

Same-current-state H and plan input hashes are verified, raw Y/U and choices
replayed exactly, mathematical pruning checked, native selected residual
wiring checked, final actual action/full-physical continuity checked and Z
recomputed. All completed source hashes are rechecked before manifest closure.
No unknown candidate labels, omitted flows or long-horizon candidate Z are
filled. Candidate continuation is speculative after its own first actionblock;
actual intermediate suffixes are also speculative and superseded by replanning.
The final contiguous factual trace alone supplies rolling Z.

Measured future tip flow/E/I and `a_base[:,1:]` are post-treatment observations,
not available planner inputs at query time. Later candidate H differs across
actual paths; shared-solver co-treatment remains. No packaging/test success
demonstrates predictor utility or final Cm-on/off policy-training benefit.

## Completed-group snapshot (2026-10-05)

First complete-group asset:
`outputs/cm-interaction-oracle/ref14-assets-20261005-completed-groups-v3/manifest.json`
and `s263-g0-asset.pt`. At snapshot UTC14:59:46 only seed263/group0 was
complete:7 fixed anchors,12 rounds,84 observed candidate slots. All H/Y/U,
chosen actions, full actual physical continuity, Z and drop replays passed.
CPU packaging took3.76seconds; total artifact size20,653,123bytes. Three
contract tests passed (pruned unknowns, full-panel missingness/choice, measured
tip versus surface/Z semantics). A fresh saved-asset load confirmed final
offset88 Z mask contains local indices0,1 only.

This partial group contains baselineZ4/7 and actual rollingZ5/7; no full-cohort
utility gate or scientific conclusion follows. The running group1 and other
unfinished groups are absent. Earlier v1/v2 development snapshots are retained
immutably (total allthree assets about60MiB), but consumers should use v3,
which includes q0, full-physical continuity and code-dependency hashes.

## Latest consumption snapshot: both seed263 groups

Use `outputs/cm-interaction-oracle/ref14-assets-20261005-s263-completed/manifest.json`
with `s263-g0-asset.pt` and `s263-g1-asset.pt`. Snapshot UTC15:19:33 includes
the two completed seed263 groups only:20 anchors,24 query rounds,168 observed
candidate slots. All inherited raw Y/U, same-current H, actual chosen action,
full physical trajectory continuity, final Z, qualification/drop and source
immutability checks pass. Saved packets were independently reloaded. CPU
packaging8.18seconds, snapshot58,333,711bytes; cumulative retained asset files
120,084,531bytes, below the1GiB budget. Seed264 groups remain outside this
snapshot. No core/run protocol changes or scientific experiments were run.

Partial factual counts are baseline13/20 to actual rolling17/20, four rescues
and zero harms. These do not establish the full32anchor decision gate.
Baseline failure details below were recomputed from the original90step panels;
rolling details are from the contiguous final actual traces. Steps are local
post-origin1-based steps; qualification `-1` means not qualified bystep60.

| Group / envID | Baseline qualification | Baseline first raw failure | Baseline post-qualification drop | Rolling qualification | Rolling drop through90 |
| --- | --- | --- | --- | --- | --- |
| g0 /36 |45|89|89|45|false|
| g1 /15 |-1|12|none|55|false|
| g1 /33 |-1|78|none|45|false|
| g1 /63 |45|63|63|45|false|

No rescued rolling trace has a raw threshold failure. All20 rolling rows have
`dropped_after_qualification=false`; unqualified g0/env6, g0/env81 and
g1/env4 still fail Z through absent qualification. A missing qualification
must not be represented as a post-qualification drop.

## FK-derived dense actual-flow sidecar

Optional independent entry: `tools/audit/reconstruct_rolling_asset_flows.py
--asset-run <completed packaged asset folder> --run-dir <new unique folder>`.
It requires GPU FK and reuses frozen `NominalSurfaceActions` plus the original
audited `oracle-y-utility-sync-s263-s264/paired_actual_flow.pt` sample table
and seed241. The historical120point link/local-position/normal sampling tensors
must match EXACTLY. Visual meshes, URDF, code dependencies, sampling reference
and every input packet are SHA256 tracked; source assets are unchanged.

Output schema `ref2dex.rolling_derived_dense_flow.v1` in
`derived_dense_flows.pt`: groups retain seed/group/rows and each offset has
seven candidate slots, original observed mask and a separate `actual` slot.
Each nonmissing record has a seed/group/offset/candidate join key, where actual
uses candidate=`actual`; missing candidates remain `None`. `points_current`
is N×120×3 current hand-surface position; `flow_metres` and
`flow_normalized_002m` are N×2×120×3 actual0→4 and4→8 corresponding point
displacements in the CURRENT object's frame, in metres and divided by.02m
respectively. The object frame remains fixed at the query, not future pose.
Actual-record execution/Z masks are retained, including the final block's
two-step bounded-Z support. Full actual flow0→8 at offset88 extends beyond Z90.

These are FK-derived from recorded native q0/q4/q8 and current root using the
frozen URDF/sample geometry. They are not dense positions directly saved by
the simulator, causal forecast flow or desired/attainable planning actions.
The immutable source `F_actual_surface=None` remains correct; downstream code
must explicitly join this audited derived sidecar. FK q0 is checked exactly
against current history, measured tip Euclidean and hand-base matrix errors
must be strictly below1e-4 at0/4/8. Endpoint/chunk identity must be below1e-5.
Failure preserves a partial sidecar and FAILED report; no corrected inputs or
relaxed thresholds are allowed. Consumers require the PASS report.

Seed263 two-group check completed in
`outputs/cm-interaction-oracle/ref14-assets-20261005-s263-dense-flow/`:
GPU1,4.47seconds,14,064,420bytes,168 candidate plus24 actual join records.
Sampling match exact; worst measured tip error1.7792e-5m, worst base matrix
error4.9621e-6, chunk identity9.5368e-7, all below fixed bounds. Two engineering
tests pass (missing-mask/join, q0 identity rejection), and fresh saved-sidecar
load verifies192 joins, flow shapes and exact normalized-to-metre conversion.
No simulator, neural model computation or scientific comparison was performed.
Budget remains <=300seconds GPU and <=200MiB derived output per invocation.
