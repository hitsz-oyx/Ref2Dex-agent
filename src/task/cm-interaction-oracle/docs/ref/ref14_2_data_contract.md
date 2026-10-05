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

Current complete asset:
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
