# P-20261001-paired-evaluator-resolution

Family HD02, probe_index_in_family1/1; Decision audit. User-provided design
approved explicitly in the active Goal. HF08 frozen at its old1/1 utility slot;
this audit only repairs/rechecks the evaluator and conditionally uses old actors.

Distinguish evaluator/RNG/initial-state variation from policy differences.
Decision: if noise cannot resolve5pp, close current route without new policy
experiments. Cheapest meaningful test reuses frozen e420 checkpoints and the
old terminal panel (training seeds286/287, evaluation seeds288/289).

## Frozen pre-execution contract

- Save every direct task tensor (including complete root positions, rotations,
  linear/angular velocities, DOFs, rigid/contact/force buffers, observation and
  history), physical actor properties, reset metadata, flags, RNN and full
  Python/NumPy/torchCPU/CUDA RNG. Restore only into a fresh simulator at frame0.
  No PhysX warm solver state is assumed restorable. All arms use matching
  engine/assets/configuration, GPU UUID and a verified shared snapshot.
- Deterministic native mean action with original normalization/clipping;
  no actor/value/Q/dynamics/optimizer update, teacher or extra noise. Separate
  actor checkpoint for each already-trained arm; RNN begins from saved state
  and then evolves under that arm's own observation (never copy off's later RNN
  to a different actor). Save actual actions before native PD mutation.
- Four plain-off panels,96 complete first episodes each, balanced three motions
  each32, frame0,1/30second control. Save full traces. Replay saved actions/RNG
  with a shadow actor to diagnose physics. Replayed applied actions must be
  identical; shadow error<=1e-5 would establish closed-loop equivalence. If
  that equivalence fails, this replay cannot decide the Goal: perform one
  actual closed-loop repeat per panel, applying native actor actions from the
  same snapshot/RNG. Its endogenous actions and RNN evolve independently and
  are saved. No off action is forced during actual closed-loop comparison.
- Primary stable success:>=3cm lift plus native hand/object contact proxy,
  consecutive45steps. Record all followup until first terminal, including
  drop-after-success (<2cm lift or6lost-contact steps), max hold, lift/contact.
  No reset may hide a terminal snapshot or erase post-success drop.
- Repeat noise gate on384 paired episodes: absolute success-rate difference
  <2.5pp; Wilson95% upper bound for per-episode success disagreement <5%, and
  analogous post-success-drop disagreement bound <5%. Full initial-state,
  checkpoint, RNG and terminal/motion pairing contracts must pass. The noise
  gate must use actual closed-loop repeats, or replay verified equivalent to
  closed-loop via the shadow check; replay alone is insufficient. This rules out net
  cancellation masking changed episodes. This screen is not a power guarantee
  or formal scientific validation; paired matrix confidence intervals reported.
- If any repeat gate/implementation contract fails, do not launch direct-Q or
  Cm-value evaluation. Report the actual failure and close the current route
  at this resolution boundary; retain all artifacts. No seed/simulator/metric
  search to pass this gate.
- Only if passed: once evaluate direct-Q/Cm-value actor checkpoints on the same
  four shared snapshots and common saved per-tick RNG. Reuse first plain-off
  results as the off arm. Existing fixed terminal gate retained: Cm >=+5pp
  versus each control, nonnegative versus each per training seed, all-episode
  drop fraction no worse than+5pp. A failure closes this Cm implementation;
  a positive Probe needs formal Validation before any utility claim.
- One admitted idle GPU,2CPU threads, whole run<=3600seconds, outputs/logs/cache
  <=8GiB, native process<=300seconds. Input hashes verified before/after; never
  overwrite old outputs or original checkpoints. Root outputs symlink unused.

## Engineering attempt r1

r1 FAILED before the first simulate/action/episode: Isaac Vec3/Quat/Transform
expose NumPy dtype metadata, and the generic property serializer recursed into
dtype.base. Corrected by explicit dtype serialization. Added a regression that
also excludes unspecified alignment padding in structured DOF property hashes;
actual named physical fields remain checked. CPU tests14passed; all six real
Isaac property classes serialize successfully in an Isaac-only CPU preflight.

r1 artifacts/log/cache retained; its recorded wall time and bytes are charged
to the same3600second/8GiB budget by r2. Scientific panel/state/actor/seed/gates
unchanged. New r2 output is unique; no native episode or policy result existed
in r1. r2 begins after fixed correction commit and fresh GPU admission.

## r2 symmetric initialization correction

r2 was deliberately stopped after startup, before any complete first-episode
export, when code audit found the first run skipped the full-state setter path
used by restored runs. All runs now perform identical native full reset,
full root/DOF setter, cold-cache restoration and verification. This prevents
setter-order changes from being mistaken for solver replay noise. Only the
verified owned PID/process group was terminated. The launcher recorded FAILED
from SIGTERM; this is an engineering stop, not a scientific result. No other
arm was started. r3 preserves the panel/gates and charges both r1/r2 costs.

## r3 replay diagnostic and closed-loop completion

r3 COMPLETED: eight plain-off processes,384 replay pairs, initial state/RNG/
applied-action contracts valid, original inputs unchanged. Success counts34/36,
2discordant success labels and2discordant post-success-drop labels; Wilson95%
upper1.879%. All first-run34 successes subsequently dropped. However shadow
actor errors.03361/.03267/.04052/.04813 exceed1e-5, so forcing actions is not
equivalent to repeating the closed-loop policy. r3's automatic UNPROMISING/
CLOSE decision is a failed surrogate contract, **not** a final Goal closure or
evidence that closed-loop noise exceeds5pp. No direct-Q/Cm-value was launched.

r4 corrects that incomplete surrogate by applying the actual native plain-off
actor once again in each frozen condition. It then uses the same2.5pp/Wilson5%
success/drop gates to decide whether the existing matrix can run. Panel, GPU,
checkpoints, native evaluator and thresholds remain fixed; no post-hoc split,
seed, simulator or policy changes. Saved action replay remains an independent
physics diagnostic. The Goal requires a valid closed-loop signal, so stopping
at the easier replay-only result would be incomplete.

Entire r1/r2/r3 cost is charged to r4. New r4 traces are losslessly gzip archived
after native completion, with compressed hash and decompressed byte hash
verified before removing only the redundant new raw file. Original HF08 data,
r3 reference snapshots/traces and checkpoint files remain unchanged. Full raw
observation/action/physical/RNG/RNN payload is retained, not subsampled.

## Terminal r4 outcome — actual closed-loop gate failed

r4 COMPLETED four actual native plain-off repeats, without forced actions.
Each reused the r3 first-run full cold snapshot and per-tick RNG schedule.
The independent completion audit verified all61 task tensors, actor physical
properties, reset metadata, stateless RNN representation, first observations
and first actions. Every applied action equals that run's native proposed
action; later actions evolve with its own observations. All384 pairs completed
their first episode and retained post-success followup and drop labels.

| Training seed | Evaluation seed | First/repeat successes | Changed success labels | Changed drop labels |
| --- | --- | --- | --- | --- |
| 286 | 288 | 12/16 of96 | 26 | 25 |
| 286 | 289 | 14/12 of96 | 22 | 21 |
| 287 | 288 | 5/3 of96 | 8 | 8 |
| 287 | 289 | 3/4 of96 | 7 | 7 |
| Total | | 34/35 of384 | 63 | 61 |

The aggregate success-rate difference is only0.2604pp, but63/384 success
labels change (16.406%), with Wilson95% upper20.440%; drop labels change61/384
(15.885%), upper19.877%. Both exceed the predeclared5% discordance bound.
One fixed panel's net success-rate variation is4.167pp. Changed-label fractions
are not net success-rate differences, and aggregate cancellation does not pass
the paired gate. This screen does not establish statistical power or a formal
negative Cm effect. Identical exposed initial state/RNG does not guarantee
identical subsequent native closed-loop trajectories; the cause of remaining
variation is not isolated here.

**UNPROMISING / CLOSE_CURRENT_ROUTE_EVALUATOR_RESOLUTION**. The evaluator's
state/trace contracts pass, while actual closed-loop repeatability fails.
The conditional direct-Q/Cm-value matrix was skipped; no V/PPO/Cm training,
new Cm mechanism, seed search, threshold change or baseline continuation ran.
Close the current HF08 implementation, keep Cm utility UNPROVEN/C3 OPEN and
self-trained baseline PARTIAL. HF08 and HD02 budgets remain1/1 consumed.

The cumulative native audit, including failed engineering attempts r1/r2,
took1321.37seconds (22.02minutes) on one admitted GPU4, with4.50GiB retained
artifacts, below60minutes/8GiB. All original checkpoint/data/config/motion and
frozen evaluator-source hashes remained unchanged. Owned native processes
exited and GPU4 was released. Lossless r4 archives were verified against raw
byte hashes; original r3 snapshots/traces remain available. CPU was used only
for file/hash/trace auditing and isolated engineering tests, not native model
evaluation. The16 focused state/RNG/trace/drop/archive tests passed.

Tracked evidence: [audited result index](P-20261001-paired-evaluator-resolution-results.json).
Native artifacts are under
`src/task/CmResidual/research/physical_value/output/P-20261001-paired-evaluator-resolution-r{1,2,3,4}`;
r4 contains `independent_completion_audit.json`, runtime manifests and the
four complete compressed traces. Native r3 code was1d063d1; actual r4 code
e9b72d3. The result index records input/output hashes and the completion checklist.
