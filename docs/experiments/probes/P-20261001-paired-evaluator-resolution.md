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
  each32, frame0,1/30second control. Save full traces. Repeat each once from its
  snapshot using off's saved action/RNG trace and a shadow plain-off actor forward.
  Shadow RNN evolves independently. Repeated applied actions must be identical.
  Shadow versus saved policy action maximum absolute error<=1e-5 is required
  to treat trace replay as equivalent to a closed-loop actor repeat.
- Primary stable success:>=3cm lift plus native hand/object contact proxy,
  consecutive45steps. Record all followup until first terminal, including
  drop-after-success (<2cm lift or6lost-contact steps), max hold, lift/contact.
  No reset may hide a terminal snapshot or erase post-success drop.
- Repeat noise gate on384 paired episodes: absolute success-rate difference
  <2.5pp; Wilson95% upper bound for per-episode success disagreement <5%, and
  analogous post-success-drop disagreement bound <5%. Full initial-state,
  checkpoint, applied-action, RNG and terminal/motion pairing contracts must
  pass, as must the shadow closed-loop equivalence check. This rules out net
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
