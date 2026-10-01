# Holding baseline continuation launched

Run `P-20261001-hold-plateau-curriculum-r1`, code `e770d4b`, parentPID1158326,
owned training processPID1158398, freshly admitted GPU5. Full manifest/commands
and unique cache/checkpoint roots are under
`src/task/CmResidual/research/contact_response/output/P-20261001-hold-plateau-curriculum-r1`.
Interactive supervisor handle90987 is only runtime identity, not research identity.

Thirteen focused checks pass for hold scoring, curriculum schedule, phase-only
reward and compiled/uncompiled checkpoint identity. Prelaunch inspection caught
the source checkpoint's `_orig_mod.` wrapper; weights are loaded canonically and
source/final inference fingerprints are compared without wrapper artifacts.
All actual PPO/model inference uses GPU. CPU reads only metadata/hashes and
event labels. Training starts correctly at new epoch1, with a fresh optimizer,
copied source weights/normalizers, and the predeclared5mm wrist exploration.

The prior feasibility run retains FAILED reporting metadata. Its four native
panels all completed in394.44s, total400.79s/16.00MiB. Separate corrected analysis
at code3f4ae77 verifies all156 frozen inputs and768 episodes, producing5 retained75.
No failed metric, seed, actor or code identity is replaced.

Current scientific status of the curriculum is pending. Training diagnostics
from elevated resets are explicitly not frame0 evaluation. Only final300-epoch
checkpoint and fresh500/501 first episodes can decide the frozen feasibility gate.
