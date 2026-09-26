# P-20260925-grab59-input-gate

- Classification: Blocker / engineering smoke for a wider self-trained
  baseline, not a Cm-effect experiment.
- Candidate pool: all 59 `_lift` or `_lift_Retake` trajectories in the
  existing read-only, single-right-hand
  `data/processed_data/inspire_geometric_dexplore` filter output.

## Question and decision

Can the complete currently filtered right-hand GRAB lift pool be loaded
and simulated together before starting a 59-motion shared actor? The raw
GRAB directory has 1335 sequences and 268 literal `_lift.npz` files;
the existing DExplore filter retains only 59 lift-like trajectories
across 29 objects. The 12-motion Probe used a curated subset, so it
does not answer this readiness question.

Stage local mesh symlinks for missing objects, create a frozen 59-motion
spec and audit every tensor for `[T,598]`, finite values, zero left-hand
contact labels, positive right-hand contact labels and nontrivial object
vertical range. Compare wrist/object-relative geometry with the 12
corrected motions as a provenance check. Then load all 59 in a 64-env
source-checkpoint full-episode smoke. A pass requires all 59 motions,
64 completed first episodes, finite evaluation metrics and no asset or
physics initialization error. If it passes, continue the self-trained
e260 actor for a short 59-motion Cm-off baseline Probe and evaluate on
heldout seeds. If it fails, fix the input before interpreting policy
training. CPU inventory plus one idle GPU, <=30 minutes, <500 MB output.

## Results

The CPU inventory staged all **59** filtered lift-like motions from **29**
objects. All tensors were finite `[T,598]`, had positive right-hand and
zero left-hand contact labels, and showed >3 cm reference object
vertical range. The 12 previously corrected motions matched the
filtered geometric references in wrist/object-relative position to
floating precision; contact labels differed in some frames, so the two
producers remain distinct and provenance is explicit.

Local symlinks to the original GRAB meshes were added for 18 objects
whose DExplore assets were missing. Existing object meshes matched raw
source hashes and were untouched. Inventory, mesh hashes, frozen
motion spec and staged links are in
`outputs/CmResidual/agent_grab59_input_gate_20260925_r3/`.

The source e260 actor was then loaded on all 59 motions with GPU6,
seed220 and 64 environments. The simulator completed 64 first full
episodes, produced finite metrics and recorded a 59-motion object map.
Held-lift was **3/64** and mean hand-object contact fraction 0.1572;
these are smoke diagnostics, not a baseline efficacy result. The
engineering gate **passed**, allowing the predeclared 59-motion
Cm-off continuation Probe. The 59 motions are the complete *currently
filtered single-right-hand lift pool*, not the entire GRAB dataset.

Smoke artifacts: `outputs/CmResidual/agent_grab59_source_smoke_s220/`.
