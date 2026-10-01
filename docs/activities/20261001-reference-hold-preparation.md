# Hold-task preparation checkpoint

Completed source-label auditr2adds the actual reference reader's hash
(`base_dexplore_task.py`);r1is retained. Source object columns198:201/contact205
and native joint columns373:391match the reader. All source data remain immutable.

Prepared synthetic reference data at
`src/task/CmResidual/research/contact_response/output/P-20261001-hold-plateau-reference-r1`.
Generation commitb2b26e0,0.071sCPU,4,677,907bytes; three generated tensors with
recorded SHA. Peaks72/50/62, inserted frame ranges73--162/51--140/63--152.
All598columns stationary, initial and remaining original rows preserved exactly.
No physical trial or new model training has used these data yet.

The native reader recomputes object, wrist/finger, body and rotation velocities
from the resulting trajectories. Upcoming collector must verify stationary
plateau velocities on the ACTUALLY loaded GPU tensors and record progress/tick
correspondence, not rely solely on raw-copy assertions. Full first episodes
must be completed to the new reference lengths633/743/578frames, with900-step
collector cap. Model utility is deferred until actor-only holding feasibility.

Tiny three-pose geometric engineering check uses the current shared bridge on
CPU: raw peak full-surface gaps0.9135/0.9456/0.4883mm. CPU is appropriate here
because only three static label poses are checked and GPU startup dominates;
native simulation/model inference for the actual Probe will use an admitted GPU.
Unsigned sampled proximity does not establish nonpenetration, frictional grip,
attributed contact force or physical holding success.

Paper revision4includes3072closed-loop episodes, failed retained gate and source
reference mismatch. Nine tables plus one source-label figure are generated from
recorded data. Nine-page review PDF/source/PDF hashes and numeric/code-identity
text checked; table captions are grouped with tables, figure layout inspected.
Earlier draft renders remain preserved. Journal readiness still NOT READY.

No own GPU/native job remains running. Next implement the frozen actor-only
hold-phase collector and its independent75-step physical scoring before launch.
