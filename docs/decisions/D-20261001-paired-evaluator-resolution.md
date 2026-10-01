# Decision — paired evaluator before further Cm experiments

User objective: freeze HF08, stop V/PPO and new Cm mechanisms, establish whether
the evaluator can resolve5pp before any existing-checkpoint three-arm inference.
Self-trained grasp remains PARTIAL; no baseline continuation belongs to this audit.

Evidence: old same-seed plain-off e0 audit12/96 ->10/96,18 discordant episode
labels; old endpoint Cm33/384 versus off41/384 is inside unresolved execution
variation. The completed V-only diagnostic does not remove this problem.

Action: implement cold-state paired evaluation. Save/reload every exposed task
tensor, all actor root/DOF states, actor physical properties, observation/history,
RNN and Python/NumPy/CPU/CUDA RNG. Each arm starts in a fresh process and a
never-stepped simulator; PhysX solver caches cannot be serialized, so warm-state
restoration is explicitly rejected. Preserve complete action/RNG/physical traces
and post-success drop. No teacher or actor/model updates.

Run one fixed repeatability audit at the existing two e420 plain-off checkpoints,
evaluation seeds288/289,96 first episodes per panel. Repeat the saved action trace
once per panel, recording a shadow actor forward to detect closed-loop mismatch.
Only after all repeatability contracts pass run the two other existing actor-only
arms on those same snapshots/RNG schedules. This is one endpoint matrix, no
checkpoint selection, new training or inference Cm mechanism.

Budget:1idle GPU,2CPU threads, entire audit<=60minutes, new artifacts<=8GiB;
per native process<=300seconds. Original checkpoint/data/result hashes frozen.
Gate/stop thresholds are in the pre-execution experiment card. If repeatability
fails, stop without a three-arm policy experiment and close this implementation
under evaluator-resolution limits. If the gated matrix fails the existing
two-control5pp/drop/nonnegative-seed gate, close HF08. Probe conclusions only;
Cm utility stays unproven and baseline PARTIAL. No external authorization needed.

Terminal decision (2026-10-01): r3 saved-action replay failed the shadow actor
equivalence contract, so it could not close the Goal. r4 completed one actual
closed-loop off repeat per frozen panel, under the same original budget and
unchanged thresholds. Full initial-state/RNG/trace/drop contracts passed.
Success counts34/384 and35/384 conceal63 changed success labels and61 changed
drop labels; their Wilson95% upper bounds20.44%/19.88% fail the5% noise screen.
The gate is conservative episode agreement, not a formal power guarantee or
claim that net success-rate noise equals the discordance fraction.

Close the current HF08 implementation and HD02 diagnostic. No three-arm
inference stage was started. Preserve all evidence; no additional training,
Cm mechanism or baseline epochs are authorized by this audit. Cm utility stays
UNPROVEN/C3 OPEN, baseline PARTIAL. Cumulative cost22.02minutes/one GPU/4.50GiB;
all original inputs unchanged, owned processes terminal and GPU released.
Any future Objective A work belongs to a separately scoped baseline/curriculum
route. See the experiment card and tracked audited result index for evidence.
