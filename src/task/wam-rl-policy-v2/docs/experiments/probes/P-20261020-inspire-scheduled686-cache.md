# P-20261020-inspire-scheduled686-cache

This probe regenerates WAM conditions for the audited measured teacher corpus.
The task path and scheduler state come from each sequence's same-execution
teacher object rollout and `TaskSpec`; the old cache is read-only geometry
binding only.  The v4 checkpoint is frozen and has no trained CONTACT/RELEASE
event input, so this run is the v5-compatible spatial-goal baseline.

The output is flat, row-addressed `tau` (264 values) and validity arrays. Two
workers use disjoint sequence indices (`index % 2`) on GPU6 and GPU7. Invalid
workspace rows remain zero with validity false and are excluded by the later
scheduled BC admission. This is a cache generation probe, not a control or
grasp-success result.


## Result (2026-10-20)

Run `inspire-scheduled686-cache-20261020-r1` completed with two disjoint
workers in 339–341 s.  It produced 60,529 row-aligned conditions: 53,042 TRAIN
and 5,799 DEV rows passed WAM workspace validity.  The cache uses v4 step80000
(identity recorded in `manifest.json`) and consumes no event token.


## Shared TaskController goal correction (2026-10-20)

The r1 cache used a separate geometric scheduler, so its goals can differ from
the actor's task10, especially during ACQUIRE/recovery/release. Retain r1 and its
BC checkpoint as historical artifacts; do not reuse them as corrected conditioning.
Run `inspire-scheduled686-cache-20261020-r2` regenerates only tau/valid from the
same immutable raw admission. Local poses and task10 are read from the same
TeacherRolloutCorpus causal replay; no raw teacher recapture is required.
Layout v2 records the shared-goal contract and completion receipts bind source
admission and frozen WAM. Two disjoint workers use GPU6/7, bounded to 21600 s each.
This is an engineering repair of conditioning, not a WAM-benefit experiment.
