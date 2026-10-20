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


### Corrected full cache result

Frozen da9f5b8b completed `outputs/wam-rl-policy-v3/inspire-scheduled686-cache-20261020-r2/`
on GPU6/7 in 406.14 s including process startup. All 563 admitted prefixes and
60,529 row identities are retained; 52,660 TRAIN and 5,693 DEV rows are active
and WAM-valid. Inactive/startup/workspace-invalid rows stay zero/invalid. Every
predicted input checks independently packed task6 against WAM point displacements;
maximum discrepancy was 7.613e-8 m (limit 1e-6). Both shard receipts bind the
original measured admission and the same frozen v4 step80000. No teacher data
were recollected and no neural checkpoint was overwritten. These counts are
conditioning admission, not manipulation success.

## Latest admission r3 regeneration: r4

User requested adding WAM to the completed corrected-data goal BC. Run
inspire-scheduled686-cache-20261020-r4 rebuilds all62605 row conditions from
inspire-teacher-rollout-audit-20261020-r3 (admission schema v2, corrected table
support); it does not reuse old60529-row tau. Frozen WAM v4 step80000 and the
existing Inspire geometry binding stay fixed. task10 and WAM share the same
TeacherRolloutCorpus/TaskController decision; no DExplore training reference.
GPU6/7 disjoint shards, graspenv Python3.8/Torch2.4.1+cu121, main/frozen code
commit in manifest, <=6h deadline, owned STOP,20GiB disk floor. Cache plus BC
and bounded no-video evaluation outputs are limited to500MiB this run.
Invalid/inactive/startup rows remain zero with explicit validity; Scheduled
training keeps all task-active commands, including invalid tau rows.

### r4 completed

Frozen5a790409 completed both shards on GPU6/7 in440.59s. All563 sequences /
62605 commands are covered (TRAIN56418/DEV6187); WAM-valid active rows are
54534/5895. TRAIN1502/DEV186 active invalid rows remain real BC supervision,
with zero tau and validity false. Independent task10/point-goal discrepancy
maximum7.61315e-8m. Receipts bind the same source r3 and frozen WAM80000.
Cache size64MiB; no old tau, raw data, checkpoint or external source was changed.
