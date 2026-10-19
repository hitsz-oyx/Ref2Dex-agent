# P-20261020-inspire-current421-bc

## Decision

Determine whether the audited same-execution DExplore teacher corpus can train a
Current421 Inspire behavior-cloning executor before adding WAM `tau`.  A usable
Current BC checkpoint changes the next step to closed-loop BC evaluation and
then scheduled BC; a failure returns to the data or observation contract.

## Fixed protocol

- `task`: `wam-rl-policy-v2`
- `experiment_id`: `P-20261020-inspire-current421-bc`
- `run_id`: `inspire-current421-bc-20261020-r1`
- `branch`: `main`
- `git_commit`: recorded in the run manifest
- source: `outputs/wam-rl-policy-v2/inspire-teacher-rollout-audit-20261019-r2/manifest.json`
- goal source: same-execution checkpoint teacher object rollout
- input: measured347 + learned CAD shape64 + task10 = Current421
- output: 12 active Inspire commands
- model: Shadow-capacity 1024/1024/512/512 ELU MLP with trainable PointNet shape64
- optimizer: AdamW, learning rate `3e-4`, weight decay `1e-5`, batch `2048`, 50,000 steps
- GPU: the run owns GPU6; it must stop at the eight-hour wall-clock limit or an owned `STOP`

## Admission and evidence

Only records accepted by the immutable measured-rollout admission manifest are
used.  The trainer loads the same-tick force, state, task and action arrays from
`TeacherRolloutCorpus`; it does not use the old reference-goal cache and does
not fabricate `tau`.

The primary metric is DEV command MAE, with root and finger MAE reported beside
it.  This is an offline probe, not a grasp success result.  Closed-loop BC and
PPO remain separate decisions.
