---
schema: ref2dex.probe.v2
probe_id: P-20261007-consequence-baseline-rebuild
experiment_id: P-20261007-consequence-baseline-rebuild
date: 2026-10-07
task: consequence-evaluator
branch: consequence-evaluator
git_commit: 76fdd8c
claim_id: C3
hypothesis_family: HF-consequence-baseline-rebuild
probe_index_in_family: 1
seed_pool: probe
seeds: [289, 290]
decision_changed_if_positive: promote the newly trained parent to s3 transfer and the remaining expert recipe
decision_changed_if_negative: diagnose native geometry and learning before spending the six-expert budget
status: UNCLEAR
run_id: baseline-rebuild-20261007-r2
---

# Rebuild a self-trained rollout substrate after asset loss

## Motivation and Decision Note

The user explicitly authorizes retraining experts and regenerating continuous
rollouts when the old six checkpoints cannot be found. This supplies the robot
H/A/physical futures and reliable success/preference examples required by ref1.
It does not change Mission/C3 or reuse the missing baseline's validation claims.
Git recovery did not restore its outputs. Backups elsewhere remain possible;
the current bounded searches do not prove permanent loss.

The external read-only Ref2Dex project still has canonical filtered GRAB
geometric tensors and native Inspire/scene assets. Thirteen selected motions
(the twelve-task panel plus s1 airplane) pass shape/finite/left-contact checks.
The local DExplore data symlink pointed at the deleted baseline; its original
link is preserved, and assets/configs are copied into owned outputs. These are
imitation references, not robot evaluator data or evidence of grasp success.

Historical s3 direct scratch PPO failed. The historical expert was derived by
transferring a learned s1 actor, so first test a new s1 scratch parent with the
existing native Cm-off approach/held-lift/progress shaping and contact curriculum.
The old parent used a CmLite-trained initialization; this new Cm-off recipe is
an adaptation and its success is unknown. No official actor weights are loaded.

First native two-epoch engineering smoke: debug seed17, eight envs. If it fails,
do not start the parent fit. Then random-init seed289,64envs,horizon32,
minibatch256,lr1e-5,200epochs; reset contact/lift fractions.5/.25,
curriculum annealing40..80, save every20epochs. Use a fresh initialization after
smoke. One GPU, at most300s smoke/3600s parent fit/5GiB total outputs. Queue waits
at most3600s for the confirmed owned PointWorld launcher and requires COMPLETED
plus an empty GPU0 compute PID list before any Isaac/model initialization.
Other users' jobs are never stopped. New-stage runtime commit and all frozen
source/input hashes are recorded in the queue manifest; later docs-only commits
may follow the code revision above.

## Minimum evaluation and next decision

After training, separately evaluate the fixed endpoint under full native frame0
starts with early termination disabled, seed290,64first episodes. Primary task
qualification is >=3cm contact-proxy elevation sustained45frames with no later
drop (<2cm or >=6 lost-contact frames). Report counts and durations, not training
reward as success. Native net-force contact is a proxy, not pairwise collision
identity. If at least8/64 reliable successes are present, proceed to the new s3
transfer and six-expert construction; otherwise inspect geometry/control and
fit before expanding. This is an operational data-readiness gate, not a formal
validation claim or a replacement for phase/object coverage in evaluator data.

The six experts are planned as airplane_base/mixed12/train5/balanced5/duck/cup;
they receive new weight hashes and a new route config. The temporary s1 parent
is not counted as one of the six. Further expert fits and robot sampling remain
pending parent qualification. Never fill six route slots with the same parent
and claim the old substrate is restored.

## Artifacts and current status

- Inputs: `outputs/consequence-evaluator/baseline-inputs-20261007-r2/manifest.json`.
- Queue: `outputs/consequence-evaluator/baseline-rebuild-20261007-r2/run_manifest.json`.
- Detached queue PID870294 was confirmed live, initially WAITING for PointWorld.
  First shell-background attempt lost its process before any GPU stage;
  `baseline-rebuild-20261007/startup-check.json` records this execution failure.
- Preserve initial failed preparation in `baseline-inputs-20261007`; it copied
  assets before encountering the broken data symlink and never ran simulation.
- PointWorld completed normally. The r2 native smoke then failed before any
  PPO update: Isaac Gym imported torch_utils before the existing native NumPy
  alias compatibility shim, raising AttributeError for np.float. The wrapper
  now imports the Torch-free native bootstrap first; no installed library or
  external project is modified. Original smoke logs/FAILED manifest remain.
- Next attempt is `baseline-rebuild-20261007-r3`, on the now-empty GPU0 with the
  same staged inputs,2-epoch smoke then200-epoch parent and original fit/output
  caps. The queue also accepts its owned PointWorld COMPLETED/exit0 record;
  GPU occupancy is still checked before every stage. r2 produced no model
  updates and does not consume a scientific Probe or justify changing recipes.
- Native GPU smoke completion and learning evaluation are pending. No real robot
  examples, expert qualification or oracle headroom claim exists yet.

r3 passed the NumPy import but failed in the first environment reset, before
PPO updates: the CPU PhysX tensor pipeline was mixed with native CUDA reference
indices/tensors. The task-local wrapper now selects the GPU pipeline explicitly
to keep native simulation views, references and actor on CUDA. Original r3 logs
are retained. Next retry is baseline-rebuild-20261007-r4, GPU0only, the same
staged inputs and2/200epoch caps. User explicitly allocates GPUs1/2to separate
OakInk2 continuation; this expert run never shares those devices. No physics
success claim follows from fixing tensor placement.

## Limitations / future evidence

New motion references differ in provenance from the deleted corrected outputs;
CPU shape/filter checks do not verify contact geometry or physics replay.
The first parent is a bounded attempt, not guaranteed recreation of an old
checkpoint. The six-expert route, continuous phase-targeted perturbations,
reliable absolute progress, local preferences and matched E0/Eoracle training
still need actual execution and auditing. Later WM/proposal work remains
conditional on useful oracle headroom.
