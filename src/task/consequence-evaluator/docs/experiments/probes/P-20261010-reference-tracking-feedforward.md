---
schema: ref2dex.probe.v2
probe_id: P-20261010-reference-tracking-feedforward
experiment_id: P-20261010-reference-tracking-feedforward
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 91550b9
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 2
seed_pool: probe
seeds: [273, 274, 275]
decision_changed_if_positive: retain velocity feedforward and fine-tune a fixed tracker before removing robot reference oracles
decision_changed_if_negative: distinguish policy distribution change from missing contact learning before another bounded control experiment
status: PROMISING
run_id: ref7_3-tracker-feedforward-train-20261010-r2
---

# Wrist velocity feedforward for sustained reference holding

## Motivation and Decision Note

Serves the C3 execution subproblem, without changing Mission or Cm claims.
User requested autonomous resolution of the diagnosed unstable drops. The raw
motion contains placement, while the measured teacher reference used here ends
in an airborne grasp; these are separate task objectives. This card addresses
holding under the reference actually used. Passing this gate does not complete
original placement or prove tau-to-action deployment.

Hypothesis: native position-only PD targets around measured next q omit the
reference velocity term. With wrist Kp200/Kd20, adding 0.1*dq_ref recovers
Kp*(q_ref-q)+Kd*(dq_ref-dq). Independent frozen-source audit reduces wrist xyz
PD target MAE from14.34/18.69/24.04mm to3.06/4.29/6.27mm using position-derived
central differences. This is a damping compensation, not inverse dynamics.
Teacher commands are used only in this diagnostic, not learning or inference.

Classification: Decision. Cheapest intervention: keep the existing policy
frozen and run complete four-arm native episodes with only wrist feedforward
changed in the intervention arm. Positive -> fine-tune with corrected control;
negative -> inspect whether old feedback is incompatible with the new base,
then decide a bounded training comparison; do not refute feedforward based on
one distribution-shifted frozen policy. No broader architecture change.

## Frozen protocol

Same native runtime, input manifest and measured543-state reference as
[P-20261010-reference-tracking](P-20261010-reference-tracking.md).
Initial tracker checkpoint is its fixed final.pt, SHA256
b1b609a0919dd7ef3217883f59bfc3bfae173549187f4651764d72b6b7c0c02c.
64 environments: teacher16, nominal16, frozen tracker16, frozen tracker+FF16;
roles randomly permuted with seed273. Full542 controls, identical initialization.
Only intervention wrist targets receive (Kd/Kp)*reference_velocity[t+1].
Native gains are read from actual actor properties; derivative uses actual
control dt, wrapped generalized wrist rotations, one-sided episode endpoints.
Finger base/residual, features, reward, bounds, coupling and native adapter stay
as before. Reference velocity is computed from positions, not commanded labels.
Role rows remain descriptive, not independent seeds or exact PhysX forks.

Container: >=8/16 teacher hold>=45. Positive screen: >=8/16 intervention reach
>=90% of archived held481, no recorded intermediate loss in qualifying rows,
median held >= old tracker median+45, and clipping<1%. Also report terminal
holding and actual separated airborne intervals; supported states are excluded
by legacy intermediate-loss count, so it cannot establish absence of drops.
Otherwise UNPROMISING for this frozen-policy intervention, not the method.

If a training run is justified, seed274, initialize from old fixed final;
128updates x32steps x64env, same PPO/reward/features, feedforward base enabled,
new initial/final checkpoint paths. Fixed final, no evaluation selection.
Evaluation seed275 teacher16/nominal+FF16/trained+FF32; near-teacher screen
>=16/32 >=433frames, median>=nominal+45, clip<1%, teacher container valid.
Any action after the frozen comparison is recorded before training.

## Budget and stops

One idle GPU2, >=20GiB free, <=512MiB passive contexts and <=10% util;
preserve foreign processes, stop own run on new heavy foreign GPU user.
Smoke<=120s, first comparison<=300s, training<=1200s, finaleval<=300s:
<=32 GPU minutes total, <=2GiB outputs. Stop on nonfinite, drift, mismatch,
unexpected early terminal or deadline. Preserve all runs; no unbounded sweep.
This is new scoped authorization within global Campaign limits, not reuse of
old completed experiment's budget.

## Limitations / future evidence

Robot-q/object-future oracle, one motion and short single-seed training. Original
placement requires a coherent raw motion hand/q/object reference and phase-aware
controlled-placement evaluation; never mix raw object targets with teacher hand
motion. No tau-only, generalization or formal superiority claim. Further contact
learning or original-task training needs its own decision-serving protocol.

## Runs

Frozen comparison r1 (code c9e3604, seed273) completed the full542 controls.
Teacher container passes8/16 hold45. Old tracker median285, near433 count0/16,
terminal held0/16. Tracker+FF median479.5, near433 count13/16, terminal held13/16,
zero clipping; hand RMSE11.88mm versus29.61mm, object RMSE52.58mm versus212.46mm.
Predeclared frozen intervention screen PROMISING. Two intervention rows have
legacy intermediate loss events; qualifying rows are checked individually.
Initial q/dq/hand/object are bitwise aligned and all native PD reconstruction
errors are zero. Randomized roles still describe one launch, not Validation.
Artifacts: outputs/consequence-evaluator/ref7_3-tracker-feedforward-{eval,audit}-20261010-r1/.

Decision before training: the intervention passes the original strong screen
without changing policy weights. Retain velocity feedforward and run the single
predeclared seed274 warm-start128-update fine-tune, then fixed final seed275
evaluation. This tests whether feedback/preload adapts to the corrected base;
no parameter sweep or checkpoint selection. Preserve the frozen corrected
controller independently even if fine-tuning degrades it. Expected~8 GPU minutes
training plus~2 evaluation, within32-minute card budget. Stop conditions unchanged.


Training r1 (a5a0a35) is INVALID_IMPLEMENTATION for training: after the first
PPO update, the local likelihood `ratio` shadowed the wrist Kd/Kp variable
captured by the step closure, causing a6-vs512shape exception. No final
checkpoint was produced; preserve FAILED manifest, initial checkpoint and log.
The frozen evaluation has no PPO loop and remains unaffected. Fix uses an
explicit wrist_gain_ratio identifier. A two-update engineering run must cross
the PPO boundary before retrying the same predeclared full protocol.

Engineering training smoke (91550b9, debug seed42,8env,2updates) completed
in14.48s, with changed parameters and finite outputs; crosses the formerly
failing PPO update boundary. This is not a scientific sample. Full train r2
retries unchanged seed274/64env/128updates into a fresh output.
Independent command-contract audit reproduces the actual reference + residual
+ wrist feedforward before native encoding to2.38e-7max error; native target
reconstruction remains exactly0. Updated audit output suffix r2; remaining3/16
terminal-lost intervention rows all show unsupported gravity-like separation.
Drop details: outputs/consequence-evaluator/ref7_3-tracker-feedforward-drop-20261010-r1/.


## Fixed final controller result

Full training r2 (91550b9, seed274) completed128updates/262144transitions in
453.20s. All13 learned tensors changed; training clipping1.398%. Native GPU
memory stayed about7.3GiB, sampled utilization4–40%, CPU exchange/simulation
bottleneck as before. Final checkpoint SHA256:
20073fc165e13f00f4e8f9f7b7f25d8b686faa3d6e5b98dbc8a6a9e867fe16b5.
No checkpoint selection or extra hyperparameter run.

Final evaluation r1 (b995a42, seed275) completes542controls/543states in86.41s.
Here nominal means the **same feedforward base with zero learned residual**.

| Role | Rows | Median held | Hold>=45 | Hold>=433 | Terminal held | Clipped steps |
| --- | --- | --- | --- | --- | --- | --- |
| Owned teacher | 16 | 484 | 12 | 12 | 12 | 0 |
| Reference + wrist FF, zero residual | 16 | 92 | 12 | 4 | 4 | 16 |
| Fine-tuned reference + wrist FF tracker | 32 | 483 | 32 | 31 | 31 | 76 |

Final predeclared screen is PROMISING:31/32 qualifying, median391frames above
nominal,76/(542*32)=0.438%clip<1%, valid teacher container. Qualifying rows have
no legacy intermediate loss. Native target reconstruction maxerror0, actual
reference/residual/feedforward command contract maxerror2.38e-7, initial
q/dq/hand/object bitwise aligned. This is not Validation or a general method
claim. Final role layout is contiguous as predeclared; do not treat its31/32
versus earlier randomized13/16 as a formal matched fine-tuning effect.

Remaining env38 last held450, separates451: hand gap11.95mm, object/table
clearance142.99mm, vz=-0.473m/s, pairfalse; separated451–454 gravity-like
intervals451–453. The measured reference remains held with176.42mm table
clearance. Thus one remaining failure is a true loss, not desired placing.
Do not claim32/32 stability or original-task completion.

Artifacts under outputs/consequence-evaluator/:
- ref7_3-tracker-feedforward-train-20261010-r2/final.pt
- ref7_3-tracker-feedforward-final-eval-20261010-r1/ (trajectory, manifests, results)
- ref7_3-tracker-feedforward-final-audit-20261010-r1/ (audit.json, behavior.png)
- ref7_3-tracker-feedforward-final-drop-20261010-r1/ (remaining loss diagnostic)
- ref7_3-tracker-feedforward-controller-20261010-r1/controller.json (checkpoint,
  reference hashes, explicit feedforward flag and complete reproduction argv).

The saved policy is only one component: evaluation must enable the declared
wrist feedforward, which is now enforced when loading a feedforward-trained
checkpoint. Loading weights without this base is a different controller.
Total actual GPU work including failed training/smoke/evaluations is about11
minutes, artifacts under32MiB, within32minutes/2GiB. Preserve all originals.

Decision after result: retain corrected control and the fixed fine-tuned
checkpoint. Measured-robot reference execution is now PROMISING at the original
strong holding gate. Next meaningful route step is replacing robot-q oracle
with11-point tau retargeting, while defining raw-motion controlled-placement
reference/metrics coherently. No additional contact sweep or Validation under
this card. Future evidence: remaining failure, randomized matched repeats,
held-out motions/seeds, raw placing and removal of robot/object-future oracles.
