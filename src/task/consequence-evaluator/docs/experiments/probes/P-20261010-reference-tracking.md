---
schema: ref2dex.probe.v2
probe_id: P-20261010-reference-tracking
experiment_id: P-20261010-reference-tracking
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: a2ea730
claim_id: C3
hypothesis_family: HF-reference-tracking-control
probe_index_in_family: 1
seed_pool: probe
seeds: [271, 272]
decision_changed_if_positive: implement 11-point tau retargeting after the robot-reference control upper bound passes
decision_changed_if_negative: inspect control feasibility before investing in tau retargeting or more inverse-action L1 fitting
status: UNPROMISING
run_id: ref7_3-tracker-train-20261010-r1
---

# Does reference-conditioned residual RL recover native object holding?

Result: UNPROMISING for the predeclared near-teacher screen; transient holding improved (nominal median0, tracker286.5, teacher483), but all tracker rows lose terminal hold.
Decision: Retain the controller for a later targeted sustained-contact Probe; do not yet replace the robot-q oracle with 11-point retargeting or declare tau-to-action complete.

## Motivation and Decision Note

The ref7_2 inverse-action model has low held-out L1 but fails the native holding
screen. The ref7_3 source review motivates a reference-conditioned controller
trained on actual interaction/object tracking. This Probe serves the execution
subproblem of C3; it does not test Cm-on/off utility or alter the Mission claim.
The user approved this minimum Probe on 2026-10-10 after the source review.

Question: Is bounded residual PD-target RL around an accurate measured robot
reference worth continuing before spending on geometric tau retargeting?
Cheapest discriminating test: one native Inspire/Gym motion, single short PPO
run, followed by a frozen complete-episode teacher/nominal/tracker screen.
Classification: Decision; stage: Probe. Positive -> build tau retargeting;
negative or unclear -> inspect the relevant controller/learning limitation.
No multi-seed Validation or formal route superiority claim is authorized here.

## Frozen contract

- Runtime: graspenv Torch 2.4.1+cu121, Isaac Gym Preview4, CPU tensor exchange /
  GPU PhysX. Keep the native Inspire PD stiffness, damping and coupling.
- Inputs: `outputs/consequence-evaluator/hand-execution-inputs-contact-capture-20261010-r1/inputs.json`,
  its owned e260 baseline and single `s3_airplane_lift` motion.
- Reference: measured env0 robot q, 11 hand points and object poses in
  `outputs/consequence-evaluator/ref7_2-same-cpu-teacher-contact-packet-20261010-r1.pkl`
  (seed282; 543 states; reference held481). Stored commanded teacher actions
  are omitted from training/inference. This old engineering packet's
  `training_allowed=false` prevents reusing it as a ref7_2 supervised split;
  the current user authorization permits using its measured states as a new,
  explicitly privileged control reference. No ref7_2 held-out score is claimed.
- Action: next measured reference q + a bounded 12 independent-coordinate
  residual; apply native coupling and exactly invert the native PD adapter to
  full 18-D commands. No baseline actor future controls enter tracker rows.
- Features: live q/dq, 11 hand points, object pose/velocity, next reference q
  and object error, fixed 24-step future hand in live object frame, previous
  residual. Future hand is never recentered onto live hand. Robot q and future
  object are oracles in this first upper-bound experiment.
- Reward: object translation/orientation tracking, relative hand tracking,
  weak actual-lift/net-force-pair hold proxy, small residual penalty. Actual
  joint position is not equated to PD preload. Dense surface/support geometry
  is measured separately for episode evaluation.
- Actor: 909->128->128->12 tanh MLP, zero final layer, random critic;
  latent Normal PPO, gamma .99, lambda .95, clip .2, Adam 3e-4, four epochs,
  minibatch512, rollout32. No command-label imitation or official actor init.
- Fixed final checkpoint, 128 updates x32 x64 =262144 environment transitions
  (4096 simulator steps). No best-checkpoint selection on evaluation.
- Training seed271; evaluation seed272. Engineering smoke seed41 belongs to
  debug pool and is excluded from Probe outcomes.

## Budget and stop conditions

One otherwise idle GPU (initial choice physical GPU2). Tiny passive foreign
contexts <=512MiB with <=10% utilization may remain; preserve all processes,
require >=20GiB free, stop our run on a new heavy foreign process.
Training <=1200s; smoke <=120s; one evaluation <=300s. Total <=27 GPU minutes;
outputs <=2GiB. If the fixed update budget cannot finish, retain FAILED run
and do not quietly shorten it or resume with a different protocol.
Stop for nonfinite tensors, input drift, wrong backend, unexpected terminals,
or requested/applied command mismatch. Preserve every failed artifact. Fixing
a demonstrated engineering defect may use a new run ID under the same budget;
no additional hyperparameter sweep or task expansion.

## Predeclared screen

Evaluation: 64 environments, 542 controls, teacher16 / nominal16 / tracker32.
Roles are separate execution rows, not exact PhysX state clones or independent
scientific samples. Same launch and reference; no significance tests.

Container must produce at least 8/16 teacher episodes with >=45 consecutive
held frames. Otherwise behavior attribution is UNCLEAR. PROMISING requires
at least 16/32 tracker episodes with >=90% of archived teacher held481,
tracker median held exceeding nominal by >=45 frames, fewer than 1% clipped
tracker steps, and no recorded intermediate loss in the qualifying episodes.
Otherwise, if the container is usable, mark this fixed-budget controller Probe
UNPROMISING; this does not refute reference tracking as a method. Final
placement is reported but is not the screen because the source has no final
settled placement. Report actual held/loss/geometry rather than training loss.

## Limitations and future evidence

Single motion, privileged robot/object reference, synchronized training phase,
short training and one seed. Role rows are contiguous and show repeated layout
patterns; they are not randomized, matched counterfactuals or independent seeds.
No tau-only mapping, held-out motion generalization,
same-state fork, Cm policy benefit, deployable object-future input, or formal
comparison to DexTrack/REGRIND. If positive, the next Probe replaces robot-q
oracle with geometry retargeting from the declared 11-point tau; only then
consider H/PW proposals and matched comparisons.

## Runs and results

Smoke r1 reached physics execution but stopped at tick1 because native reset
returns an observation dictionary while env_step returns a tensor. This is an
interface defect, not a controller result; preserve its FAILED manifest/log.
The fix is `a2ea730`; smoke r2 passed all 8 controls. Both raw runs remain in
`outputs/consequence-evaluator/ref7_3-tracker-smoke-20261010-r{1,2}/`.

Training r1 (`a2ea730`, seed271, GPU2) completed 128 updates /262144 environment
transitions in453.97s, changing all13 learned state tensors. Fixed final
checkpoint SHA256 is
`b1b609a0919dd7ef3217883f59bfc3bfae173549187f4651764d72b6b7c0c02c`.
Training clipping rate1.012%; no nonfinite or requested/applied mismatch.
GPU usage stayed about7.5GiB, observed utilization1–35%; CPU tensor exchange
and native simulation dominate this small model. Final checkpoint is
`outputs/consequence-evaluator/ref7_3-tracker-train-20261010-r1/final.pt`.

Evaluation r1 (`c8bdb05`, seed272) completed543 states/542 controls in75.56s.
The change from training to evaluation code only added diagnostic state/PD
target recording; model, feature/reward/action contracts are unchanged.

| Role | Rows | Median held | Hold>=45 | Hold>=90% reference481 | Terminal held |
| --- | --- | --- | --- | --- | --- |
| Owned teacher | 16 | 483 | 12 | 11 | 11 |
| Nominal robot-q reference | 16 | 0 | 0 | 0 | 0 |
| Trained residual tracker | 32 | 286.5 | 32 | 0 | 0 |

Tracker held range57–288, mean233.97; zero clipped evaluation controls. Its
mean hand-coordinate RMSE29.68mm is similar to nominal28.97mm despite the
holding difference. This is a descriptive single-launch control signal, not
formal causal superiority or generalization. All tracker objects eventually
return to the table; controlled final placement is false. The legacy outcome
reports zero intermediate loss events because supported states are excluded
from that loss definition. **Do not interpret this as zero drops.**

The teacher container passes (12/16 hold>=45). The strong predeclared screen
fails (0/32 reach>=433 consecutive held frames), hence `UNPROMISING` for this
fixed-budget controller screen. It does not refute reference tracking as a
method; the transient lift/hold signal warrants preserving this implementation.
Do not lower the gate after seeing the result.

Read-only independent implementation review found no clear PD inversion,
coupling, reference/frame, reward, PPO/GAE or dispatch defect. Post-run audit
also confirms initial q/dq/hand/object bitwise alignment across64 rows and
exact reconstruction of every captured native PD target (maximum error0).
Per-update reward is a synchronized32-tick phase slice, not a phase-matched
learning curve; training revisits the complete trajectory only about7.6times
per environment.

Raw evaluation: `outputs/consequence-evaluator/ref7_3-tracker-eval-20261010-r1/`.
Screen/contracts/figure: `outputs/consequence-evaluator/ref7_3-tracker-audit-20261010-r1/`.
Reproduce the audit with `tools/audit/audit_reference_tracking.py` and the
training/evaluation/reference paths above, choosing a fresh output directory.

An independent offline diagnostic using teacher commands **not as model
inputs or training labels** finds that matching the teacher PD target around
measured next q would exceed the chosen wrist y/z residual limits in9.04%/7.20%
of source ticks (artifact `ref7_3-tracker-reference-capacity-20261010.json`).
This limits the upper-bound interpretation but does not establish the observed
drop mechanism: during ticks340–364 the teacher wrist target residual remains
within the chosen limits, and the learned residual is not saturated. Do not
claim that increasing bounds alone will recover the late hold.

Decision after result: keep this control route as a more informative next
direction than another inverse-L1 sweep. Before tau retargeting, the next
decision-serving experiment should separate insufficient late-phase/contact
learning coverage from the controller/reward objective. No additional sweep is
run under this card. Needed future evidence: sustainable late hold, removal of
robot-q/object-future oracles, randomized role/layout or matched repeats, and
held-out motion/seed Validation. Total executed GPU work was under10 minutes;
run outputs are under10MiB, plus about3.4MiB reusable build cache.
