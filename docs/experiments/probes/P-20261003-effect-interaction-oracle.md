# P-20261003-effect-interaction-oracle

Decision Probe, explicitly authorized by user. Question: can perfect short
effect + attributed interaction information improve actual grasp selection
over current state/action alone and effect alone? Positive joint gain licenses
coupled Cm learning; failure diagnoses this fixed oracle/control recipe and
does not refute all contact priors. Cheapest actual utility test: fixed-option
task-Q ranking, with no learned Cm and no full PPO run.

Engineering prerequisite: native CPU read pipeline + GPU PhysX. Fresh same
48-env scene/seed751 replay is bit-identical for ALL72 ticks, including full
27-body state and net forces; attributed normal-force reconstruction max
1.418e-6N. Raw lambda is force-valued, on body0 along normal; friction
components remain unqualified, not policy input. r1 failed on source CUDA
reference placement; r2 device-only fix passes. No post-frame0 state setters.

Fixed design before scientific collection:

- One own frozen scratch P0 (15ed218f), corrected Inspire shape filters,
  nominal airplane mass, same three synthetic references,96envs,202ticks.
- TRAIN scene seed761; EVAL scene seed762;32whole episodes/motion per scene.
  All scenes retain the same env indices/layout and native physics settings.
- Common decision tick36 (before first lift44); action-conditioned horizon32
  controlticks (1.067s), reaches all three lift starts by63. It is one fixed
  preparation decision, not horizon tuning or terminal-success oracle.
- Eight globally fixed options: zero and7 Gaussian12D requests sigma.25,
  generated once with NumPy RandomState805; held tick36 through stop+30.
  Same P0 feedback+PD projection. Candidate options are public inputs.
- Each query freshly recreates the whole scene and replays actual baseline
  PD targets for first36ticks. Require bit-identical predecision root/q/dq,
  all27-body poses/velocities; no cross-env clone or same-reset assumption.
- E: ALL32 subsequent object SE3/twist packets, position relative to decision
  root, quaternion canonicalized; no final success/height105 summary.
- I: ALL25 hand bodies relative to object (pose/twist) on32ticks, plus contact
  statistics for ALL25 hand bodies and table at both physics steps/tick:
  pair-specific count, normal force, weighted object-local contact position
  and second moments. All actual contact records retained without top-K
  truncation. This is a rich deterministic descriptor, not an information-
  preserving encoding of every contact manifold or certified friction cone.
  Counts distinguish potential-contact records from positive-normal-force
  records; force and geometry moments are averaged over two physics steps.
  Position/twist are expressed relative to the object; effect position and
  rotation are relative to the decision pose. I can implicitly contain some
  effect information: these are input ablations, not independent causes.
- Four identically shaped MLP task-Qs (64/64,ReLU), same shared initialization,
  TRAIN-only normalization, same minibatch order and AdamW schedule1500updates
  each, BCE on complete105-tick TRAIN success. Arms state+request, +E, +I,
  +E+I mask unused privileged slots to zero. GPU fitting/inference.
- EVAL candidate terminal outcomes are never supplied to Q selection.
  Query features use only ticks36..67. Argmax Q with lowest option-index tie;
  selected per-env actions are actually re-executed in a new whole scene.
  Compare deployed short futures to selected queries at fixed position1e-6m,
  quaternion1e-6,velocity1e-5; violations invalidate conditional oracle claim.
  Joint positions1e-6 and velocities1e-5; short descriptor max error1e-4
  in its recorded scaled units. Independent NumPy final-Q probabilities
  must match within5e-5 and reproduce ALL96 argmax choices per arm.
- Primary outcome: actual full25002vertex clearance20mm and object-root rise
  30mm on ALL105 late/plateau/drop ticks. Report every episode/motion, P0 and
  all four selectors. No retrospective best-candidate success as control gain.
- PROMISING requires joint selector gain >=5 percentage points over state
  and effect-only, joint >=P0, verified prefix/query/deployed audits. If not,
  UNPROMISING for this fixed finite-option representation/optimization recipe;
  engineering failures remain UNCLEAR rather than a scientific negative.

Budget one admitted idle GPU, <=3600s/12GiB including fresh queries, all4fits,
actual deployed rollouts and audit. No official actor checkpoint,6000new Q
updates, zero Cm/actor/PPO updates; oracle query cost reported explicitly.
Exploratory single scene pair / one model seed; no validation/generalization,
formal upper bound, oracle learnability or journal-ready claim.
