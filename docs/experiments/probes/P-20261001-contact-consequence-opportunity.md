# P-20261001-contact-consequence-opportunity

Decision Probe, HF09 contact-consequence-control, slot1/3. User's active Goal:
candidate opportunity → consequence learning/ranking → directly executed
replanning; policy training and final stable grasp evaluation follow positive
mechanism evidence. HF08/HD02 remain closed and their inputs unchanged.

Question: do real six-expert contact proposals offer useful short-window
improvement larger than repeated-base noise? Positive: learn the consequence
ranker and controls. Negative: review candidate opportunity before any ranker
or PPO, without declaring the core Cm hypothesis refuted. This is the cheapest
test because a predictor cannot select an advantage absent in its action set.

Frozen design before execution:

- Airplane, three canonical motions, six self-trained checkpoints and hashes
  from `hf02_temporal_canonical_route.json`; base is `source_e260`, index4.
  No official actor, no new PPO/V update. Candidate pool reused as actions,
  not treated as six equally competent airplane policies. Fixed seed330.
-96 seeded reference-frame initializations; balanced motion assignment.
  Select at most32 first-episode contact decision states with10 history steps,
  quotas11/11/10 and sufficient remaining first-episode horizon. At least24
  complete states required; fewer means engineering/support failure, not an
  opportunity conclusion. No manual start-frame selection or seed search.
- Capture complete cold native state/properties/RNG once. Each fresh branch
  restores it with the same setter order and replays the saved pretrigger
  actions/common per-tick RNG. Save actual trigger state, observation and
  history. This is tolerance-checked approximate pairing, not hot solver cloning.
  Object position error<=1mm, rotation<=.01rad, wrist<=1mm, finger<=.01rad,
  DOF velocity<=.05, object linear velocity<=.05m/s and angular velocity<=.1rad/s,
  also over the10step physical history; actual trigger contact must remain true.
  Any invalid state prevents a positive gate. No filtering
  of difficult rows to improve the gate.
- At trigger, cache each expert's normalized clipped current action. Execute
  the selected candidate unchanged for2steps; then frozen base observes its
  own resulting state and controls8steps. Six arms, two fresh repeats per arm;
  the first base run also records prefix/actions/RNG. Reuse identical candidate
  proposals; confirm actual applied actions and complete10step first-episode
  windows. Every arm computes all six expert forwards in the same order,
  preserving point-sampling RNG placement; only action execution changes.
  Actor and RMS parameters/buffers remain frozen.
- Physical outcome: mean positive lift relative to trigger, supported by the
  native hand-force AND object-force contact proxy, in mm; contact fraction;
  contact loss; drop if object is already>=3cm above motion frame0 rest height
  at trigger and then falls below2cm or loses contact for6steps. Report drop
  eligibility; no eligibility cannot establish drop-risk learning. The contact
  proxy does not identify collision pairs and is labelled as such.
- Select each state's best candidate with repeat0 supported lift; confirm its
  real uplift against repeat1 base using repeat1 outcomes. Do not evaluate a
  candidate maximum on the same noisy repeat that selected it. Mean confirmed
  uplift must be>=max(2mm,2×mean absolute base-repeat noise); at least20% states
  must improve by>max(2mm,2×their base-repeat noise). Mean contact change>=−.05
  and all-state drop increase<=.05. Full input/state/execution contracts pass.
  This is a local opportunity screen, not formal power or learned policy value.
- Single admitted idle GPU4,2CPU threads, whole native run<=60minutes,
  including any engineering retries; new output<=8GiB, each native<=300seconds.
  Original checkpoint/config/motion hashes before/after, unique owned output
  `src/task/CmResidual/research/contact_consequence/output/`, no shared output
  symlink writes or original checkpoint overwrite. Stop on budget/input drift.

Implementation and isolated tests cover physical labels, independent-repeat
selection, noise threshold, pose/joint/velocity tolerances, and the actual
collector loop's cold-state/prefix/candidate2→own-base8 action path. Real native
execution is still required; CPU tests do not establish physical opportunity.

Engineering r1 failed before the first native action/episode: the simulator
uses Python3.8, which lacks `str.removeprefix`. Replaced with the existing
compatible startswith/slice pattern and expanded the native-player test to
exercise real six-expert loading, compiled wrapper keys and frozen parameters.
No scientific result or checkpoint update existed. r1 logs/cache remain;
r2 charges its time/storage to the same fixed60minute/8GiB budget, preserving
seed/panel/gates and all input hashes.

Engineering r2 also failed before the first applied action: the custom player
did not call native `get_batch_size`, leaving `has_batch_dimension` false and
flattening96 observations into one actor input. Full cold snapshot restore had
passed. Added the required batch initialization and a native-loop regression
that refuses action inference before it. r3 charges both failed attempts under
the same budget; no candidate outcome or scientific negative existed in r1/r2.

## Terminal r3 and next decision

All12 native phases COMPLETED;32 first-contact states, complete10step windows,
six arms/two repeats, actual cached candidate2→own-base8 execution verified.
The first-repeat-selected candidate has second-repeat mean uplift2.224mm,
median0.541mm, versus base mean absolute repeat noise0.210mm. Resolved-opportunity
fraction34.375%, contact change+0.04375; these numerical screens pass.
However only11/32 states pass every arm/repeat trigger/history tolerance.
Most discrepancies are thresholded contact flags; some velocity/rotation
histories also differ. No state is>=3cm above rest at trigger, so there is no
drop-risk support. Do not discard the failing states and call the rest positive.

The native automatic label was UNPROMISING because its positive gate failed.
The independent completion audit correctly classifies the scientific result
as **UNCLEAR / PAIRING_CONTRACT_FAILED**: candidate opportunity has a signal,
but approximate branches cannot confirm per-state counterfactual ranking.
The raw result remains unchanged; label helper corrected for future runs so
contract failure is not confused with a valid negative opportunity result.

All checkpoint/config/motion/source hashes verified unchanged at completion,
all owned native processes terminal and GPU4 released. Cumulative r1/r2/r3
native wall480.24seconds,215.3MB retained. CPU was used only for isolated tests
and file/trace auditing. [Audited result index](P-20261001-contact-consequence-opportunity-results.json).

Next: use actual-state sequential randomized interventions with private
assignment p=1/6, complete2+8 labels and episode/frame-group splits. Multiple
contact points cover later states; do not reuse old10-step options as labels.
This follows the mechanism memo's declared randomization fallback, leaves
this paired gate unchanged, and does not claim Cm selection already works.
