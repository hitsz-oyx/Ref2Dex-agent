# P-20260924-s3-expert-outcome-cm

- Classification: Decision Probe.
- Branch: `agent/grab-multitrajectory-baseline`.
- Question: Can an action-conditioned outcome Cm rank self-trained s3
  full-episode grasp experts from the initial physical state, instead of
  scoring isolated wrist/finger actions?

## Rationale and decision

The V1.45 self-trained s3 expert portfolio has real full-episode grasp
diversity: its fixed start-frame route achieved 499/640 versus 431/640 for a
single expert. Earlier local wrist/finger options failed to yield sustained
lift, so an entire grasp policy is the candidate action sequence here. A Cm
that forecasts contact-supported lift conditional on the candidate policy's
initial action could score the choice before execution. It must beat a
state-plus-expert-ID head and an action-shuffled placebo; otherwise action
information has not earned a policy decision role.

Use the three existing self-trained, no-official-actor s3 experts: e180,
back240 and back260. Collect first pre-action observation, hand/object
state, actual first action, and full first-episode contact/lift outcome from
each expert on the same s3 motion and seeds301–305, 64 environments/arm,
early termination disabled. Seeds301–303 fit the outcome models; seeds304–305
are untouched Probe tests. Before fitting, check the candidate arms' initial
motion ID, start frame, hand and object states align per seed/environment;
abort the router analysis if they do not.

Fit the same small model with (1) physical state + expert ID + candidate
first action, (2) physical state + expert ID, and (3) shuffled candidate
action. Forecast both full-episode held-lift and contact-supported maximum
lift. The action-aware model must exceed both controls by at least 5pp in
heldout episode-level success AUC and its shadow selected-expert success
must exceed the best fixed expert by at least 5pp over the two heldout seeds.
These are exploratory gates, not formal policy utility. Passing would
justify one real online Cm-on/off/placebo route evaluation with the same
expert portfolio; failing would stop this high-level Cm feature design.

Collection budget: at most 2 idle GPUs simultaneously, <=60 minutes total,
<1 GB outputs. Stop on checkpoint/input drift, GPU conflict, incomplete
first episodes or feature alignment failure. No official actor participates.

## Results

Pending.
