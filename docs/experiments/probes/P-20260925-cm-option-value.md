# P-20260925-cm-option-value

- Classification: Decision Probe.
- Branch: `agent/cm-option-value`.
- Candidate Cm: a task-aligned predictor of expert-option held-lift from
  current physical state, actor observation and the expert's proposed
  initial 18-D action.

## Question and decision

The five-expert seed219 Probe showed an optimistic 14/64 option headroom
over fixed route A, with 3/64 recovery in a repeat of the same route.
Does the *candidate action* carry useful information about which expert
will hold and lift from a particular initial state, beyond the state,
observation and expert identity?

Run all five frozen self-trained experts on the same 12 converted
motions, 64 first full episodes, seeds223 (fit) and224 (heldout). Export
the first pre-action observation, joint state, object state, proposed
action, expert identity and whole-episode held-lift label. Align all
five runs by environment, motion ID and start frame. No official actor
checkpoint is allowed. The same expert portfolio and outcome rule are
used in both seeds.

Fit three fixed-capacity regularized logistic models on seed223 only:

1. **Action-aware Cm candidate:** standardized joint/object state,
   32 PCA components of the 1442-D observation, expert one-hot and
   the proposed initial action.
2. **Action-blind control:** identical inputs and fitting except action.
3. **Action-shuffled control:** same dimensionality as aware, but permute
   action vectors across environments *within each expert* using RNG seed
   20260925223 for training; test with independent RNG seed 20260925224. This
   preserves expert identity and the action distribution while breaking
   the state/action pairing.

Use `LogisticRegression(C=1, max_iter=1000)` after training-set-only
standardization. Score Brier error over all 320 heldout expert-state
rows. For each of the 64 states, choose the expert with highest
predicted success probability; count its observed held-lift and compare
with the action-blind choice and the frozen object route on seed224.
The model is `PROMISING` only if action-aware Brier is at least 10%
lower than blind and 5% lower than shuffled **and** its observed top-one
route exceeds each of blind and frozen object routing by >=5/64.
Otherwise do not integrate this initial-action Cm into the online
policy. This is a Probe; the selected route is evaluated on one seed
and outcomes from separate simulator executions may vary.

One idle GPU, <=40 minutes, <500 MB output. Stop on checkpoint or
motion drift, missing initial features, incomplete episodes,
nonfinite values or invalid alignment. Labels and action features from
seed224 must not be used for model selection or fitting.

## Results

Pending.
