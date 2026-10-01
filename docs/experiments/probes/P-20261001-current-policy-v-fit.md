# P-20261001-current-policy-v-fit

Classification: Decision Probe. User authorized continuation of the factual V
diagnosis and directly requested GPU use. Single-session implementation.

Question: can the saved V, with unchanged input representation, fit frozen
current-policy GAE targets better and transfer that improvement to independent
episodes? Positive results favor current-distribution adaptation as a useful
next mechanism check. Fit-only gains favor a generalization/data-coverage check;
no fit gains favor an optimization/representation check. None proves policy utility.

Cheapest test: reuse the completed 107584-row, 192-episode s286/s287 panel.
Separate fits initialized from their corresponding e420 V; no environment
collection, actor, Q, dynamics, reward or PPO update. HF08 remains PAUSED at 1/1.

Fixed contract before execution:

- Experiment run r1; split seed20261001. SHA256 ranks motion/env identity,
  pairing assignment across checkpoints without looking at reward/success.
  Each motion has26 fit and6 holdout episodes:78/18 per checkpoint. Histories
  never cross episode, checkpoint or discontinuous step boundaries.
- Frozen PPO-critic GAE(gamma=.99, lambda=.95, horizon32), terminal bootstrap0,
  saved normalization/context/history/network. Return scale63.89336 divides
  training loss only. Episode holdout never supplies gradients.
- Restore V's10 Adam states from the saved20-parameter joint V/Q optimizer;
  saved lr=.001, clip norm5 over V only (different from original joint V/Q
  clipping). Row-uniform fit sampling with replacement, batch128, exactly1000
  updates per checkpoint, sampling seed20261001+checkpoint seed.
- Record stages250/500/750/1000, decide using final1000 only; no holdout
  checkpoint selection. Fixed gate: both checkpoints reduce both fit and
  holdout raw-unit GAE RMSE by at least20% => PROMISING. Otherwise UNPROMISING
  for this bounded adaptation check; invalid inputs/run => UNCLEAR.
- Report MC, PPO baseline, phase and primary-success slices separately.
  One-second held phase differs from45-step primary success; rare slices
  cannot establish calibration. GAE fit gains cannot prove MC accuracy or
  repair counterfactual action ranking.
- One idle GPU admitted after UUID/process check;2 CPU threads;1200seconds
  hard process budget;1GiB new artifacts. Internal stop1170seconds leaves
  exit margin. Stop on input drift, nonfinite, memory/disk/time violation.
  Preserve original checkpoints; save only new V artifacts under task output.

Engineering admission: verify cached GPU features equal direct history
features and reproduce each prior saved-V GAE RMSE within.001 before updating.
Focused tests cover paired episode split, boundary features, and optimizer
moments/ownership. Code commit and input hashes recorded by runtime manifest.

Outcome: PENDING.
