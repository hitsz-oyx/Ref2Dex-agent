# P-20261001-current-policy-v-fit

Classification: Decision Probe. User authorized continuation of the factual V
diagnosis, then asked why GPU was unused; hardware reassessed for GPU training.
Single-session implementation.

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

## Completed r1

Run status COMPLETED; fixed gate **UNPROMISING**. This label applies to the
predeclared two-checkpoint20% adaptation gate, not to all V adaptation or Cm.

Implementation commit `cc37c03`; all15 focused tests passed. GPU4 UUID
`GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307` admitted with2MiB/no compute owners.
Whole process30.604seconds (internal28.797seconds), peak torch allocated
memory685054464bytes, combined output/support10503124bytes. GPU released;
all original source/checkpoint/data hashes unchanged. Cached history versus
direct features maximum absolute error0; original saved-V RMSE reproduced.
Both V optimizer step sets advance7680 ->8680; only new V/Adam artifacts saved.

Raw reward-unit frozen GAE RMSE:

| Checkpoint | Split | Episodes | Before | After1000 | Reduction |
| --- | --- | --- | --- | --- | --- |
| s286 | fit | 78 | 30.615 | 7.413 | 75.8% |
| s286 | holdout | 18 | 13.756 | 11.269 | 18.1% |
| s287 | fit | 78 | 19.786 | 5.880 | 70.3% |
| s287 | holdout | 18 | 30.839 | 16.719 | 45.8% |

Both networks can fit these factual targets substantially better using the
same representation. Some gains transfer to independent episodes, but s286
misses20%, so the fixed joint gate fails. This does not establish why the
original online training had larger error: the present target/distribution is
frozen and narrow, and V-only optimization differs from joint V/Q updates.
Neither convergence nor original-training insufficiency is established.

PPO critic holdout GAE RMSE remains10.601/16.209, versus adapted V11.269/16.719.
Holdout realized MC RMSE is32.355 ->33.525 (s286) and68.619 ->65.507 (s287).
Reducing GAE error therefore does not establish stable complete-return
improvement. Primary-success coverage is particularly sparse: s286 has6 fit,
0 holdout successes; s287 has0 fit,1 holdout success. On the lone s287 holdout
held_1s episode,41 rows, GAE RMSE37.147 ->106.119 and MC149.080 ->255.954.
This rare diagnostic phase cannot carry a general scientific claim, but it
prevents treating aggregate improvement as verified held-grasp value accuracy.
Do not reassign split or select an intermediate checkpoint after seeing this.

Decision: close HD01 at1/1, retain HF08 PAUSED, do not add updates/seeds to chase
the gate. Optimization can reduce observed fit error; the next decision must
address useful held-grasp target/coverage and transferable decision information
before another costly PPO/utility run. No online policy update is launched by
this diagnostic. North-star Cm utility remains OPEN.

Full metrics, split/input hashes, command, runtime and artifact audit are in
[the results record](P-20261001-current-policy-v-fit-results.json). Local new
artifacts are under `src/task/CmResidual/research/physical_value/output/`
`P-20261001-current-policy-v-fit-r1` and its `-support` directory.
