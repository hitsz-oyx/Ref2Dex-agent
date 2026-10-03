# P-20261003-cm-model-based-critic

Family: Cm model-based decision interface
Type: Offline critic augmentation plus matched policy Probe
Status: COMPLETED EARLY — `UNPROMISING` at policy gate

## Question

Can Cm enter policy learning through conservative short synthetic transitions used to
train a direct-Q critic, while leaving the actor observation, action, reward, and success
definition unchanged?

## Offline Probe

The frozen physical ensemble predicted one-step consequences from the fit split. The
synthetic target was the ensemble mean minus disagreement, using Cm-predicted object
translation/velocity/contact/events, observed hand q/dq and object orientation, and a
frozen direct-Q continuation. Rows with normalized ensemble disagreement above `0.5`
were excluded. A Q copy was fine-tuned for 600 updates on a fixed `0.25` blend of the
real complete return and the synthetic target; no coefficient or threshold scan was
run.

The Probe used 120,000 fit rows from 1,536 episodes and 209,788 held rows from 384
episodes. The gate retained `99.62%` of fit rows. Held frozen direct-Q scored RMSE
`26.833`, MAE `9.108`, row Spearman `0.589`, and episode Spearman `0.743`. The
model-based Q scored `25.712`, `8.377`, `0.606`, and `0.774`, respectively. This is a
`PROMISING` critic-only result and is recorded in the offline result JSON.

## Matched policy Probe

Because the offline gate passed, both arms started from source epoch 260 with training
seed `292`, 64 environments, and 300 epochs. The control used the original physical
bundle; treatment used the same bundle with only `direct_q` replaced by the saved
model-based Q. Both used the same `direct_q` teacher, reward contract, and evaluation
configuration. Seed `293` evaluated 96 complete episodes: control stable success was
`10/96`, treatment `4/96`, a `-6` episode difference. This violated the predeclared
per-seed nonnegative gate, so evaluation seed `294` was stopped before completion.

The offline critic improvement therefore did not translate into the policy gate. Close
this model-based-Q policy recipe. Do not scan model weight, uncertainty threshold,
training seed, epoch, or extra data; retain the critic result as a value-equivalence
clue only. Cm still retains its physical consequence prediction role, and the overall
Cm policy-utility claim remains open.

Evidence: [offline results](P-20261003-cm-model-based-critic-results.json), [matched
policy results](P-20261003-cm-model-based-critic-policy-results.json), and [decision
memo](../../decisions/D-20261003-cm-model-based-critic-augmentation.md).
