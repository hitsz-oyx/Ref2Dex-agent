# Decision Memo: Cm model-based critic augmentation Probe

Date: 2026-10-03

## Question

After the selector, MVE action ranking, residual, task representation, and task-value
auxiliary Probes failed to produce policy utility, is there one remaining higher-level
use of Cm that is both distinct and cheap to test?

## Evidence

The frozen Cm ensemble still predicts one-step physical consequences, but every tested
interface that lets those predictions choose or shape an action has failed its policy
or held utility gate. Direct-Q remains the strongest task-value baseline. The previous
MVE Probe used Cm at action-selection time; it did not test the MBPO/value-equivalence
role of using conservative synthetic transitions to train a critic while leaving the
actor observation and action unchanged.

## Decision

Run one offline Probe on training-time model-based critic augmentation. From fit-split
real states, generate one-step Cm transitions only when ensemble disagreement is below
a fixed predeclared gate. Train a fresh direct-Q-shaped head on the real return target
plus the frozen-model one-step target, and compare its held RMSE, MAE, row Spearman, and
episode Spearman against the frozen direct-Q. Cm remains a physical consequence model;
it does not enter the actor input, action, reward, or success label.

The Probe is promising only if the conservative synthetic target improves held RMSE or
MAE without reducing episode Spearman by more than `0.01`, and the model target is finite
on at least 20% of fit rows. Otherwise close the model-based critic route and do not run
policy training, coefficient scans, or extra data collection.

Because the offline gate passed, one matched policy Probe is authorized by this memo:
source epoch 260, one training seed `292`, 64 environments, epoch 300, and evaluation
seeds `293` and `294` with 96 environments. The control uses the original physical
checkpoint and the treatment uses the same bundle with only the direct-Q state replaced
by the saved augmented Q. Both arms use the same `direct_q` teacher and reward contract.
The treatment must be nonnegative on stable success for each evaluation seed and reach
an aggregate `+8` percentage-point stable-success margin to remain open; otherwise this
route closes without seed, epoch, or coefficient scans.

## Cost and safety

This is an offline GPU Probe using existing frozen checkpoints and collections. It does
not change the mission claim, external project, or running processes. No user approval
or external authorization is needed.

## Outcome

The offline Probe passed its held criterion, so the memo-authorized matched policy Probe
was run. On evaluation seed `293`, direct-Q achieved `10/96` stable successes and the
model-based-Q arm achieved `4/96`; the per-seed nonnegative policy gate failed. The
second treatment seed was stopped before completion. The policy route is therefore
closed even though the critic-only target improved held metrics. See the final route
review in `D-20261003-cm-route-review-model-based.md`.
