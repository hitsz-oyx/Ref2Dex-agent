# Frozen oracle-selector training diagnostic

User asked whether fits converged, terminal success was sparse, and a single
decision could test the proposed privileged-information upper bound. This is
a saved-artifact diagnostic, not a new optimization/physics Probe or formal
Validation. GPU6 was idle when checked; frozen batched inference used it.
No original experiment/model files were changed. Source SHA256s and full
metrics are in `20261003-oracle-selector-training-audit.json`.

## What the recorded optimization establishes

All four supervised binary classifiers trained1500steps on the same768
candidate-context samples, with only96distinct contexts. Full-dataset metrics
below are recomputed from final frozen weights. Heldout refers to the original
r1global-option collection, NOT r2fixed-background deployed grasp utility.

| Arm | Train BCE | Train accuracy | Heldout accuracy | Heldout BCE |
| --- | ---: | ---: | ---: | ---: |
| State+action |0.200061|91.67%|88.93%|0.270874|
| Effect |0.000134411|100%|95.57%|0.264488|
| Interaction |0.0000187086|100%|96.22%|0.281116|
| Joint |0.00000946773|100%|96.09%|0.304050|

State minibatch BCE first100mean0.519777 vs last100mean0.208282;
steps1301--1400mean0.205800, suggesting a recent plateau, not a proof of
optimality. All three oracle classifiers fit every training label correctly.
Their tiny training BCE rules out gross failure to fit these data, not
overfitting, calibration error, insufficient data or distribution shift.
Joint heldout BCE exceeds state despite better classification accuracy,
consistent with costly overconfident mistakes. There was no per-checkpoint
heldout learning curve, multiple optimization seeds or stopping validation;
therefore no certified generalization-convergence claim.

## Label and action support

Train successes279/768=36.328%; heldout302/768=39.323%. Overall class balance
is not extremely sparse. However, motion0has0/256positive samples in BOTH
scenes. Train motion1has171/256, motion2has108/256; heldout180/256and122/256.
Each scene's finite-bank best is64/96: all32motion0contexts are unreachable
within that bank, while all64other contexts have a successful candidate.
Train59contexts and heldout58contexts have both positive and negative options.
The classifier does have within-context ranking supervision in those cases,
but no successful alternative or terminal-label progress gradient for motion0.

Later qualified r2panel independently reaches the same relevant constraint:
all4motion0subjects fail every option, and state already attains8/12bank
capacity. These observations do not prove absence of useful contact knowledge
or imply new actor training could not learn richer actions.

## Scope correction

The model reads one fixed decision-time context, a held12Dadjustment and,
for oracle arms,32ticks of true future trajectories/contact statistics. Its
label depends on the complete late105tickwindow. It does not predict merely
one-step success, but it makes only one decision and never updates its choice
through grasp approach, closure, lift or stabilization. Frozen P0 continues;
no actor/PPO is trained from privileged observations. This is an inexpensive
candidate-selector screen, not the user's proposed fully privileged policy
upper-bound experiment. Binary terminal labels also discard graded progress
among failures. A future policy test needs useful executable control support,
repeated observation/control and process supervision, while retaining the
original terminal success metric for comparison. Do not add fit steps to
rescue this closed zero-headroom action bank.

Reproduce (no updates):

```bash
CUDA_VISIBLE_DEVICES=GPU-9442e3c0-2f76-67d9-c326-d86b42440622 \
  /home2/wyy/miniconda3/envs/graspenv/bin/python \
  scripts/audit_oracle_selector_training.py \
  --fit src/task/CmResidual/research/contact_response/output/P-20261003-effect-interaction-oracle-r1/fit \
  --output /tmp/oracle-selector-training-audit-repeat.json
```
