# P-20261020-inspire-paired-bc-dev

## Decision

Determine whether Scheduled686 has a stable offline action advantage over the
best Current421 checkpoint before spending effort on an online WAM-conditioned
bridge.

## Protocol

- `task`: `wam-rl-policy-v3`
- `experiment_id`: `P-20261020-inspire-paired-bc-dev`
- `run_id`: `inspire-paired-bc-dev-20261020-r1`
- `branch`: `main`
- `git_commit`: recorded in the audit output
- split: DEV from the audited same-execution teacher corpus
- paired admission: active corpus rows with regenerated tau `valid=true`
- checkpoints: Current421 `best.pt` and Scheduled686 `best.pt`
- output: `outputs/wam-rl-policy-v2/inspire-paired-bc-dev-20261020-r1/paired.json`

## Result (2026-10-20)

There were 5,693 paired rows after applying the active mask; the cache's 5,799
valid DEV rows also include rows that are outside the active BC prefix. Current421
had command MAE 0.036321, root MAE 0.016717 and finger MAE 0.055925.
Scheduled686 had command MAE 0.036683, root MAE 0.016646 and finger MAE
0.056720. The scheduled minus current command delta was +0.000362; root improved
by 0.000071 while finger worsened by 0.000795. This is not evidence of a stable
tau benefit, so the next bridge step must remain an engineering closed-loop
check rather than a PPO commitment.
