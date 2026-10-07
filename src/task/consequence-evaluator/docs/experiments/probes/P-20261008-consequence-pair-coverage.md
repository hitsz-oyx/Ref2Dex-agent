---
schema: ref2dex.probe.v2
probe_id: P-20261008-consequence-pair-coverage
experiment_id: P-20261008-consequence-pair-coverage
date: 2026-10-08
task: consequence-evaluator
branch: consequence-evaluator
git_commit: e62dd08
claim_id: C3
hypothesis_family: HF-consequence-oracle-headroom
probe_index_in_family: 2
seed_pool: probe
seeds: [296, 297, 298]
decision_changed_if_positive: use the unchanged local-state preference contract for a fresh matched evaluator fit
decision_changed_if_negative: keep evaluator fitting paused and redesign collection around explicitly repeated current states
status: UNCLEAR
run_id: pair-coverage-20261008-r1
---

# Can repeated waves recover strict current-state preference coverage?

Result: Three-wave train/val/test collection is `READY` under the unchanged
contract, with only 2/1/5 train/val/test pairs; physical geometry remains
valid, but the sample is too small for a decision-useful evaluator fit.
Decision: Stop before fitting and design explicit twin current-state branches;
do not relax hand RMS or add generic waves.

## Purpose and decision

The first six-route collection produced 0/0/2 train/val/test local preference
pairs under the predeclared state contract, despite a valid geometry audit.
This Probe tests the cheapest remaining explanation: two waves did not repeat
the same motion/phase/current state often enough. It keeps the contract fixed:
24-step decision-known residual plans, object translation <=3 cm, object z <=1
cm, rotation <=15 degrees and measured 11-point hand RMS <=2 cm.

The result changes only whether a fresh evaluator fit is eligible. It cannot
change the global Cm claim, promote a weak specialist, or justify relaxing a
state threshold after seeing the labels.

## Protocol and resource boundary

Use the frozen route and corrected native motions from the first collection.
Collect three waves with 24 environments for each of train/val/test, using
seeds 296/297/298, amplitude 0.08, the same 1200-step cap and one GPU at a
time. This produces at most 72 episodes per split and preserves seed-group
isolation. Label with the existing abstaining rule on CPU.

The required gate is nonempty train and val pairs under the unchanged rule,
plus valid native geometry diagnostics. If either split is empty, stop before
any model fit and record the raw artifact for future targeted annotation. If
both are nonempty, freeze the source/label hashes and write a separate fit
Decision Note; no fitting is authorized by this card alone.

## Outputs

Collectors write to `outputs/consequence-evaluator/continuous-20261008-pair-{train,val,test}-r1/`.
Labels write to `outputs/consequence-evaluator/labeled-continuous-20261008-pair-all-r1/`.

## Result and stopping Decision Note (2026-10-08)

All three collectors completed 72 episodes each. Their manifests are
`COMPLETED`, use commit `e62dd08dcc601282e84c0c49bc860e29125f6bc3`, and stay
within the 900 s / 2 GiB collector budget. The CPU labeler accepted all 216
episodes and reports `READY`; train/val/test pairs are 2/1/5 under the
unchanged contract. The geometry audit has 149,841 valid frames, 49,388
native proxy frames, 45,984 proxy-and-near frames, and only one proxy-far
frame. Repeated waves therefore repair the empty-split gate without exposing a
native geometry failure.

The fixed-contract diagnostic is
`outputs/consequence-evaluator/pair-coverage-diagnostic-20261008-pair-r1/coverage.json`.
It reproduces 2/1/5 pairs at 2 cm hand RMS; diagnostic limits of 3/5/8 cm
produce 2/1/5, 3/1/5 and 4/1/5. The extra waves increase coverage, but the
substantive sample remains one validation pair and two training pairs.

Question: is this enough evidence to fit the matched evaluator?

Root decision: stop before fitting. `train_matched.py` would resample the same
two training comparisons for every update and select on one validation pair;
that is an engineering execution check, not a decision-useful oracle-headroom
Probe. The raw, label and diagnostic artifacts are retained. Do not relax the
2 cm state contract or claim E0/Eoracle ranking information from this run.

The next useful experiment is a new targeted collector that creates explicitly
paired current states (same reset/history and two residual branches), followed
by a fresh fit Decision Note. Generic waves and automatic specialist epochs
are closed for this route until that protocol is specified.
