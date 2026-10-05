---
schema: ref2dex.probe.v2
probe_id: P-20261005-rolling-gt-y
experiment_id: P-20261005-rolling-gt-y
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: 73334e9
claim_id: C3
hypothesis_family: HF-rolling-gt-y
probe_index_in_family: 1
seed_pool: probe
seeds: [267]
decision_changed_if_positive: prioritize a newly paired receding-horizon oracle experiment rather than long-Y fitting
decision_changed_if_negative: distinguish absent sampled rolling signal from right-censored horizons and target resolution
status: UNCLEAR
run_id: rolling-gt-y-s267
---

# Ref13_1 Rolling GT-Y audit

Result: pending; protocol frozen before rolling outcome calculation.
Decision: Audit saved trajectories before any rolling intervention acquisition.

## Decision Note / purpose

Decision experiment for MissionC3: ref13 only tested one-shot short-Y utility.
Ref13_1 asks whether the SAME short-Y semantics becomes informative when rolled
along actual trajectories. Distinguish delayed local signal from permanently
coarse labels; use the existing32×7 complete factual branches, no new model or
simulator. A signal would motivate new same-current-state rolling interventions;
absence would separate observed no-signal from insufficient saved horizon.
No change to root claim, old data, U coefficients, candidates, Z or prior gates.

## Frozen inputs and computation

Input `outputs/cm-interaction-oracle/oracle-y-utility-sync-s263-s264/` and its
hashed two batch reports/seven panels each. ALL32 anchors/7 candidates retained;
verify source/assembly/current-state/provenance, raw height/pair90/valid90,
originalY8 andZ90 and fixedU before calculating rolling statistics. Original
history physical72 only32steps; all Y heads depend only on current height/pair,
future height/pair and reference rest, which are recorded for90steps. No missing
geometry/force fields are invented. Reconstruct six continuation heads and
late-height-failure/height-held-fraction directly from these sufficient fields.

At offset tau, current state is original `before` iftau0 else post-step tau.
Risk is currentheight>=3cm above common reference rest AND currentforcepair.
The local32step contact-loss run starts0, continues through localstep8, exactly
as the original `continuation_outcomes`; no whole-episode loss-history change.
Y16 uses local9..16; Y32 uses local9..32. U=Y7+.25Y3−Y6, unchanged baseline-first
then arm-order tie break. Reproduce originaltau0 Y/U bit-exact.

Primary schedule tau=0,8,16,24,32,40,48,56; eachFULL32future steps. No padding,
truncation or retaining unsupportedtau64/72/80. Dense integer tau0..58 is a
DECLARED boundary-resolution diagnostic to identify signal between scheduled
queries, especially step89 failures beyond the lastprimary endpoint88. It does
not replace the8-step primary cadence. Beyond58 is right-censored, not no-signal.
Original finalZ labels are conditioning only for descriptive separability.

For every same-original-anchor pair with different finalZ, orient good(Z1) vs
bad(Z0). Report |Ugood−Ubad|>.02 (separation) AND Ugood−Ubad>.02 (correct sign),
first sampled time/lead, persistent sign across observed subsequent pre-event
queries, current-risk masks, and inverse/sign-changing cases. Raw scores forALL
states/times retained; interpretable pre-event pair comparisons require BOTH
current states at-risk. Declare event types: bad qualified then postqual drop;
unqualified bad with physical/proxy threshold event; unqualified without any
observed threshold event (no drop lead, censored/no invented failure). Record
bad trajectory's FIRST raw height<2cm or6consecutive forcepair-loss relative to
originaltime0 separately from actualZ postqualification failure. Do not label
allZ0 as a witnessed drop, or ignore earlier failures to gain a longer lead.
Report pair/anchor/group counts; pairs are not independent samples.

Initially tied discordant pairs and the16 originalall-utility-tied states are
the delayed-signal diagnostics. Primary decision signal: at least one initially
tied Z-discordant pair becomes correctly separated at a scheduledtau>0 with
>=8step lead before its observed firstbad event, BOTH states at-risk, and no
subsequent observed pre-event inverse beyond.02. This is only PROMISING for
recorded-path rolling observability, not candidate choice at a common newH.
If none: UNPROMISING only for fully observed primary query/event pairs; UNCLEAR
for rolling-control utility/near-boundary opportunities with missing queries.
No additional thresholds/seeds/labels or model fitting after result inspection.

## Rolling intervention support / ceilings

At tau>0 original candidate worlds already have DIFFERENT H. Their rolling
Y describes continuing factual paths with no new K8 intervention; it is NOT
GT-Y for seven actions newly applied at one shared currentstate. Therefore
no saved-data stitch/switch can report an executed rolling policy Z gain.
Report original GT-Z25/32 finitebranch-access ceiling and which of its two
baseline-failed rescue opportunities have an at-risk pre-event scheduled
rolling top1 pick of aZ1 branch; retain all cadence/tie diagnostics. Call this
NON-EXECUTABLE branch lookup support, not an attainable rolling upper bound.
A true rolling controller's upper bound is UNKNOWN from these recordings.

## Resources, review and limits

CPU-only file/label/statistical audit, zero neural inference/training or new
simulation; GPU adds no benefit here. Main script<=120s,<=30MiB newoutputs,
unique `outputs/cm-interaction-oracle/rolling-gt-y-s267/`, no prioroutput changes.
Bootstrap267,2000 resamples by32 originalanchors for descriptive pair coverage,
not formalValidation/sharedsolver noise inference. Stop on any hash/finite/
validwindow/label/current-state drift. Commit code before main computation;
semantic contract tests then independent CPU rawlabel/lead/causal-scope review.
All provenance and standalone figures live with outputs; docs only summaries.
Sharedsolver fourgroups,29s3/3s7, forcepair proxy, boundedZ terminal-edge delay,
post-treatment factual paths and unsupported horizons limit conclusions.
Actual rolling selection, new interventions, modellearning andpolicyutility are
future evidence/next Decision work, not required to answer this cheap audit.
