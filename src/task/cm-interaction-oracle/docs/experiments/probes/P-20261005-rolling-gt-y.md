---
schema: ref2dex.probe.v2
probe_id: P-20261005-rolling-gt-y
experiment_id: P-20261005-rolling-gt-y
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: de6ac9716ec07deca7441a893bd128e6e44fda7f
claim_id: C3
hypothesis_family: HF-rolling-gt-y
probe_index_in_family: 1
seed_pool: probe
seeds: [267]
decision_changed_if_positive: prioritize a newly paired receding-horizon oracle experiment rather than long-Y fitting
decision_changed_if_negative: distinguish absent sampled rolling signal from right-censored horizons and target resolution
status: PROMISING
run_id: rolling-gt-y-s267
---

# Ref13_1 Rolling GT-Y audit

Result: PROMISING recorded-path rolling observability:46/50 initially tied discordantpairs separate correctly atscheduledqueries,8/32anchors; rolling-control utilityUNCLEAR.
Decision: Retain short-Y and prioritize a new same-current-state rolling oracle test; no long-Y/predictor/PPO fitting or saved-trajectory policy-gain claim.

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


## Completed correction result

Code `de6ac97`, run `rolling-gt-y-s267`: all original224 factual outcomes retained,
originalY8/Z90 andtau0Y/U replay EXACT. Primary8 fullwindow queries and59dense
boundary diagnostic queries use the same unchanged Y/U semantics, no padding.
Main CPU label/statistical runtime1.43s, artifactsabout0.70MiB, zero new neural
fits/inference/simulation/policytraining. Fourteen semantic/regression tests pass.

124Z-discordant candidatepairs exist across the32 originalanchor panels;
50 have exactly tiedU atoriginaltau0. Primary cadence resolves46/50 with correct
sign, both states currentlyat-risk,>=8steps before the bad branch's first raw
height/6proxy-loss event, and no subsequent observed pre-event inverse>.02.
These46 comparisons belong to8anchors; first qualifying leads29–47steps,
pair median34. The anchor coverage is8/32=25%, descriptive anchorbootstrap95%
9.375–40.625%; not46 independent trials or a control-effect confidence interval.
All120/124 pairs have some primary correct pre-event separation, which includes
those already distinguishable atoriginaltau0 and is not the delayed-signal gate.
Of16 originalall-seven-utility-tied anchors, only4 contain different finalZ;
all4 exhibit a primary delayed signal. The other12 have no discordantZ pair
and therefore cannot test successful-versus-failed path separability.

Crucially6 ofthe8 signalanchors alreadyhave baselineZ1. Env33's baselineZ0
was already rescued by one-shot middle−. Env36 enters the primary8 count
because thumb− separates from middle+ atoffset32 (firstbad event65,lead33),
NOT because it separates from baseline onthe8-step cadence. These are useful
factual-path observability signals, not8 new baseline rescues.

### Two original opportunity examples

Env33/seed263/clock84: middle− andgrip+ originallyhave identicalfullY8/U1.25
butZ1/0. Offset16 yields1.25 vs1.1875, gap.0625; grip+ firstthreshold event48,
lead32. Thusrolling localY canresolve this originaltie. Baseline versusmiddle−
was already slightlydifferent atoffset0; offset48 scores.02083 vs1.25,
30steps beforebaseline's firstthreshold event78. No newstate-matched control
was attempted; this only shows later factual-path discrimination.

Env36/seed263/clock42:

| Offsettau | BaselineU | Thumb−U | Complete future endpoint | Interpretation |
| --- | --- | --- | --- | --- |
| 0,8,16,24,32,40,48 | 1.25 | 1.25 | 32..80 | Identical local labels |
| 56 | 1.166667 | 1.166667 | 88 | Both weaken equally; no ordering signal |
| 57 (dense diagnostic) | .125 | 1.125 | 89 | Gap1,32steps beforebaseline drop |
| 58 (dense diagnostic) | .083333 | 1.083333 | 90 | Gap1,31steps beforebaseline drop |
| 64 (next primaryquery) | unavailable | unavailable | 96 | Six futuresteps unrecorded; no fabricated labels |

Baseline firstpostqualification heightdrop89 enters the full32step window
atoffset57. The declared8stepcadence's lastsupportedquery56 misses it byone
endpoint; offset64 cannot be evaluated. This supports the horizon explanation
but **does not show the8-step rolling controller rescues this case**. Thumb−
Z1 still ends only1.838mm above its20mm threshold and descends; durable success
beyond90 remains unknown, asintheoriginalrecord. Dense queries are a boundary
diagnostic, not an after-the-fact replacement cadence.

### Intervention support and corrected research decision

Retrospective branchlookups onprimarytau0/8/16/24/32/40/48/56 returnfinalZ counts
24/24/24/24/23/24/24/24. These count independent originalworld path labels under
successivelydifferentH; no choices were switched in a simulator. The only
baselinefailed opportunity newly suggested bylate signal isenv36, andit is
outside supportedprimaryquery times. The prior GT-Z25/32 upper is merely
an original finitebranch-access ceiling. Actual rolling oracle performance,
attainable upperbound, samecurrentstate candidate contrast andactionauthority
for reintervention allremain UNKNOWN from this recording.

Correction toref13 interpretation: itsUNPROMISING label remainsvalid for its
fixedone-shot contract. It provides no no-go evidence against rolling short-Y.
The newPROMISING result is scopedto delayedfactualpath separability; it warrants
prioritizing a new rollingGT-Y same-current-state/prefix replay Decisionexperiment
before extendingYto90 orfittinga predictor. Do not replace theone-shot Zresult
with a fabricatedrolling gain. Nativeexecutionforecast/PPO/MSE tuning andfinal
matchedtraining-policy claim remainpaused/OPEN. Userasked firstthecheapcorrection;
this auditcompletes thatscope, withoutlaunching a newinteractioncampaign.

Outputs `outputs/cm-interaction-oracle/rolling-gt-y-s267/` include immutable
`frozen_protocol.md`, hashed`manifest.json`, allY/U/currentrisk in`rolling.npz`,
`rolling.csv`, allpair/event/lookup details in`result.json`, andstandalone
`rolling_examples.png`. Independentrawsemantic/statistical review andscoped
verification are archived withthecompletedrun. Earlierref13 data/candidates/
labels/gate andresults remain unchanged; no selective subsetretraining.


## Closing verification

Independent CPU review PASS (`independent_review.md/.json`, `raw_replay.json`,
`review_archive.json` in the outputfolder):54code/inputhashes checked, allfull
rollingY8/currentrisk/U/oldZ, rawfirst/postqualification events,124pair signs/
initialties/leads, queries, lookup diagnostics and267bootstrap exactly match.
Root independently checked example traces,lead summaries andoriginalbaseline
rescue attribution. Four of46 primaryqualifyingpairs have no subsequenteligible
query and thus no extra persistence evidence. The other42 stillcover all8signal
anchors; no gate, labels, cadence orthreshold were changed after inspection.
Fourteen semantic/regression tests pass; final repository verification is
recorded in `verification.log`. This correction closes the requested cheapaudit.
