---
schema: ref2dex.probe.v2
probe_id: P-20261006-actual-flow-y-ranking
experiment_id: P-20261006-actual-flow-y-ranking
date: 2026-10-06
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: b324e1e6abd82a612725407aeb693bfb2fb76eae
claim_id: C3
hypothesis_family: HF-actual-flow-y-ranking
probe_index_in_family: 1
seed_pool: probe
seeds: [271, 272, 273, 274]
decision_changed_if_positive: qualify the passing oracle representation for a separately frozen real rolling learned-Y probe
decision_changed_if_negative: retain GT-Y utility and diagnose the fixed predictor contract without PPO or execution forecasting
status: UNPROMISING
run_id: actual-flow-y-ranking-s271
---

# Ref14_3: actual hand flow to same-state Y ranking

## Decision Note / motivation

Mission B requires physical prediction to improve decisions. Ref14_1 rolling
GT-Y gives27/32 versus23/32 baseline; noise tolerance provides an offline .70
ranking reference, not a learned-policy guarantee. Test whether actual hand
motion supports held-anchor Y ranking and whether E/I bottleneck or hybrid
preserves more information than direct flow. Cheapest Decision Probe: immutable
full32-anchor repack and audited FK sidecar; no new simulator collection.
The user explicitly authorizes this stage in ref14_3. Native execution, desired
flow, PPO, critic and reward changes remain deferred.

## Frozen data and split

Inputs: `outputs/cm-interaction-oracle/ref14-assets-20261006-full32` and
`outputs/cm-interaction-oracle/ref14-assets-20261006-full32-dense`.
Require COMPLETED/PASS and SHA256 identities. Use only fully observed seven-slot
panels (43plans,339query rows expected), excluding certified pruned panels and
mixed actual records. Never impute missing candidates. Join seed/group/offset/
candidate and row identity exactly. H uses current relative physical72,
q/dq36, history q/dq PCA32, current hand-point PCA16, all fit-source only.
Raw corresponding actual flow is0→4/4→8,720values in current object axes,/.02m.
E12/I14 atstep8, original Y8 from32-step windows and U=Y7+.25Y3-Y6 unchanged.
No actions, arm IDs, future object/contact or Z in predictors.

Four outer folds shuffle sorted (seed,env-row) anchors with seed271; all offsets
and candidates of an anchor stay together. Each outer source uses three inner
anchor folds (seed272) to generate E/I OOF features, with its own H and label
normalizers; outer held anchors never enter any preprocessing or fit. Full
outer-source E/I supplies outer-test features. Aggregate only outer-held Y
predictions. Shared-solver groups and previously exposed cohort remain Probe
limitations; anchor OOF is not independent solver-group Validation.

## Matched arms and budget

H control; Direct(H,F); Bottleneck(H,predE/I); Hybrid(H,F,predE/I);
GT_EI(H,trueE/I) diagnostic. All Y heads have identical input slots H+720+26,
zero unused slots,64/32 tanh layers,8outputs; same initialization seed273,
batch64schedule274,400AdamW updates,lr.001/weightdecay.01/clip2.
Y supervision is all8 standardized MSE, scale floor.05; no test tuning, output
clipping or ranking-loss changes. Consequence heads use the same64/32 backbone,
26outputs,400updates; E/I scale floor.001, initialization/sampling273/274.
Report Y-head and total pipeline parameter counts separately: bottleneck/hybrid
have extra consequence fits; equal Y heads do not imply equal compute.
Four full plus12inner consequence fits,20Y fits. GPU0 when idle, oneGPU,
main<=600s, smoke<=120s, saved-weight replay<=120s; <=150MiB predictor artifacts,
<=1GiB packaged inputs and<=200MiB dense sidecar. Stop on leakage, hash drift,
nonfinite, resource conflict or cap. Fixed code commit before main fitting.

## Metrics and decisions

Primary pooled GT-strict same-state pair accuracy; GT ties excluded, prediction
ties=.5, all tied counts disclosed. Also per-fold and per-anchor accuracy,
mean/median top1 regret (first-index ties), all8MSE and per-channel MSE.
Frozen within-panel flow shuffle (seed272): same candidate permutation for all
arms, recompute E/I from shuffled flow, then score the held panel. H and GT_EI
are negative controls. Report accuracy degradation and anchor-bootstrap95%CI
(seed271,2000 draws); do not equate repeated panels/pairs with independent data.
Offline screening: >=100panels and >=100strict pairs; candidate arm passes if
accuracy>=.70, median regret<=.125, and shuffle degradation>0. H should tie
at.5 on strict pairs because same-state inputs are identical. Insufficient
coverage is UNCLEAR; adequate but no arm passing is UNPROMISING; any pass is
PROMISING for this oracle offline contract only. Hybrid/bottleneck uniqueness
requires >=.03accuracy over Direct and positive paired anchor-bootstrap lower
bound; otherwise unique contribution UNCLEAR. No Z retention from lookup.
A passing result permits designing a new real rolling intervention; it does
not certify deployable inputs or authorize unlimited rolling costs.

## Limitations / future evidence

Actual hand motion is post-treatment and may carry object responses. Full-panel
admission depends on baseline upper-bound pruning; this is selected support,
not every query. Cohort32 and only four shared solver groups constrain external
validity. OOF/source-full distribution mismatch, finite training budget, model
class and single optimizer seed remain limits. Future independent multi-seed
Validation and prospective flow availability, real rolling learned-Y Z, then
matched trained-policy Cm-on/off are still required by Mission.

## Completed result and root decision

Main code commit `b324e1e`, run `actual-flow-y-ranking-s271`: four-fold held
predictions on339complete panels/32anchors (2373candidate rows),2276strict
pairs and4843GT-tied pairs. 25anchors provide strict pairs. GPU0 main30.79s;
36fixed fits (16consequence,20Y); zero new PhysX. Models30,392,909bytes.
Full asset packaging and dense FK passed on all four groups; no partial snapshot
or unobserved candidate imputation enters fitting.

| Arm | Held pair accuracy | Shuffled | Mean top1 regret | Median regret |
| --- | ---: | ---: | ---: | ---: |
| H | .499780 | .499780 | .040653 | 0 |
| Direct | .573814 | .498682 | .129394 | 0 |
| Bottleneck | .578207 | .483304 | .107147 | 0 |
| Hybrid | .579525 | .499121 | .126690 | 0 |
| GT_EI diagnostic | .673550 | .673550 | .081828 | 0 |

All three candidate arms fail the frozen .70accuracy screen: UNPROMISING for
this model/budget/support contract. They use flow: shuffle degradation is
+.07513[.02053,.13331],+.09490[.03553,.15226],+.08040[.02096,.14415].
Bottleneck−Direct=.00439 CI[-.06238,.06863]; Hybrid−Direct=.00571
CI[-.02899,.04507]. Unique predicted E/I contribution remains UNCLEAR.
The bootstrap is anchor-descriptive, not four-group independent Validation.
Each Y head has60,136parameters; consequence head59,066; bottleneck/hybrid
inference pipeline119,202, with16extra consequence fits shared between them.
All median regrets are zero on tie-heavy support; mean regrets reveal that the
learned selectors currently underperform baseline-first H in offline regret.
No deployment or actual rolling Z is inferred from these numbers.

Saved-weight GPU replay PASS in3.95s: H/label normalizers, full nested OOF,
held predictions, shuffle outputs and all summaries exactly reconstructed
(max errors0). Source Y-head training strict accuracy: Direct .653–.701,
Bottleneck .585–.666, Hybrid .676–.711, GT_EI .691–.705. Source fit is already
modest and held performance drops; these observations cannot uniquely identify
objective mismatch, underfitting or limited generalization. They do NOT establish
that additional epochs, a ranking loss or physical information would solve it.

Independent read-only code/CPU/NumPy review agrees on split identities,
input equality, strict pairs, labels, regret and shuffle. One small numerical
issue: seven-candidate panel329 crosses predict's256-row final batch boundary;
identical H produces utility spread1.1921e−7 and5pseudo-ranked strict pairs.
Other2271H strict pairs are exact predicted ties; all main candidate/GT_EI
strict pairs have no prediction ties. Original H .499780 is retained, not
silently rounded in raw evidence. Future inference should align chunks to seven
candidates; no training or control conclusion is affected by this tiny defect.

Root decision: do not launch real rolling learned-Y or execution/PPO. Preserve
rolling GT-Y utility and the positive action-sensitivity signal; stop this fixed
fit, without calling actual-flow/E/I globally ineffective. The next cheapest
Decision is to preregister a source-only panel-ranking objective comparison
against these immutable heads, checking whether source ranking can improve and
transfer before spending on simulation. No extra epoch/seed retries occurred.
A new ranking-objective Probe must freeze its objective and gates before fitting;
independent held support and final matched training-policy Validation remain due.

Artifacts in `outputs/cm-interaction-oracle/actual-flow-y-ranking-s271/`:
`protocol.md` freezes pre-run card bytes; `manifest.json`, `models.pt`,
`result.json`, `replay.json`, `report.json`, `ranking.png` and
`independent_review.json` preserve provenance, signed comparisons, per-fold/
per-anchor metrics, parameters, ties, source-fit diagnostics and review.
110Task tests pass,1skipped; repository changed verification passes.
