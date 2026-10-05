---
schema: ref2dex.probe.v2
probe_id: P-20261005-gt-consequence-sufficiency
experiment_id: P-20261005-gt-consequence-sufficiency
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-gt-consequence-sufficiency
probe_index_in_family: 1
seed_pool: probe
seeds: [231, 232]
decision_changed_if_positive: prioritize action-conditioned consequence prediction on this early-hold representation
decision_changed_if_negative: distinguish GT prognosis from missing action-related information before further Cm fitting
status: PLANNED
run_id: gt-consequence-s231
---

# Do GT physical consequences preserve task-relevant action information?

Result: Pending frozen analysis of the existing 854 randomized ref7 windows.
Decision: Compare GT prognosis, remaining action information and cross-half arm consequences before any Cm training.

## Motivation and Decision Note

Ref8 asks to close action→(E,I)→Y on strong-action-response early-hold data.
This is a Decision for the Task's GT-usefulness-before-Cm subgoal, ultimately
serving trained-policy matched Cm-on/off utility. Reuse ref7; no new simulator
samples. Existing local ref7 joint gate remains UNPROMISING; this is a NEW
representation question authorized by ref8, not a retrospective gate rescue.
GT prognosis alone is insufficient: strong baseline failure state can predict
Y without capturing action-induced task variation. Small incremental action
benefit also does not identify causal mediation: randomized a does not randomize
post-treatment E/I; confounding, measurement error, weak action→Y, model error
and power remain. Neither model comparison nor arm regression proves sufficiency.

Root chooses six matched scorers H/Ha/HE/HI/HEI/HaEI plus three PREDECLARED
signed-force extensions HI_signed/HEI_signed/HaEI_signed. Main E12/I14 are
unchanged native readouts; signed extension adds asinh of 18 world net-force
components. No representation/epoch/seed search. Cheapest discriminator is
one fixed fit per scorer and CPU statistics on already saved observations.
One currently idle physical GPU6; timeout600s inclusive; outputs≤100MiB;
no simulation, Cm, selector, policy training or external modifications.
Stop on input/code drift, nonfinite values, environment overlap or resource
conflict; retain failed artifacts. Positive opens a separate consequence
predictability Probe; negative/unclear reports which link is unsupported,
without abandoning the core claim. Mission/Campaign boundaries unchanged.

## Frozen protocol before outcomes

Input `per-finger-control-s227/interventions.pt`, SHA256
`138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149`.
Original collection commit a81b36d, 15arms/K8/20% driver range, all854 complete
windows including193 early failures. Recheck dose/geometry and attach the full
finger PD/q/tip amplitude table to this card. Readout E12/I14 and signed18
only at step8; Y ONLY step9..16/32. Six original continuation heads, plus
late physical height-failure (<2cm once) and height-held fraction (≥3cm).
All risk trials retained; no future survival filter or Y in H.

Environment cluster=episode_id modulo168; motion-stratified80/20 split seed231;
all waves for an environment stay on one side. H=train-only standardization
and PCA32 of historical1390/actor1442/context435/base18, CONCATENATED with
train-standardized current physical72 to protect explicit present-state
information from PCA compression. Action=14 nonzero-arm one-hot indicators;
no post-treatment realized action/PD/baseline-feedback sequence as an input.
E/I/signed and all8 Y heads standardized using train only (scale floor.001).
Zero padding gives all9 models identical input width, parameters and initial
weights, seed231, MLP64/32 tanh, AdamW lr.002/decay.001, clip2,
200 full-batch epochs, no validation selection. Save train loss traces,
normalizers/PCA/split/weights/predictions. Last50 loss change reports fitting
limits, not convergence proof.

Primary normalized error averages contact32, late physical height failure,
height-held fraction. Report all8 heads, physical failure AUC/Brier and
factual cross-arm prognosis ranking within motion×phase quarter. Ranking
includes ONLY strata with≥10 eligible pairs; not same-state action ranking.
Paired bootstrap2000 on held-out environment clusters seed232 gives fixed-fit
95% intervals; it does not include training-seed variation.

Support: ≥120 test windows, ≥25 test environments, ≥5 test trials per motion,
≥15 height failure AND nonfailure test cases, ≥3 held-out trials for EACH arm.
GT prognosis PROMISING if HEI vs H primary gain≥10%, physical-failure gain≥5%
and both bootstrap lower95>0; UNPROMISING if support is adequate but these
fail, otherwise UNCLEAR. Extensions do not rescue the main registered label.
Conditional action increment is small only when HaEI vs HEI primary AND
physical point gains≤2% and upper one-sided95≤5%. Report HI/HE, Ha effects
and the signed extensions with identical comparisons, without significance
claims from picking the best scorer.

Action-related check: independently residualize joint [E12,I14,signed18,Y8]
on the original CURRENT-ONLY nuisance controls and arm indicators, full cohort
and two five-wave halves. Preserve signed/vector contrasts and all14 arms.
Cross-half, leave-one-arm-out transfer fits only13 source-half arm contrasts,
uses feature scaling and uncentered PCA3 learned on those13, ridge1 without
intercept, predicts held arm in the OTHER half. Report E/I/EI/EI_signed and
zero/source-mean baselines. No in-sample 14-arm many-feature fit. Descriptive
bridge positive requires EI transfer contact32 AND physical-failure gains
vs BOTH baselines≥10% with both correlations≥.30. This coarse bridge is
not a mediation estimator; shared-zero/noisy contrast uncertainty remains.

Overall chain PROMISING only if adequate support, main GT prognosis positive,
small remaining-action increment AND the EI arm bridge is positive. Otherwise
keep per-link labels: absent GT gain is UNPROMISING for this representation;
positive prognosis without action bridge is UNCLEAR for actionable sufficiency.
This cannot form a formal scientific conclusion or complete policy utility.

## Limitations and future evidence

Single collection and fit seed, official frozen source actor used only as
research substrate. Current I aggregate net-force/proximity≠paired contact,
slip or friction margin. E8 may reflect failure already beginning; retain all
trials, disclose early-failure sensitivity descriptively only. Negative arm
bridge may reflect noisy14-arm estimates or nonlinear/state-specific effects.
A richer H, model-capacity/fit adequacy, independent validation, matched
same-state candidate comparisons and trained-policy Cm causal utility remain
future evidence if a concrete decision requires them; no sweep by default.
