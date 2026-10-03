# Fixed-budget physical action-contrast acquisition

Decision Probe. Question: does prioritizing disagreement about ACTION EFFECTS
produce a better learned physical contrast than prioritizing absolute response
disagreement or uniform labels? Positive permits a separately designed native
training-data acquisition experiment; negative stops this construction.

Reuse old randomized FIT492/493:3072rows,1536shared(acquisitionseed,environment)
blocks with two different actor histories each. TEST494/495 is independently
acquired IID7-arm data,3072rows/1536paired blocks; previously viewed, not fresh
Validation. Native source physics/behavior is legacy, not corrected655 or own
policy utility. Inherit already audited81current features by byte identity;
reaudit raw physical targets, assignment, propensities and row identities.
Observed output is3D `nextpos-currentpos-currentvelocity/30`, not contactforce,
force closure or stable lift.7categorical pulse arms, ±.01 reference commands
on three wrist translation axes; actual command/execution semantics inherited.

Strata=(FITacquisitionseed,motion),6x256sharedblocks. Select256initialblocks
with seed4500, quotas[43,43,43,43,42,42] in sorted stratum order. Wholeblocks
always retained:512initialrows. Three bootstrap ensembles independently
resample initialblocks WITH replacement withinstrata (seeds4510..4512), keeping
bothactors. All87→64ReLU→64ReLU→3models share initial seed4520 but bootstrap
datasets differ;600updates each, AdamW lr.001/weightdecay.0001/eps1e-8,
batch128, schedules4521..4523. Common initial512row mean/populationstd for
input87/target3, floors1e-5/1e-6; no pool/test outcome normalization.

For every unlabeled FIT current81context, evaluate all7hypothetical one-hot
arms in physicalmetres. Contrastscore=sum population ensemble variance over
3coordinates×3plus-minus effects. Absolutescore=sum variance over7responses
×3coordinates. Average score over two actors in a sharedblock. Acquisition
uses ONLY currentcontexts, frozenbootstrap models and block identities;
assigned arm/response labels of unacquired rows cannot affect the score.
Select256additionalblocks with SAME quotas: highestscore perstratum, stable
block-key tie break; uniform uses independentseed4530. Record all selections
BEFORE accessing extra labels. No outcome-selected quotas/filters/subgroups.

Fit three finalmodels (contrast/absolute/uniform),1024observed rows each,
identical initialseed4540 and1000update schedule4541; same model/optimizer/
initial-only normalizers above. Total4800newupdates:1800ensemble+3000final.
Store initialization, all schedules, optimizer checkpoints, first3states/losses
perfit and selections. Finalcheckpoint only; no test selection. No neural
actor/native simulation in this screen.

TEST effect predicts each plus-minus difference. For independent randomized
armA with recordedpropensityp, pseudo Z[j]=Y*(1[A=plus]/p_plus-1[A=minus]/p_minus).
Identified risk difference is sum9 `(Da²-Db²)-2*(Da-Db)*Z` inmm². It identifies
RELATIVE causal-contrast risk under randomization, not absolute effect RMSE.
TESTraw pseudo inherits no fitting nuisance. Preserve pairedactor environment
blocks within6(TESTseed,motion) strata. Fixed2000bootstrap replicates seed4550.

Fixed pooled gates: contrast-uniform point<=-1mm² AND one-sided95%upper<0;
contrast-absolute point<=-.5mm² AND upper<0; both TESTseed contrast-uniform
points<0; contrast-zero pooledpoint<0. PROMISING iff ALL6 gates, otherwise
UNPROMISING; source/coverage failure is not a scientific label. Per-motion,
overlap, factualEPE and ensemble distributions are diagnostic only.

Independent CPU NumPy all-model/candidate scores/ranks/selections, all held
contrasts/IPW/pairedbootstrap/metrics/gates; all rawFIT/TESTresponses/assignments
and metadata. Replay first3AdamWsteps of each6fit (18/4800 updates), remaining
4782not replayed. Finalprediction tolerances2e-6m; selectedranks must agree
exactly or fail audit, no favourable reranking. Cpu audits are file/statistics
and independent small-model arithmetic; main models runGPU. Engineering smoke
checks cancellation of common model-specific offsets, and exact risk identity
on a complete synthetic randomizedpopulation; smoke is not scientific evidence.

OnefreshidleGPU,<=600s/64MiB, protectedcard/code/source SHA, uniqueownedoutput,
stoponlyownedchild on contention/drift/budget/auditfailure. All3072FITpool and
3072TEST native windows are already paid; NOT1024online environment samples,
not online sample efficiency. No final policy utility, novelty, cross-hand or
journal claim. No local hyperparameter/seed/labelbudget rescue if gates fail.
