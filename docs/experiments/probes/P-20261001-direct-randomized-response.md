# P-20261001-direct-randomized-response

Next Decision Probe. Freeze this design before fitting and before collecting
the new test cohort. Hypothesis: direct randomized contrast supervision improves
conditional action-effect prediction beyond matched factual training, a global
effect matrix and zero effect. The statistical methods below are established;
using them alone is not the intended journal novelty claim.

## Data and question-dependent decision

- Fit ALL3072windows from randomized-effect-risk-r2, seeds492/493, actors286/287.
  That dataset was inspected in the previous screen. It is now training data,
  never described as pristine Validation evidence for the new method.
- New test cohort: same actors, initial seeds494/495,four768-env batches,3072
  windows, no test model selection. Same current-proximity acquisition,pulses,
  source/config/checkpoint protection, complete-window requirement and budgets.
- Independently sample arm0..6perenvironment withp=1/7using a separate CPU
  Generator(9900+evaluation_seed); no forced label-count balance. Draw before
  first physics, execute only at trigger, record all realized arm counts. This
  maintains assignment independence even if other environments have weak numeric
  solver interactions. Preserve shared initial-seed/environment blocks across
  actors when bootstrapping. Never retrospectively rebalance assignments.
- Primary one-step world residual over passive velocity; context is CURRENT
 63physical/geometry features plus18reference commands, with six overwritten
  reference-command inputs canonicalized to0. No future physical state input.
  Do not tune a contact threshold, use historical motion identity as a predictor,
  or change horizon after observing errors.

## Fixed fitting procedure and strong controls

- Seeds611/612/613;two128-unitSiLUlayers,Adam,lr0.001,batch256uniformfitrows.
  Current/input/target normalization uses fit only; no heldout test checkpoint.
- `direct_contrast`: predict a3x3world contrast matrix from81context inputs.
  Use independently acquired randomized labels, not contrast of two cold replays.
- Crossfit nuisance mean response: two folds are acquisition seeds492/493,
  keeping shared initial/environment/actor blocks together. Each fold fits a
  current-context81->128->128->3nuisance model on the OTHER fold for1000updates.
  Every nuisance input/target scale uses its own fit fold only. Report these extra
  fits/costs; future test outcomes never train a nuisance model.
- Residualized pseudo-target for each axis is
  `I(T=+)/p+*(Y-mu_oof(X))-I(T=-)/p-*(Y-mu_oof(X))`.
  Its conditional mean remains the physical contrast under randomization.
  Direct contrast model gets1000updates with physical normalized MSE on this
  matrix, fixed weight decay1e-4. Do not clip pseudo-labels or select subsets.
- `factual`: matched current context81plus6pulse indicators (87inputs),
 128/128SiLU/3outputs,1000updates on the same randomized physical outcomes.
  Evaluate candidate arms at the SAME current context. Same weight decay1e-4.
- `factual_compute_matched`: same factual model but3000updates to match direct
 1000plus two1000-update nuisance fits perseed. Report measured costs separately;
  counting updates does not assert identical FLOPs, output capacity or data access.
- Strong constant control: global FIT mean residualized pseudo-contrast. Also
  report unadjusted global FIT IPW mean as a diagnostic, and zero effect.
  No test response chooses a global mean, penalty coefficient or training step.

## Identified scoring, gate and resources

Use fresh test propensities and ORIGINAL observed Y to compute identified risk
differences against every control. Do not use test crossfit nuisance estimation
or infer absolute effect RMSE. Average per-model losses across fixed seeds,
not error of a selected ensemble. Descriptive bootstrap2000replicates,seed12026,
paired environment blocks within6initial-seed/motion strata across actors.

PROMISING requires the one-sided95%upper risk-difference bound for direct minus
EACH factual,factual_compute_matched,global_residualized,zero to be<0, and direct
minus factual point risk<0in every optimizationseed. Else UNPROMISING. All
subgroups are reported; none replaces the pooled gate. Positive results justify
a new held policy/task screen; negative results stop this particular learner.

Fit GPU5 or an independently admitted idle GPU;test collection GPU4sequential.
At most2GPUs concurrently,total<=4. Fit<=900s/100MiB;test<=3600s/2GiB. Freeze
sources and model checkpoints before test acquisition. Preserve all failures.
This is a predictive Decision Probe, not policy Validation, cross-object evidence
or a completed top-journal contribution.
