# P-20261001-null-action-recovery

Decision Probe; independent from the frozen fresh causal-transfer run. No
published-model reproduction, Validation, or first-principle novelty claim.

## Question and decision

Does predictor-level inverse recovery encode physically overwritten action
coordinates into object predictions? Positive evidence motivates physically
canonical action targets before investigating inverse losses for manipulation.
Negative evidence discards this particular interference mechanism. This tests
an objective boundary; it does not show that inverse learning always fails.

## Ground-truth invariance and population baseline

Native action coordinates7,9,11,13,16,17are overwritten by dependent finger
targets. With current state and the remaining12commands fixed, replacing these
six entries does not change ANY native PD target. Test the exact mapping on
held states and independently sampled replacements, bitwise. No physics rerun
is needed for this code-level invariance. Full observation/reward conventions
may explicitly include raw actions; the present target is physical object motion.

Independent uniform[-1,1] null commands, normalized by1/sqrt(3), have zero mean
and unit variance. Any inverse estimator from a truthful physical transition
and current state therefore has population normalized MSE>=1on the null block,
or>=6/18=1/3in the full18-D average. This elementary conditional-variance bound
is not claimed as a new theorem. Finite sample estimation will fluctuate.
Recovery below this floor from MODEL-GENERATED predictions indicates that those
predictions encode null commands. It need not prove harmful deployed planning.

## Fixed data and matched variants

- Reuse immutable factorization-r1 `features.pt`:fit32902,test32896historical
  episode-disjoint frames; physical residual targets and63current-state features.
  No fresh causal-transfer outcomes, inputs or models are touched.
- At every fit minibatch replace all six null commands independently by uniform
  draws. Labels remain fixed, because PD targets remain fixed. Null augmentation
  is shared by all variants, not privileged extra physical data.
- Same81->128->128->3SiLU effect MLP and66->128->128->18inverse head.
  Inverse input is standardized current state and MODEL-GENERATED normalized
  object residual, never the observed future object state.
- `factual`: physical MSE only, raw18action input.
- `full_inverse`: physical MSE+1.0*normalized18-D inverse MSE, raw18input.
- `effective_inverse`: same but inverse loss ignores six overwritten targets;
  sum over12coordinates divided by18keeps each physical dimension's weight.
  Forward model still has raw18input, isolating the recovery-target change.
- `quotient_inverse`: same effective inverse objective and raw18input slots,
  with six overwritten forward-input coordinates hard-set to0. Exact forward
  null invariance follows by construction; reporting it is not empirical novelty.
- Seeds511–513,1000Adam updates,lr0.001,batch256halfnear/halfall, fixedfinal
  weights. Same weight initialization/minibatch/null-draw schedules across variants.
  State/action/target scales from fit only; null scale is known distributional SD.
- Held null replacements use fixed independent RNG999/1000and the SAME physical
  rows/effective commands. Report near/all factual RMSE (both replacements),
  paired null prediction RMS, full/null/effective inverse errors, everyseed andcost.

## Frozen gate and budget

PROMISING for this *interference hypothesis* requires full-inverse seed-mean near
null-recovery MSE<=0.8, null prediction RMS>=1mm and>=2times factual-only null
prediction RMS, greater null response than factual-only in everyseed, and factual
near RMSE<=1.25times factual-only. Otherwise UNPROMISING. Canonical/quotient
variants are diagnostic; do not retroactively substitute a more favorable gate.

One separately admitted idle GPU5 while GPU4 runs physical collection,2threads,
<=900s,<=100MiB. This obeys the simultaneous<=4GPU limit. Stop for PD mismatch,
source/feature drift, nonfinite tensors, budget or admission failures. Preserve
failures. Do not choose a lambda,seed,representation or target after testing.
