# Decision: corrected gradient information, without refitting the forecaster

Forecasting primary fails10%global-control improvement. Stop its exact fits.
Broader secondary predictions and physical105labels permit a DIFFERENT cheap
question: can short-horizon physics reduce policy-gradient noise for the actual
longer holding objective? This is reused-data exploratory evidence, not fresh
Validation or a rescue of the failed gate. Existing Q-Prop/EPG/DR-PG techniques
already establish control variates; no methodological priority claim.

Frozen Cm/state-only/global30predictors fromr2; originaltest525/526only1536rows,
all3motions, no filtering. Query all8candidate primitives at each recorded
PRE-intervention state. Reward is independently audited physical105, not30.
No new model training, native simulation or candidate selection.

Fixed prospective future actor starting point: random seed751 two64ReLU layers
on FIT-normalized69features,8logits with final weights0 and bias[2,0,...,0].
Hence deterministic action0is the same frozen manipulation initializer.
Use all last-layer gradient coordinates: score vector(onehot_a-pi), outer product
with hidden activation augmentedby1. Earlier layer gradient is0at zero final
weights. Known assigned propensity1/8; importance pi(a)/(1/8).
Control-variate coefficient is fixed1, no coefficient/temperature/seed selection.

Estimator exactly follows the identity in physical-control-variate-literature.md:
sampled corrected residual plus full8-action expectation. Compare squared gradient
norm means against state-only and global-action control variates. Their population
gradient means agree under the specified randomized experiment, so second-moment
DIFFERENCES identify integrated gradient-variance differences; absolute variance
or task benefit is not identified by raw second moments. Verify exact finite-sum
cancellation for arbitrary rewards and misspecified q, and sampled probabilities
against saved assigned-action predictions.

Reused-data candidate gates: Cm secondmoment >=20%lower than BOTHcontrols over
ALL1536rows and lower againstboth in eachtestseed. Report allmotions and mean
gradient differences too; no formal mean-equality assertion from finite rows.
PROMISING permits a NEW prospective matched policy-training design with true
physical105return and the exact correction, never uncorrected dense shaping.
Failure ends this exact candidate. Single admittedGPU for model inference,
CPU statistics/algebra tests,<=300s/50MiB. No external authorization needed.
