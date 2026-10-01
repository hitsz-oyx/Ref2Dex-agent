# P-20261002-support-feature-policy-training

Decision Probe: freeze the design in
[policy-training decision](../../decisions/D-20261002-support-feature-policy-training.md).
Actual physical105reward, frozen predictive features, matched state-only and
privileged-global trained policies. No model reward or new predictor fitting.
Macro policy learning is reference-conditioned contextual off-policy RL, not
general continuous manipulation or a novel algorithm.

Run id `P-20261002-support-feature-policy-training-r1`. Initialize before collecting
training529--540; after each fresh768-environment panel and independent full
geometry/control audit, one full-batch update per policy. Seed752, Adam0.01,
beta.9/.999,eps1e-8,foreachFalse,gradclip10,noentropy. Common FIT per-motion
physical105mean baseline is independent of assigned action. Fresh final-only
evaluation541/542randomizes four policy groups before physics. Keep all motions
and all9216training/1536evaluation trajectories.

GPU simulation/model computation, CPU tiny initializer and independent NumPy
statistics/forward/gradient/Adam audits. Initializer CPU avoids GPU startup for
three small random weight packets; all actual inference/training uses GPU.
CPU model reconstruction is a deliberately independent numeric audit of GPU
execution, not the primary model computation. At most1800s/3GiB, one freshly
admitted GPU per stage. No shared or external output writes/checkpoint overwrites.

Primary Cm pooled physical105success exceeds BOTH trained controls by>=5pp and
is no worse in each evaluation seed. Original failed forecast/noise/position/
disturbance gates stay unchanged. Positive result permits multi-training-seed
Validation; failure ends this exact feature-transfer learner without tuning.
Source actor may be loaded only for native evaluator bootstrap, never acts or
provides weights to our learned base/macro policies. All artifacts and failures
remain retained; terminal results will be written separately.
