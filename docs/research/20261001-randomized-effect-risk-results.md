# Prospective randomized effect risk: completed, candidate not supported

Design frozen in0b1d2da; indexed-device correction510a00e. First runr1failed at
35.85s before collecting windows/scoring. Preserved without replacing its log.
Corrected `P-20261001-randomized-effect-risk-r2` completed137.84s, including
four new batches and frozen-model analysis. Total experimental cost173.69s.
One GPU4,2threads;8.29MiB outputs. No model was trained or selected on outcomes.

All3072windows acquired before termination;768per actor/evaluation panel,
256permotion, initial seeds492/493. Every motion's assigned counts are
37/37/37/37/36/36/36 forzero,x+,x-,y+,y-,z+,z-. Propensities are recorded and
audited. Assignment consumes an independent CPU generator, not native RNG.
Maximum sampled gaps per panel17.72,17.53,19.38,19.30mm. No partial windows
are excluded. True potential-outcome clones are neither needed nor asserted.

Risk difference is identified by the known-propensity randomized estimator;
negative means the first predictor is better. Units mm² sum over three response
coordinates and three wrist axes. This is NOT absolute conditional-effect RMSE.

| Comparison | Point mm² | Central95%descriptive interval | One-sided95%upper |
|---|---:|---:|---:|
|Nominal minus command|0.1223|[-0.9194,1.1866]|1.0126|
|Nominal minus zero|-0.3261|[-1.9851,1.3497]|1.1242|
|Command minus zero|-0.4484|[-1.3931,0.4554]|0.2961|

The nominal-minus-command points are0.0852/0.1044/0.1772for seeds411–413;
all have the wrong direction for the fixed gate. Both primary upper quantiles
also cross0. All three gates fail: UNPROMISING. Do not substitute favorable
motion/axis/actor subgroups or select a seed. Intervals are descriptive for this
fixed one-object substrate; no formal generalization or policy conclusion.

The11.18%historical factual advantage does not establish a useful causal
action-effect advantage in this new test. This screen provides no positive
evidence that either archived predictor beats zero effect robustly. It does
not prove the models have zero information or universally fail.

Independent CPU/NumPy audit reconstructs pseudo-targets from all physical
windows, nine individual risk points and saved risk arrays; bootstrap quantiles
and input SHA also match. Audit file `closeout_audit.json` accompanies the run.

Decision: freeze and end direct use of these archived factual models for
control. Next hypothesis learns conditional effects directly from randomized
physical supervision, with a new independent initial-seed cohort for testing,
matched newly trained factual/compute controls and a global-effect control.
No PPO is justified by the current result.
