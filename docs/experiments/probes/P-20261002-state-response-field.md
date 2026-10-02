# State-conditioned immediate object response: fixed Decision Probe

Question: does current-state conditioning predict randomized request response
better than a motion/phase response mean? If yes design direct actor integration;
otherwise stop this field recipe, with no width/steps/target/seed scans.
This is information screening, not Cm policy utility or Validation.

Cheapest clean distribution: reuse initial547 only, all three stochastic arms
with identical u00 mean0/std0.05. Fit baseline drift b(x) on unchanged547 only;
fit G(x)=E[epsilon outer (Y-b(x))|x] on stochastic547. Fresh test570 only using
the SAME u00 Gaussian behavior, with no optimizer/policy updates. Never use
the old final568/569 or updated behavior as zero-mean data. Common request
streams across the three replicas are preserved and jointly clustered.

Input x=current normalized70 features. Label Y=actual next-minus-current object
position/.005m and linear velocity/.05m/s,6outputs, exactly the independently
audited native target. Twelve standardized Gaussian request innovations give
72response outputs. Neither executed targets nor future states enter x.
G measures Gaussian-smoothed request response including projection, not an
executed-action intervention, open-loop derivative or guaranteed controllability.
Selecting the original105reference ticks stop-74..stop+30 uses known time only,
ALLthree motions, no future success/contact eligibility. Exactly20160baseline
and60480response fit transitions,192/576episodes;60480response test transitions.

Fixed b and G networks70->64ReLU->64ReLU->6/72, seed1081/1082, zero final layers,
Adam3e-4,2000steps EACH, batch4096 with replacement, fixed private seed1083/1084.
No output normalization, clipping or early stopping. Baseline is frozen before
G fitting. Strong control: FIT-only mean epsilon outer residual in each motion
and four known-time phases: before lift_start, lift_start..stop-75,
stop-74..stop, after stop. Empty FIT cells are zero; no TEST statistics fit models.

Primary paired risk difference D=mean(||G_state||²-||G_global||²
-2< G_state-G_global, epsilon outer (Y-b(x)) >)/72. This omits a common noisy
target-norm constant; D<0 favors conditioning. Require D<=-0.10*FIT global
mean squared response energy AND upper exploratory95%CI(D)<0 AND positive
nonzero FIT global energy. Otherwise UNPROMISING (invalid audit -> FAILED).
2000bootstrap resamples of192common-stream episode indices, retaining allthree
replicas/ticks together, seed1085. Conditional one-panel uncertainty, no formal
independent training replication. No retuning after seeing TEST.

Fresh idle single GPU4 admission, native768env/202ticks, all raw states retained;
GPU b/G fits and bulk inference; CPU physical/model audits and bootstrap only.
<=3600s/1GiB total new data. Unique run directory, fixed committed code and
hashed inputs; abort only owned child on contention/budget/source drift.
Independent native audit retains all original physical, request, target, PD,
state-write and full-mesh105 checks; a new cohort wrapper permits only570 with
u00 stochastic behavior without editing the old fixed-cohort auditor. Preserve
all old failures/checkpoints/paper inputs. No official actor calls/initialization.
