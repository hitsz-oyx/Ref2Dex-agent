# Equal-budget joint physical critic: no required policy gain

P-20261003-budgeted-physical-critic-r2 completed with **UNPROMISING** Probe
status. Design22e47f5, scene-size amendment42e2cbd, implementation48f8d2b,
pre-fit normalization correction09e76ac. Six frozen conditions: two pass, four
fail. These are exploratory results from one optimization initialization,
not formal Validation or a general refutation of physical models.

| Evaluation | P0 | Cm auxiliary Q | Action-removed auxiliary Q | Task-only Q |
| --- | ---: | ---: | ---: | ---: |
| All four seeds, each common method /768 |244|383|399|Not pooled across distinct controls|
| Seeds655/656, each method /384 |129|200|198|189 coldQ|
| Seeds657/658, each method /384 |115|183|201|220 equal-budgetQ|

Cm-minus-off is -2.083pp pooled. Cm-minus-coldQ is +2.865pp in blockA,
below the fixed +5pp requirement. Cm-minus-equal-budgetQ is -9.635pp in
blockB. The +5pp overP0 and motion1 safeguard pass; +5pp overoff, coldQ,
budgetQ and each-seed noninferiority fail. Cm counts96/104/95/88 per192 on
655/656/657/658; off101/97/104/97. The common Cm/off weights are bitwise
identical across both evaluation bundles. No selectively pooled control.

| Motion, common methods /256 | P0 | Cm | Action-removed |
| --- | ---: | ---: | ---: |
| 0 |0|0|0|
| 1 |214|208|202|
| 2 |30|175|197|

ColdQ and equal-budgetQ also have zero motion0 success. In blockB,
equal-budgetQ motion1 is105/128 and motion2 is115/128. This retains partial
self-trained manipulation evidence; it does not establish Cm utility,
all-motion competence, generalization, force closure or hardware results.

All panels use768 native environments and the original scene layout.
Fresh short651/652 run101controlticks with no complete task105 labels;
common653 and extra654 run202ticks. Physical pool2304pairs; common measured
task labels768; budgetQ measured labels1536. Input SDK normalization comes
only from common653, available to all methods within their budgets. The
extra654 never enters Cm/off/cold training or input normalization. Target
normalization uses physical data only; task-only controls have zero auxiliary
gradient. The common self-trained P0 foundation cost is separate and unchanged.

Both data allocations cost310272environment-controlticks:
1536*101+768*202 versus1536*202. Aggregate fresh collection plus evaluations
uses1085952ticks. Actual clock: control approximately1/30s, simulation1/60s,
controlFrequencyInv2, substeps4; controlticks are not internal solversteps.
There are6000critic optimizer updates and4000actor optimizer updates;
3000aux-bearing critic updates are a subset of6000, not an additional cost.
All four critics and all four actors use their respective common initial
parameters. Critics are frozen before policy fitting; physical predictors
are not called by policy objectives or by deployed actors.

All eight native audits pass: P0 reconstruction, private balanced assignment,
zero duplicates/Gaussian/antithetic draws, original placements, actual held
execution, PD target map, full25002vertex mesh and full105tick task criterion.
Training audit reconstructs all current/future SDK inputs and task labels,
verifies normalization/data separation, initial parameters and final neural
outputs. Current deployment observations reconstruct exactly; independent
mesh clearance maximum3.397e-7m, native target maximum4.769e-7. All final
critic/physical/actor NumPy comparisons are within1.795e-6. Critic1500 and
actor500/1000 weights, optimizers and private random generators are retained.
No independent optimizer-trajectory replay is claimed.

The first valid short651collector was completed before the normalization
accounting correction, which preceded any complete task data, fitting or
policy evaluation. r1 then stopped under protected-code drift; r2 inherits
exactly that artifact under SHA, independently audits it, and never repeats
its simulation. r2 executes1008384new ticks plus77568inherited ticks. Failed
r1 and original code remain retained. r2 wall1059.607s,1757385080bytes;
r1+r2 recorded wall1142.990s, within the combined2400s limit. Actual fitting
uses62.448s on an idle GPU. Every owned parent/child PID is absent at closeout.
Protected input hashes remain unchanged within each completed execution.

ResultSHA256 `a08d5bb10b43ed3cb8b245b8a244d84b7b86b61398171a9c73b7983ff403578f`.
ModelsSHA256 `ecf49c0f9e3eafbc826788c3d817224572ec7949e93a0fff2587b57ee701d08e`.
Common actorSHA256 `cc0b584b2ad7b11ccfc678cf78ce41b209536542422d6fc57bffe12f6a7b8996`.
Budget actorSHA256 `b88bc5cbbce7d0a4f86a2018795a7f46fc49a34963c93c440bc75a213af0e06c`.

Close this fixed joint-auxiliary/data-allocation recipe. No coefficient, width,
steps, seed, physical horizon, truncation ratio or label-budget rescue scan.
No Validation is launched from this failed gate. Generic joint predictive
representations already have prior art; this negative Probe establishes
neither method novelty nor publication readiness. The original core research
question and goal remain active; next work requires a higher-level decision.
