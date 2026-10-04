# Action-contrast disagreement acquisition: completed negative

P-20261003-contrast-acquisition-r1 completed **UNPROMISING**, code
`4baf9a7c426c7d8ed532bacc5246fbac5b8ee06c`. The prospective card fixes512initial
labels plus512additional labels for each of contrast-disagreement,
absolute-disagreement and uniform acquisition. All shared actor/environment
blocks remain intact; quotas, initial models, normalization and final-model
initialization/update schedules are matched. Acquisition uses unlabelled
current contexts and hypothetical commands, not the unacquired assigned arm
or response. No outcome-selected subgroup or hyperparameter rescue.

| Contrast acquisition minus comparator | Point relative risk (mm²) | One-sided95% upper | Central95% descriptive interval |
| --- | ---: | ---: | ---: |
| Uniform acquisition | +0.362847 | +2.202990 | [-1.863499,2.560501] |
| Absolute-disagreement acquisition | -1.923996 | -0.126221 | [-4.183198,0.261130] |
| Zero-effect predictor | -2.988855 | -0.446212 | [-6.013878,0.010139] |

Three of six gates pass: absolute-disagreement margin/upper and better-zero.
Uniform margin/upper and both-test-seed superiority fail. Contrast-minus-uniform
points are +0.069200mm² on494 and +0.656493mm² on495. Signs favor uniform but
the pooled descriptive interval includes zero: no formal claim of harm.
The negative one-sided upper against absolute disagreement does not imply a
central95% interval excluding zero. Their different definitions are preserved.

Additional256block selections overlap107betweencontrast/absolute,
50contrast/uniform and53absolute/uniform.512initialrows are common;1024observed
labels per final learner. Whole-block bootstrap initial models contribute
1800updates and final fits3000:4800actual new updates. MainGPU6 fit17.255559s,
CPU smoke1.763641s/audit2.333500s; parent22.104057s/9923571bytes. Zero new native
ticks or actor updates. Owned parent718566 and children718598/718717/719585
have exited, protected inputs unchanged.

Independent audit rebuilds all3072FIT and3072TEST raw physical responses,
assigned arms/propensities and pseudo labels, every new model forward, all
bootstrap block selections/ranks/final training selections, initial-only scales,
common model initialization/schedules and complete paired risk/bootstrap/gates.
Maximum physical forward difference5.97e-9m.18early AdamW updates manually
replayed, parameter max4.83e-7/loss1.57e-7; remaining4782not replayed. Context
geometry inherits the previous audited packet by protected byte identity;
no fresh FK reconstruction or subagent review is claimed.

The evaluation identifies RELATIVE conditional physical-effect risk under the
original independent randomized test assignment, not absolute causal-effect
RMSE. Bootstrap uncertainty is descriptive for fixed fitted models, not
independent training-seed Validation. All test windows were previously viewed,
with legacy source physics/behavior; no corrected-physics, own-policy,
cross-hand, novelty or stable-grasp utility result. All3072FITpool+3072TESTnative
windows were already paid, so1024selectedlabels cannot be called1024environment
interactions or online sample-efficiency savings.

Close this exact disagreement acquisition recipe. Its advantage over absolute
disagreement cannot rescue failure against uniform. Do not launch new native
collection or actor training from this failed gate. This does not reject all
active exploration/replay mechanisms. Next incorporate actual recent evidence
into the paper while preserving the unmet mission and full failure record.
Goal ACTIVE; journal NOT READY.
