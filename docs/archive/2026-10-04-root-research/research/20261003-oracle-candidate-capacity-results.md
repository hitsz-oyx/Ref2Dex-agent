# Eight-option capacity is already attained

P-20261003-oracle-candidate-capacity-r1 COMPLETED / UNPROMISING, code1c08b24.
0new optimization,6000inherited task-Q updates,18new full202tick worlds:
2changed stable-ranking choices plus16previously unexecuted motion0branches.
All scenes use seed763/12instances, one controlled subject and11P0 background
controllers. Existing84short queries and27full worlds (baseline+26deployed)
are reused by SHA. No repeated collection or model fit.

## Actual stable selector utility

| Selector | Full105 successes /12 | Motion0/1/2 |
| --- | ---: | --- |
| P0 |5|0/4/1|
| State + option |8|0/4/4|
| Short oracle effect |8|0/4/4|
| Short oracle interaction |8|0/4/4|
| Joint short oracle |8|0/4/4|

Raw-logit selection fixes float32 sigmoid saturation; CUDA and independent
NumPy agree on ALL48subject/arm choices. I/joint subjects1/7 move from option0
to3/4. Both actual revised executions succeed, so original and corrected
utility counts coincide. Probability maxerror<=7.16e-7; raw-logit maxerror
5.0068e-6<=1e-4. The bug is real but does not explain absent incremental
utility on this panel. Original r2records remain unchanged.

## Observed finite-bank capacity

All8motion1/2subjects already attain success with state selection. Each of
the4motion0subjects now has full actual execution for ALL8alternatives:

| Subject | Options0/1/2/3/4/5/6/7 |
| --- | --- |
|0|0/0/0/0/0/0/0/0|
|3|0/0/0/0/0/0/0/0|
|6|0/0/0/0/0/0/0/0|
|9|0/0/0/0/0/0/0/0|

Observed best among this finite bank is8/12, exactly attained by state and
all three oracle selectors. Remaining headroom is0on these subjects/options.
This is retrospective candidate capacity, NOT retrospective policy gain or
an upper bound over all continuous actions, controller classes, seeds or Cm.
It explains why this particular input comparison cannot expose positive
joint utility: the baseline selector already attains its observed bank limit.
It does not explain all earlier learners or reject hand-object information.

All129native panels are qualified (111inherited+18new). Independent SciPy
geometry/protocol audit covers84nonzero and12zero candidates, all45full-world
labels and all32motion0candidate labels. Every selected AND capacity branch
matches its retained short query through ALL68control/136physics frames,
whole-world maxstate/force error0. All105mesh labels,32capacity rows/counts
and bounded gates agree. New schema retains unqualified friction exclusion
and compressed contact moments; not full manifold or force-closure evidence.

New collector wall 683.810s, terminal new bytes 554139436 (before final CPUaudit),
within900s/2GiB. Independent final CPUaudit 26.506s. Protected source SHAs
unchanged at completion;41owned collector/native-audit PIDs absent when
checked. One idle GPU6 for physics/inference, CPU independent arithmetic.

Next decision: close this eight-option/single-preparation bank before more
Cm fitting. The next cheap exploration should change control class, e.g.
an object-frame whole-grasp primitive using a successful own native hand-
object relation, then test reachability. Such a desired relation is a control
target, distinct from the future-interaction oracle tested here. No increase
of Gaussian magnitude/seed/horizon/epochs is authorized by this result.
After useful candidate headroom exists, revisit matched effect/I information
and ultimately policy-training utility. Core mission/claim unchanged;
journal readiness NOT READY, no formal Validation or universal refutation.
