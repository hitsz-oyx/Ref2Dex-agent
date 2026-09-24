# P-20260924-cmv13-object-loo-representation

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe — frozen representation value.

## Question

On the fixed object-balanced e320 H10 support pool, do frozen tokens from the
verified mixed GRAB/Inspire ObjectInteractionCm V1.3 checkpoint encode
load-bearing outcome information that transfers through a newly fitted head
to a held-out object identity and depends on the proposed action?

Use 64 positive and 64 failure windows per identity, sampled deterministically
from the pinned sources in `P-20260924-stratified-e320-support`. Fit only small
heads in five object-LOO folds; the Cm checkpoint remains frozen. Compare:

1. calibrated action-aware dense tokens using the V1.3 Inspire interaction
   density (10135 sampled hand points, K=32, radius 2 cm);
2. the same token path with zero hand flow;
3. the same token path with within-object shuffled actions;
4. normalized raw q, dof velocity, object state and action;
5. a 1538-point action-aware runtime-density diagnostic.

The online hand-flow estimate is the pinned mixed-dose action+velocity linear
calibration, not future q and not the known-bad nominal PD endpoint.

## Fixed gate and decision

Proceed to a small matched PPO Cm-token on/off/placebo Probe only if the
10135-point action-aware representation has mean LOO ROC-AUC >=0.65, exceeds
both raw and action-blind means by >=0.05, exceeds action-shuffled by >=0.05,
and reaches AUC >=0.60 on at least four of five held-out objects. Otherwise
stop the current V1.3 representation route. The 1538-vs-10135 comparison is a
density diagnostic, not an extra route-selection gate.

One idle GPU, <=60 minutes, <=1 GB output. Stop on source/checkpoint/calibration
drift, incomplete episodes, non-finite features, fewer than 64 rows per class,
GPU conflict, or wall-budget overrun. This Probe may establish transferable
representation signal only; it cannot establish policy utility or causality.

Engineering note: the first smoke caught that the generic
`ObjectInteractionCm/extract.py` construction passes `cfg.model`, which drops
the task-level `meta` and silently defaults to K=8/radius 5 cm. This Probe
passes the full embedded config and asserts K=32/radius 2 cm. The failed smoke
is not scientific data.

## Result

The completed run is
`outputs/CmResidual/agent_cmv13_object_loo_s249/report.json` (640 balanced
windows, 128 per object). Mean object-LOO ROC-AUC was:

* action-aware, 10135 points: 0.6387 (3/5 folds >=0.60);
* action-aware, 1538 points: 0.6362;
* action-blind, 10135 points: 0.6694;
* action-shuffled, 10135 points: 0.6382;
* raw state/action: 0.5640.

The action-aware arm missed the absolute and 4/5 gates, was 0.0307 below the
action-blind arm, and exceeded the shuffled arm by only 0.0005. Increasing the
runtime hand surface from 1538 to 10135 points changed mean AUC by only
+0.0025. Mug and waterbottle action-aware folds were near chance (0.5150 and
0.5142). Sample-valid fractions were 0.922–0.984, so empty interactions do not
explain the result.

Classification: `UNPROMISING` for action-aware cross-object representation.
Stop the current V1.3 representation route and do not launch the matched PPO
Probe. The tokens contain transferable state/geometry information, but this
experiment supplies no evidence of incremental action information or policy
utility. The density diagnostic says 1538-point inference is already adequate
for this failed discrimination target; it does not prove density sufficiency
for every downstream task.

Scope clarification: object-LOO holds out each identity only while fitting the
small probe head. The frozen V1.3 checkpoint was pretrained using an index whose
train split already includes all five identities (airplane 17, cubesmall 11,
mug 25, toothpaste 6, waterbottle 11 sequences). This is a head-transfer
diagnostic on new physical transitions, not an end-to-end unseen-object test of
the Cm representation. The action-aware arm fails the fixed gate even under
this narrower setting.
