# P-20260924-train5-causal-cm

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe / architecture versus data diversity.

## Question and design

Did the prior signed-action Cm causal head fail mainly because it saw
only three training object identities, or does its six-region
representation still fail when the fit pool grows to five? Collect
randomized ±0.1 wrist-z, five-step physical followups from the frozen
read-only official actor on the corrected train5 split, seed 186,
steps 50..200 stride 10; use official weights solely as a data
generator. Assign treatment independently within each simulator state;
inspect executed action dose and source hashes. `cubesmall` appears
twice in DExplore's hard-object sampler, so both motion IDs must be
grouped as one object in leave-one-object-out (LOO) evaluation.

Fit the existing signed geometric causal head, a raw state/action head,
and a constant train-object effect using identical data and bounded
updates. Primary gate before seeing alarmclock labels: geometric LOO
mean absolute object ATE error must beat both raw and constant by >=20%,
and sign must match the randomized object ATE on >=4/5 identities.
If it passes, collect/evaluate the same design on untouched alarmclock
as an exploratory held-out test. If it fails, do not attempt Cm PPO
with this representation; redesign its physical interaction features
or target. This is a single-seed Probe, not a proof of Cm usefulness.
One GPU for collection <=15 minutes, <=100 MB; CPU fit <=30 minutes.

## Initial result and architectural follow-up fixed before execution

The official-origin train5 randomized source completed: 1,024 rows,
balanced ± arm counts (507 each among treated records), with the
sixth motion ID correctly recognized as duplicated cubesmall.
LOO object ATE MAE was 21.36 mm for six-region geometric Cm,
15.46 mm raw state/action, and 15.55 mm constant; geometric signs
matched 4/5 objects but it overpredicted mug (59.3 vs 17.0 mm) and
underpredicted waterbottle (18.6 vs 44.0 mm). The predefined magnitude
gate **failed**, so no alarmclock intervention or Cm PPO on this
architecture. Source and report:
`outputs/CmResidual/agent_expert_train5_randomized_s186_h5/` and
`outputs/CmResidual/agent_train5_causal_head_loo_s186/`.

One minimal architecture check is worth doing on the *same frozen data*:
the six local-region tokens lack global object scale. Append a fixed
five-dimensional pre-action shape descriptor—mesh xyz extents (each
in 0.1 m units), log volume relative to 1e-4 m³, and log surface area
relative to 0.03 m²—to the causal head's state input. Keep the exact
LOO folds, randomized labels, 300 fit steps, and raw/constant controls.
If shape-conditioned geometry lowers object ATE MAE by >=20% versus
the previous geometric head **and** beats both raw and constant, retain
this representation for a later untouched-object test. Otherwise
drop this simple global-shape augmentation; do not tune descriptors
on LOO outcomes. Read-only meshes, CPU <=30 minutes, <10 MB.

### Shape augmentation result and final data-coverage check

The fixed five-number global descriptor reduced geometric LOO object
ATE MAE only from 21.36 to 19.99 mm (6.4%), still worse than raw
15.46 and constant 15.55 mm; gate failed. Thus missing simple global
mesh size is not the primary fix. No held-out alarmclock labels were
collected or used. Report:
`outputs/CmResidual/agent_train5_shape_causal_head_loo_s186/`.

The last bounded check for the *same* six-region representation is
data coverage: add three different preselected fit identities—cup
(hollow), phone (flat), duck (irregular), with stamp as fallback if one
fails the fixed >=16/64 official diagnostic held-lift and >=0.30
contact gate. Reuse the exact conversion pipeline, keep alarmclock
untouched, collect the same seed-186 ±0.1/H5 randomized transitions
on an 8-identity train split, and repeat object-level LOO. If the
geometric head still fails to beat both raw and constant by >=20%
object-ATE MAE, stop this architecture/data scaling route; changing
only more of the same data is unlikely to rescue the current Cm.
At most four short GPU screenings plus one collection; <=20 minutes
GPU and <1 GB new data. This remains a Probe, not a policy-utility
claim.

### Eight-object result

The three prespecified additional identities passed the unchanged
physical screen: cup 63/64, phone 20/64, duck 64/64 official diagnostic
held-lift (mean contact 0.818/0.398/0.903). Their corrected tensors
are finite and have contact-frame wrist/object separation 135–146 mm.
The object-disjoint train8/alarmclock split is pinned in
`outputs/CmResidual/agent_crossobject_pilot_split_v3/manifest.json`.
An independent seed-186 ±0.1/H5 official-origin randomized collection
completed with 1,007 treated rows across nine motion IDs, where the
ninth is duplicated cubesmall. Eight-object LOO object ATE MAE:
**geometric 14.09 mm, raw causal head 7.96 mm, constant 15.28 mm**.
The predeclared geometric >=20% advantage gate failed decisively;
mark this six-region architecture/data-scaling route `UNPROMISING`
for causal effect extrapolation and do not test it on alarmclock or PPO.
The raw causal head's lower LOO error is a new *different architecture*
signal, not proof of within-object action ranking or policy utility.
