---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref31-actor-weight
experiment_id: P-20261013-goal-point-wam-ref31-actor-weight
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 81956abb
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 41
seed_pool: probe
seeds: [300, 301]
decision_changed_if_positive: retain association weighting for a longer matched human-pretraining comparison
decision_changed_if_negative: defer weighting expansion and prioritize robot own-state coverage
status: RUNNING
run_id: wam-ref31-actor-weight-20261013-r1
---

# Independent OakInk actor-weight short A/B

Result: Pending matched300-update warm-start comparison.
Decision: Keep this human-data intervention independent of Chunk-R repair.

Decision Probe serving task C3: does reducing supervision of a static selected
object while hands move near a different object improve target-associated actor
prediction? Cheapest discriminating test: same existing base WAM,300 updates
per arm, same original source/category sampler draws and stochastic seeds.
No windows removed, no goal replacement, no World supervision changes. Full
original model trainable in both arms; shared parameters can affect World, so
measure World retention rather than claiming its outputs are fixed.

Mismatch is the audited geometric proxy: OakInk only, endpoint<2mm, peak object
motion<5mm, hand peak>=5mm, target min current/future>100mm, other object future
near<=50mm (exclude target-approach stratum). This is not a verified semantic
annotation of manipulated-object identity. Lower marked window actor loss to.1;
retain original valid-point denominator, so all other actor terms stay weighted1.
World labels/loss weight1 unchanged, including every marked window.

Base: outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt.
Census: outputs/goal-point-wam/main3-curation-audit-20261013-r1.
Original vs downweight: AdamW1e-5,weight_decay.01,batch32,300 updates each,
bfloat16,clip1. Model/stochastic seed300,draw301; exact same initial parameters
and9600 sampled windows. Check initial paired World loss and all-one baseline
actor equivalence. Fixed first128 VAL rows per source from the already audited
panel, no selection by observed A/B errors. Report target-interaction, mismatch,
other OakInk and each domain, first8/all24 actor and h24 World EPE.
PROMISING signal: >=5% target-interaction actor improvement, <=5% each-domain
actor and World regression. Single fixed endpoint; no checkpoint selection.

Idle GPU3, paired budget2400s, <=2GB new artifacts, global400GB. Stop nonfinite,
input mismatch, foreign GPU process or deadline. Logs include utilization,
memory and ETA. No new branch, no large pretrain, no robot actor adaptation.

Limitations/future evidence: single seed, short warm-start, geometric proxy,
small held-out panel; null signal does not refute association quality. Actual
robot outcome/Cm utility and GRAB/ARCTIC far-to-contact admission changes are
outside this independent comparison and deferred.
