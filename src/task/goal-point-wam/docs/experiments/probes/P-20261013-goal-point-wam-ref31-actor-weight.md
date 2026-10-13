---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-ref31-actor-weight
experiment_id: P-20261013-goal-point-wam-ref31-actor-weight
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: 51a4fa64
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 41
seed_pool: probe
seeds: [300, 301]
decision_changed_if_positive: retain association weighting for a longer matched human-pretraining comparison
decision_changed_if_negative: defer weighting expansion and prioritize robot own-state coverage
status: UNCLEAR
run_id: wam-ref31-actor-weight-20261013-r2
---

# Independent OakInk actor-weight short A/B

Result: Completed paired300 updates; target-interaction actor13.189->13.167mm
(0.169% improvement), below fixed5% signal gate; UNCLEAR.
Decision: Preserve implementation and evidence; no larger weighting run now.
Prioritize robot actor/executor compatibility and actual visited-state coverage.

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

## Engineering correction before completed comparison

r1 original300 finished, second arm stopped before its first optimizer update:
first-batch World differed under nondeterministic GPU reduction. One-update
smokes1/2 reproduced it with identical input/RNG and initial parameter identity;
smoke2 actor trajectory difference0, World translation/rotation up to.0084m/
.026 component, World loss relative3.1e-5. Enabling deterministic algorithms
and CUBLAS_WORKSPACE_CONFIG=:4096:8 made smoke3 all predictions/loss exactly
identical. r2 reruns both300-update arms with original strict loss1e-7 and
prediction1e-6 checks, identical input/RNG checks. No scientific gate relaxed.
Incomplete r1/smokes are engineering records, not paired research evidence.

## Completed independent A/B

Run wam-ref31-actor-weight-20261013-r2, commit51a4fa64,244.16s including
three evaluations,115/114s training per arm. GPU3 ~7GiB and50–70% utilization.
9600 identical draws, 1166 marked (12.146%).
First paired input/RNG/trajectory/translation/rotation/World loss exactly equal;
only marked actor terms downweighted. Independent original-sampler/census-mask
and saved384-window metric audit passes. No data removed or World term changed.
All original modules trainable, so learned shared World outputs may differ.

| Fixed VAL panel | n | Original actor mm | Downweight actor mm | Original World mm | Downweight World mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| oakink2 | 128 | 15.6652 | 15.6617 | 3.7522 | 3.7485 |
| grab | 128 | 74.7163 | 74.7147 | 21.8751 | 21.8196 |
| arctic | 128 | 54.3845 | 54.3829 | 30.2919 | 30.3069 |
| oakink2_mismatch | 31 | 19.3877 | 19.4030 | 2.6192 | 2.6164 |
| oakink2_aligned_proxy | 97 | 14.4755 | 14.4660 | 4.1142 | 4.1104 |
| oakink2_target_interaction | 81 | 13.1889 | 13.1666 | 4.6554 | 4.6509 |

All24 window-macro Euclidean actor error and h24 valid-object mean World EPE.
Target-interaction gain0.0223mm/0.169%, below5% fixed gate; first8 gain.00128mm.
Mismatch panel worsens.0152mm. Three-domain/World preservation passes, but
there is no useful positive scale-up signal. UNCLEAR because single short
warm-start with small panels cannot refute the data association hypothesis.
Do not promote either arm into robot C/adapter, do not delete static windows,
and do not rerun large pretraining from these tiny differences.

Evidence: outputs/goal-point-wam/wam-ref31-actor-weight-20261013-r2/
result.json, root_weight_audit.json, *_contract.json, *_curve.json,
train_indices.npy and *_val.npz; original.pt/downweight.pt preserved.
