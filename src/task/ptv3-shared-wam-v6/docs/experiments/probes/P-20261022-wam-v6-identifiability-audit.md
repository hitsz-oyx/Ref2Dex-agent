---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-identifiability-audit
experiment_id: P-20261022-wam-v6-identifiability-audit
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-v6-identifiability-prefix-self-consistency
probe_index_in_family: 1
seed_pool: probe
seeds: [178,180]
decision_changed_if_positive: prioritize causally supplied contact intentions or predicted-object coupling if supported matched dispersion or consistency gaps justify them
decision_changed_if_negative: inspect supervision and generalization rather than assume more conditions or model scale are the blocker
status: PREPARING
run_id: identifiability-audit-20261022-r1
---

# Existing-checkpoint identifiability, prefix and hand/object consistency

Result: protocol/code prepared; numerical run pending.
Decision: read-only Decision Probe requested by owner at09:55; no training, model expansion, data additions, training control, credentials or remote push.

## Questions and cheapest discriminating evidence

1. Do independent GRAB recordings with similar observed H4 hand/object state,
same CAD, same supplied object-h24 endpoint goal and E_next type retain broad
GT futures? Does adding realized contact region, hand region and interaction
annotation reduce dispersion beyond label-shuffle controls? This can prioritize
causally supplied intent from a planner; it does not make future GT contact
causally available or prove intrinsic stochasticity.
2. Do healthy4371 versus existing flow8500 differ on training-admitted prefixes,
time-disjoint unadmitted prefixes and sequence-held-out prefixes? Report all
three horizon bands1–8/9–16/17–24 crossed with pre-contact/contact/post-release.
These are offline forecast errors, not proof of closed-loop controllability.
3. How much does replacing GT object with the model's own object prediction
change contact distance/clearance/penetration proxy for the same predicted hand,
GT tick and GT query mask? Keep GT-hand/GT-object as approximation calibration.

## Fixed inputs, sample selection and leakage controls

GRAB only,~99.8% of actual local training draws. Recipe
`outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json`
and its stats. No raw/prepared/manifest changes. Requested4096 TRAIN bank
windows, up to8 nonoverlapping H4+K24 supports/capture. Evaluation up to96
windows/panel, at most3/capture,28-frame separation; balanced coarse phase
selection. Seed178 selects panels/bank,180 capture bootstrap/permutation.
Model flow noise keeps existing fixed per-sample seed172,no best-of-N.

Raw native GRAB recording names must be disjoint across TRAIN/DEV. All frozen
TRAIN windows across every slot/alias mark inclusive `[t-3,t+24]` support.
An unadmitted same-sequence prefix is accepted only if its complete28-frame
support avoids every such interval. Report these as `heldout_prefix_local_recipe`:
remote ancestor admitted-window manifests are unavailable, so lifetime
pretraining nonexposure cannot be certified. Do not silently label them strict
held-out prefixes. Sequence holdout is checked against local frozen TRAIN and
native GRAB split provenance; remote full window lineage remains a limitation.

Bank requires complete valid futures for observed full hands and known E_next.
Pair only same CAD/presence/type, across different native captures. Gates20/50/
100mm on observed H4 hand,64-point object H4 and32-point supplied endpoint goal
Euclidean vector RMS;100mm is weak-match sensitivity. One minimum raw-condition
pair/capture-pair, no GT future used in base matching. Additional time-delta<=3
frames control avoids attributing an existing v6 input's omission to contact.

Dispersion=0.5*mean squared GT displacement-vector difference,mm²; separately
hand/object and prefix8/all24. Square root is RMS spread,not EPE. Contact region
is the canonical CAD's8 octants from GT contact-weighted centroid; hand region
is dominant palm/finger among16 cached parts. Add metadata interaction name
parsed from original GRAB recording. These future realized labels are oracle
upper-bound probes; separately report causal current-frame contact descriptors.
Coarse labels can miss multiple simultaneous contacts; a negative result does
not rule out richer intent. No Ref tensor/annotation is supplied; do not invent it.
200 two-endpoint capture-bootstrap replicates and100 label permutations within
original CAD/presence/E_type groups; no independent validation claim.

## Model and geometry protocol

Healthy source: `outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt`.
Existing FM8500 pinned by hardlink at
`outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/flow-step008500.pt`;
this protects audit provenance from routine recent-checkpoint retention without
changing training/source contents. No new training checkpoint is generated.
Strict FP32 evaluation for both,AMP/TF32 off; observed inputs/goal/event identical.
Optional initialization arm omitted to prioritize the requested two-checkpoint
comparison. Single-frame input is teacher/recorded observation, not self rollout.

Self-consistency uses all2048 ordered object queries and24 ticks. Extend model32
predicted points to dense CAD with the SAME observed-only affine fit used by
existing geometry diagnostics; never use GT motion in the predicted-object
reconstruction. Report affine residual to quantify missing nonrigid flow.
Positive GT queriesC>0.25: mean unsigned nearest transported-hand-skin distance,
3mm recall. All valid queries: geometric probability/GT contact MAE. Negatives
C<0.01: GT-reference clearance violation with original2mm tolerance. Predicted
hand skin uses the existing observed778-per-hand bone-frame transport.
Nearest oriented object-point signed depth/inside fraction at every12th hand
vertex is explicitly a penetration PROXY,not mesh SDF or a watertight inside
test. Evaluate GT-hand/GT-object with the same proxy to expose reconstruction
bias. Matching GT ticks/masks makes pred-object minus GT-object a paired gap;
it can be negative through correlated drift,so gap magnitude must be interpreted
alongside absolute GT EPE and contact. No physical success claim.

## Execution and boundaries

Owner authorizes GPU0 now, overriding the previous run-specific0–3 prohibition
for THIS diagnostic only. CUDA_VISIBLE_DEVICES=0 asserted in audit code; first
check09:56 GPU0=6MiB/0%. GPU4/5/6 existing FM RUNNING,step8860 at read-only
status check; never signal/pause/restart/modify it. GPU7 untouched. Environment
`/home2/wyy/miniconda3/envs/graspenv/bin/python`,2CPU threads,GPU budget40% of
one3090,batch2,3600s wall,3GiB artifacts including pinned checkpoint,20GiB disk
floor. All outputs/cache/temp under project. Detached nohup/new session launch,
environment allowlist,no credentials. Exact command,code and results appended
after run. Existing user `docs/PROJECT_MEMORY.md` remains untouched.
