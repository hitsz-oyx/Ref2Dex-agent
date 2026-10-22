---
schema: ref2dex.probe.v2
probe_id: P-20261022-wam-v6-identifiability-audit
experiment_id: P-20261022-wam-v6-identifiability-audit
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: 7b32fde96caac093ddbf346c71c1b2f2459ce858
claim_id: C1
hypothesis_family: HF-v6-identifiability-prefix-self-consistency
probe_index_in_family: 1
seed_pool: probe
seeds: [178,180]
decision_changed_if_positive: prioritize causally supplied contact intentions or predicted-object coupling if supported matched dispersion or consistency gaps justify them
decision_changed_if_negative: inspect supervision and generalization rather than assume more conditions or model scale are the blocker
status: COMPLETED
run_id: identifiability-audit-20261022-r1
---

# Existing-checkpoint identifiability, prefix and hand/object consistency

Result: COMPLETED. FM8500 held-out prefix hand/object20.28/9.36→21.78/10.97mm despite improved C; strict conditional support insufficient; geometry gap negative and object rigidity worsens.
Decision: hand off numerical evidence for owner review; do not expand model/data or start training. Keep existing GPUs4/5/6 training untouched. Conditional identifiability and physical controllability remain unproven.

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

## Resource change and admission corrections

10:11 GPU0 was claimed by another user's VLN evaluation after our09:56 free
check. Initial all-stage run2548726 completed bank/variance,then safely failed
CUDA OOM while loading the first forecaster; no other process was signaled.
Do not reuse its incomplete evaluator output. Geometry-only identity/translation
smoke onGPU1 PASS: GT affine max1.18e-5mm,matched-anchor gap3.05e-5mm; a
17.320508mm object translation measured17.320501mm and leaves GT-anchor
distances bitwise unchanged. Smoke codef89b6a47.

GPU0 regained spare headroom; model evaluation under7223bfb9 uses batch1 and
1.25GiB allocator bound, saving partial records every8windows. It yields below
2GiB free. Healthy TRAIN and sequence panels finished; later local-prefix
processing yielded as another job's memory grew. Resume only on an authorized
0–3 card with>=3.5GiB free/util<=50%,one card at a time; never touch4/5/6/7 or
other jobs. Dedicated inference-only controller bounds retry/wait to30min.
Bank4096,panels96 each (TRAIN95 captures,sequence66,local-prefix93),seed178.
Raw GRAB TRAIN1040/DEV92 captures,zero overlap;26147 unadmitted disjoint local
prefix candidates across902TRAIN captures. Gate20/50/100mm yields15/197/1045
distinct-capture pairs; strict gate support is limited,especially after matching
E_next time within3frames (2/34/117pairs). Precise results will follow.

## Bounded mesh cross-check protocol

Once GPU0 is free, use first16 frozen sequence-panel windows (no result-based
selection),ticks8/24,32fixed vertices/hand. Existing watertight GRAB CAD triangles
with CUDA point-triangle distance and two generic-ray parity directions; require
ray agreement,report>1mm inside fraction/depth and GT-hand-proxy calibration.
Predicted CAD is the observed-only affine32 extension; GT uses recorded poses.
Skip/report nonwatertight or CAD/query alignment>1mm. Analytic unit-cube control
requires two known inside points and600mm summed depth. Bounds15min,~3.6GiB
allocator; inference only, no optimizer/capacity/data changes. This distinguishes
point-normal proxy artifacts from true inside tests on the transported hand
proxy; it is still not penetration of the recorded future hand mesh.

## Group and presence controls

The raw pair reduction is descriptive,not a population-controlled variance
decomposition: the matched subset changes. Explicitly separate future-contact
PRESENCE from REGION. In the50mm bank,requiring both futures to contain contact
already changes197pairs/3030.8hand8mm² to188pairs/2029.2;same-region66pairs
yields2033.7. Thus the apparent33% reduction is not evidence for regional intent.

Add a proper complete-link grouping check on the same cached bank:no new
window extraction or network inference. Same CAD,presence,Etype,all-pair H4/goal
gate;4–8distinct captures/group,each capture used once/gate. Refine group
partitions by future-contact presence,then region/finger/interaction;pool
unbiased within-partition variances with explicit df,singletons excluded rather
than counted as zero.100 within-original-group label permutations and200group
bootstraps;fewer than8groups receives no bootstrap confidence claim. Report
20/50/100mm plus event-time3 controls;insufficient support remains inconclusive.
Mesh review also reports best-SE3 fit residual/pair-distance distortion for
model32 points,same16windows/ticks,with GT rigid residual calibration. No new
model or fitting of learned parameters is performed by these geometry fits.


## Completed numerical results,10:46 +08

All numbers below are existing-weight inference or frozen-label statistics.
Execution complete; scientific identifiability decision remains UNCLEAR. Four
targeted tests PASS,real geometry identity/translation and mesh/cube controls PASS.
Both checkpoint models finite;four normalizer buffers exactly equal. Main
successful resumed evaluator97.12s/peak tensor0.457GiB;reviewed mesh~24s/0.434GiB.
Resource waiting/OOM/yield logs are retained;only owned audit processes yielded.
Existing flow training still RUNNING9700/max43200 at final read-only status check.
No new training,model expansion,training-data mutation,other-job signal,credentials
or remote Git action. Final free space149.46GiB;output1.4GiB+tmp184MiB<3GiB.

### 1. Proper grouped conditional variance

Complete-link50mm primary:5groups,30distinct captures;low support.20mm:
1group/4captures;Etime3 control:0groups at20mm,1group/4captures at50mm.
Thus the sample cannot certify exact full-condition identifiability. Labels are
realized oracle descriptors;no planned causal intention variable was tested.

|Refinement|windows contributing/df|hand8 mm²|hand24 mm²|object8 mm²|object24 mm²|change vs previous hand8/hand24/object8/object24 %|
|---|---|---:|---:|---:|---:|---|
|base|30/25|1609.33|30650.48|995.40|1379.39|—|
|plus_future_contact_presence|30/25|1609.33|30650.48|995.40|1379.39|+0.00 / +0.00 / +0.00 / +0.00|
|plus_object_region|26/18|1927.43|34286.62|1181.74|1657.17|-19.77 / -11.86 / -18.72 / -20.14|
|plus_finger|20/13|2190.48|44201.04|1047.22|1469.11|-13.65 / -28.92 / +11.38 / +11.35|
|plus_interaction|16/10|2663.34|42515.16|1345.44|1876.20|-21.59 / +3.81 / -28.48 / -27.71|

Positive percent means variance reduction;negative means increase.50mm actual
region/finger/interaction results remain within corresponding within-group
permutation intervals;with5groups no bootstrap precision claim is made.
At100mm weaker matching:30groups/150captures;after contact-presence control,
hand24 variance43293→25134mm² after region (−41.94%,group-bootstrap reduction
95%17.12–65.32).Hand8 reduction34.22% has CI−1.53–61.77%.This is a coarse
oracle-information signal,not proof of missing causal input at an exact state.
100mm plus Etime3 leaves only1group/7captures. No Ref annotation available.

Paired-dispersion controls (different estimator and population from grouped table):
4096nonoverlapping TRAIN bank windows.20/50/100mm15/197/1045 distinct-capture
pairs.50mm raw hand8 variance3030.77;both futures contain contact188pairs gives
2029.22;matching actual region66pairs gives2033.67 (+0.22%,no regional gain).
The original197→66 comparison gave32.90% apparent reduction,95%CI−24.95–70.14,
largely explained by contact-presence exclusion. Do not attribute it to region.
Matching already available E_next timing within3frames retains34pairs and
changes hand8 variance3030.77→1443.21/object8 757.95→107.83mm². This shows
why a type-only grouping cannot diagnose the actual full v6 input contract.
Pair raw condition means/quantiles/null bands/CIs are retained in JSON;no
silent clipping of high-motion targets. Future contact octants/finger labels
are coarse;null findings cannot exclude richer planned waypoint sequences.

### 2. Forecast errors and phase stratification

Every panel96windows,no overlapping28-frame supports within raw captures.
TRAIN95captures,heldout_sequence66,local-recipe-uncovered93. Complete TRAIN
has1040GRAB captures,DEV92;zero raw recording overlap. Current local recipe
has26147 eligible unadmitted disjoint prefix candidates across902TRAIN captures.
Lifetime nonexposure to remote ancestors is UNKNOWN for unadmitted prefixes.
Sequence holdout is relative to verified native/local TRAIN split,not a new TEST.
DEV has been repeatedly inspected;this is a Probe,not formal validation.

Each cell below is healthy hand/object → flow hand/object EPE,mm:

|Panel (96 windows)|t+1–8|t+9–16|t+17–24|
|---|---|---|---|
|train_prefix|10.420/4.550 → 9.903/4.973|28.290/8.991 → 24.562/11.337|46.036/11.239 → 37.836/15.735|
|heldout_sequence|20.277/9.361 → 21.784/10.971|92.691/25.114 → 95.652/29.218|162.516/17.632 → 165.863/23.512|
|heldout_prefix_local_recipe|40.638/2.801 → 42.340/5.548|181.012/4.832 → 187.657/11.815|319.947/5.983 → 326.372/15.858|

Heldout-sequence prefix8 paired capture-bootstrap FM-minus-healthy deltas:
hand+1.507mm[0.892,2.072],object+1.611mm[1.078,2.118]. TRAIN hand−0.517mm
[−0.953,−0.116],object+0.423mm[−0.053,0.865]. Local-uncovered prefix hand
+1.702mm[0.744,2.604],object+2.747mm[2.341,3.212]. No held-out prefix gain.
These errors evaluate forecasts from recorded H4;physical controllability and
24/8 self-feedback execution are not measured in this audit.

Prefix8 phase breakdown;windows can span multiple phases,counts are not additive.
Phase derives from cached hysteresis state/transitions on each realized GT tick.

|Panel|phase|windows/captures|frame count|healthy hand/object mm|flow hand/object mm|
|---|---|---|---:|---|---|
|train_prefix|pre_contact|35/35|265|10.386/3.002|9.982/4.059|
|train_prefix|contact|56/56|379|8.970/6.481|8.972/5.806|
|train_prefix|post_release|26/26|124|14.926/1.953|12.584/4.379|
|heldout_sequence|pre_contact|8/8|47|14.642/3.686|15.817/4.378|
|heldout_sequence|contact|72/55|499|13.329/13.366|14.769/14.626|
|heldout_sequence|post_release|35/34|222|37.089/1.559|38.816/4.153|
|heldout_prefix_local_recipe|pre_contact|46/45|358|41.945/0.892|43.482/3.567|
|heldout_prefix_local_recipe|contact|9/8|60|14.104/23.923|15.808/23.550|
|heldout_prefix_local_recipe|post_release|44/44|350|43.850/1.133|45.719/4.490|

All27horizon×phase combinations and global phase24 results are in summary.json/
summary.md and per-model panel records. Each row includes sums/counts plus
windows/captures/frames. Heldout prefix C-head MAE0.055457→0.049339,95%delta
[−0.009842,−0.002393]. Its GT-object geometric contact MAE0.068262→0.069415,
so improved learned C does not imply improved predicted hand geometry.
Do not assign this difference solely to the FM algorithm:baseline and flow
differ in additional updates,new branch and the newer geometric objective.

### 3. Hand versus GT/predicted object: paired geometry

Heldout96windows/66captures,prefix8. Contact-mask C_GT>0.25;negativesC_GT<0.01.
All positive-query distances use the same GT frames/query IDs. The soft positive
mask includes proximity out to~17mm;GT-hand/GT-object distance is8.9998mm,
so absolute contact distance is not expected to be3mm. Clearance is the existing
GT-direction-referenced loss diagnostic with2mm tolerance and33mm margin cap,
not a closed-mesh physical clearance certificate.

|Metric|query/vertex-frame count|healthy GTobj / predobj|flow GTobj / predobj|flow pred−GT gap|
|---|---:|---|---|---:|
|contact_distance_mm|304885|15.024737 / 10.477000|15.534063 / 10.629932|-4.904131|
|geometric_contact_mae|1572864|0.068262 / 0.056603|0.069415 / 0.057557|-0.011858|
|clearance_violation_mm|937846|0.676419 / 0.696735|0.705195 / 0.693073|-0.012121|
|oriented_penetration_proxy_mm|99840|0.112203 / 0.089904|0.110538 / 0.071140|-0.039398|
|oriented_inside_proxy_fraction|99840|0.018079 / 0.014403|0.017508 / 0.014353|-0.003155|

Contact-distance gap healthy−4.548mm[−9.653,−1.242],flow−4.904mm
[−10.351,−1.410]. Flow clearance gap−0.012mm[−0.183,0.170]. It is not a large
positive separation penalty in these samples. Negative gaps cannot certify
accurate deployment:common motion errors,nearest-any-hand correspondences and
object deformation can hide absolute errors. No sole-cause/co-drift claim.

Dense predicted object is observed-CAD affine extension of model32 points.
Prefix8 non-affine residual mean0.587→3.193mm over24576point-frames,so this
reconstruction smooths some FM nonrigidity. The point-normal penetration values
above are proxies and need the mesh cross-check below.

Watertight CAD cross-check:first16fixed sequence windows/15captures,t8 andt24,
64hand vertices/window (32/hand),1024vertex-frames/model/anchor/time. All CADs
watertight/aligned;no exclusions or two-ray disagreements. In every model/
GTobj-or-predobj anchor/time,0/1024 points show>1mm inside penetration. This
sparse negative finding does not prove full-surface or continuous collision
freedom. HAND is still transported observed skin,not recorded future mesh.
Unit-cube known inside/depth PASS (2inside,total600mm);every actual CAD crossing
positive-control fraction1.0;GT best-rigid residual max0.0000313mm.

Object32 rigidity residuals on same16windows/15captures (best SE3 fit,mm):

|time|healthy rigid residual|flow rigid residual|healthy/flow pair-distance distortion|
|---|---:|---:|---|
|8|4.601|7.827|4.737 / 6.552|
|24|5.706|15.352|5.165 / 12.393|

This is direct evidence of increased nonrigid object forecast error in the
current package,not proof of why it happened. Keep rigid readout/consistency
and supervision/condition-use questions for the owner’s next decision;no new
training or architecture change is authorized or launched by this audit.

## Artifacts and exact reproduction

Root: `outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/`.

- `panels.json`: exact immutable triples,recording provenance,phase masks,bank rows.
- `bank.npz`/`bank.json`: compact GT arrays and conditions,4096TRAIN windows.
- `identifiability.json`/`pairs-{20,50,100}mm.npz`: pairs,variance/null/bootstrap.
- `group-identifiability.json`: proper complete-link groups/df/null/controls.
- `contact-presence-sensitivity.json`: reproducible contact-occurrence confound.
- `{healthy4371,flow8500}-{train_prefix,heldout_sequence,heldout_prefix_local_recipe}.json`: per-window and aggregate crossed metrics.
- `evaluation.json`,`summary.json`,`summary.md`: full machine/human handoff tables.
- `geometry-smoke.json`,`checkpoint-audit.json`,`mesh-penetration.json`: controls and mesh/rigidity records.
- `provenance.json`,`resource-controller.json`: exact commands,code,PIDs,resource yields.
- `tmp/identifiability-audit-20261022-r1/launch*.json`: actual frozen command arrays.
- `tmp/identifiability-audit-20261022-r1/{audit,evaluate,evaluate-resume-01,mesh-reviewed,groups-cached}.log`: stage stdout/stderr.

Implementation:bank/pairsf44f243c,evaluator7223bfb9,group/mesh review322e5adb,
table generator7b32fde9. Frozen paths and launch receipt commits identify actual
code;environment.git_commit may be the working-tree HEAD at subprocess entry.

Actual successful evaluator command (CUDA_VISIBLE_DEVICES=0,new session,closed stdin):

```sh
nohup /home2/wyy/miniconda3/envs/graspenv/bin/python -u /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/identifiability-audit-20261022-r1/eval-code/src/task/ptv3-shared-wam-v6/tools/audit/audit_identifiability.py --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --baseline /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt --flow /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/flow-step008500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1 --seed 178 --bank-samples 4096 --panel-samples 96 --batch-size 1 --phase evaluate --memory-gib 1.25
```

Actual reviewed mesh command (CUDA_VISIBLE_DEVICES=0,same detachment):

```sh
nohup /home2/wyy/miniconda3/envs/graspenv/bin/python -u /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/identifiability-audit-20261022-r1/mesh-reviewed/src/task/ptv3-shared-wam-v6/tools/audit/audit_mesh_penetration.py --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --baseline /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt --flow /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/flow-step008500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1 --mesh-root /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/data/raw_data/GRAB/tools/object_meshes/contact_meshes --samples 16
```

CPU group and table recomputation (no network inference/no training):

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/group_identifiability.py --output outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1
python3 src/task/ptv3-shared-wam-v6/tools/audit/summarize_identifiability.py --output outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1
```

For a full fresh rerun,use audit_identifiability.py --phase all --seed178
--bank-samples4096 --panel-samples96 --batch-size1 --memory-gib1.25 with
the exact manifest/stats/baseline/pinned-flow args above and a NEW output
directory;then run group_identifiability.py,audit_mesh_penetration.py --samples16
and summarize_identifiability.py on that directory. Flags have spaces before
their values as shown in the exact commands. Do not overwrite current evidence.

No push performed. Existing user docs/PROJECT_MEMORY.md is untouched.

### Copyable full fresh-output reproduction

The following commands are recorded for the owner,NOT launched here. They use
the same existing checkpoints/seed and a fresh output directory. Machine command
arrays also in `reproduce-commands.json`. Ensure GPU0 is free before replay.

```sh
env CUDA_VISIBLE_DEVICES=0 TMPDIR=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/identifiability-audit-20261022-r1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 REF2DEX_RUN_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent REF2DEX_V6_V5_CODE_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent REF2DEX_V5_BASE_CODE_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent /home2/wyy/miniconda3/envs/graspenv/bin/python -u /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/audit/audit_identifiability.py --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --baseline /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt --flow /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/flow-step008500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-recompute-20261022-r2 --seed 178 --bank-samples 4096 --panel-samples 96 --batch-size 1 --phase all --memory-gib 1.25

/home2/wyy/miniconda3/envs/graspenv/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/audit/group_identifiability.py --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-recompute-20261022-r2

env CUDA_VISIBLE_DEVICES=0 TMPDIR=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/identifiability-audit-20261022-r1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 REF2DEX_RUN_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent REF2DEX_V6_V5_CODE_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent REF2DEX_V5_BASE_CODE_ROOT=/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent /home2/wyy/miniconda3/envs/graspenv/bin/python -u /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/audit/audit_mesh_penetration.py --manifest /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json --stats /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json --baseline /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt --flow /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-20261022-r1/flow-step008500.pt --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-recompute-20261022-r2 --mesh-root /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/data/raw_data/GRAB/tools/object_meshes/contact_meshes --samples 16

/home2/wyy/miniconda3/envs/graspenv/bin/python /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/ptv3-shared-wam-v6/tools/audit/summarize_identifiability.py --output /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/ptv3-shared-wam-v6/identifiability-audit-recompute-20261022-r2
```

Global `tools/verify.py --changed` still reports pre-existing missing v2 metadata
in six trajectory-policy cards;no new-card error reported. Unrelated cards are
untouched. This does not replace the4 passing targeted tests and real-input
numerical controls described above.
