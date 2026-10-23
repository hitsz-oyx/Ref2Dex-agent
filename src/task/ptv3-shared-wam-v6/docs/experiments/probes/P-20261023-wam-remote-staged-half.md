---
schema: ref2dex.probe.v2
probe_id: P-20261023-wam-remote-staged-half
experiment_id: P-20261023-wam-remote-staged-half
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-run-manifest
classification: Decision
status: RUNNING
run_id: remote-half-20261023-r1
claim_id: expanded-nine-source-unknown-condition-training
hypothesis_family: source-stratified-half-full-expansion
probe_index_in_family: 1
seed_pool: probe
seeds: [170, 171, 172]
decision_changed_if_positive: continue investing in expanded-source unknown-condition prediction
decision_changed_if_negative: diagnose expanded-source optimization before additional scale or execution claims
---

# Remote source-stratified half → full, 2026-10-23

Owner explicitly moved conversion to the four-4090 host, accepted hours of NAS
processing, and requested training after the first source-stratified half passes
eight gates. Remaining conversion starts only after a real half-training update.
No local or remote Git push in this round. Code reaches the short remote checkout
through a local Git bundle and remote `git pull --ff-only <NAS-bundle> HEAD`.

**Decision question:** Does expanding the source/mother support improve the fixed
unknown-goal/event prediction panel while training loss decreases? The cheapest
authorized discriminator is a staged half-data fit, preserving its checkpoint and
panel before separately identified full-data expansion. This is a Probe, not a
held-out generalization or execution validation.

## Frozen selection and budget

Each source × inherited split independently selects `ceil(N/2)` native mother
URIs in SHA256(`20261023/source/resolved-native-mother-URI`) order. Relative episode
names are not unique across replay roots; the full native URI defines the mother.
TRAIN/DEV labels are inherited without reallocation. The second stage is the
strict complement; no object identity affects selection or admission thresholds.

| source | half TRAIN mothers / baseline windows | half DEV mothers / baseline windows |
|---|---:|---:|
| 0 | 249 / 2,512,052 | 35 / 546,069 |
| 1 | 521 / 146,145 | 46 / 13,636 |
| 2 | 134 / 187,502 | 17 / 25,666 |
| 4 | 13,224 / 965,352 | 3,304 / 241,192 |
| 5 | 910 / 116,685 | 244 / 31,336 |
| 6 | 114 / 299,126 | 30 / 68,395 |
| 8 | 50 / 1,474 | 0 / 0 |
| 9 | 215 / 20,654 | 47 / 5,414 |
| 10 | 501 / 237,354 | 125 / 59,040 |
| total | 15,918 / 4,486,344 | 3,848 / 990,748 |

19,766 half mothers. These window counts are frozen local census references;
remote admission and actual ordered tuple payloads must be independently equal.
NPZ container timestamps are not used to assert cross-host tuple identity.

All outputs live under
`/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/`.
Fixed half conversion: **93,649,118,672 B**; complement: **94,915,949,899 B**.
Both freshly reconstruct all selected remote mothers, including the 1,177 that
the local historical conversion reused; hence the total differs from local
182,331,193,978 B. No old data is deleted or overwritten.

8 converter processes, one CPU thread each, GPU0 for accelerated reconstruction,
64 mothers/batch. Initial GPU check: all four GPUs no compute processes, load0.32;
NAS free51,657,441,280,000 B. Based on the owner's existing12.2MB/s observation,
half write projection7,676s=2.13h, provisionally2–2.5h; this is not a measured ETA.
Complete data/metadata/audit/staging/training budget412,405,067,451 B; NAS floor30GiB.
Each conversion stage bounded6h; every batch preserves receipts; explicit manual
resume reuses checked completed sequences. A failure never automatically restarts.

## Eight gates and scope

1. Reopen every shard and production-reader index, reconcile all tuples/counts.
2. Native mother URI, hand payload and output path split isolation;288 source/split
   windows checked against SciPy SE3 and recorded targets; brute-force surface
   distance/contact/part recomputation at three frames/window.
3. SHA-bound plan, tuple reference, census, recipe, stats and panel tensors;
   conversion actual/expected bytes equal;1.5% source-stratified full-file hashes
   with saved array payload identity revalidation.
4. NAS capacity and30GiB floor; retained checkpoint ceiling136GiB; all runtime,
   caches and temporary outputs target NAS through the short checkout link.
5. Complete phase × numerical geometry × source matrix, including zeros.
   **source8 DEV remains zero.** The current owner's primary/secondary criteria
   use the fixed TRAIN panel, not nine-source DEV generalization. An initial
   optional question asked whether stricter official DEV restoration was desired;
   no new owner reply is claimed. The agent records the limited nine-TRAIN/eight-DEV
   scope explicitly and proceeds under the current autonomous training request.
   The technical gates remain required; no TRAIN is reassigned to DEV.
6. Per-source/split legacy goal and intentional event certificates both zero;
   raw NONE/CONTACT/RELEASE/UNKNOWN labels counted separately. Inputs remain
   canonical unknown goal and event. The96 independently certified new source4
   rollouts are not silently merged into this experiment.
7. Every selected remote tuple payload compared with the frozen local parent;
   NAS HOT3D native segment arrays rechecked for crossing in every admitted window.
   Original raw NPZ byte identity across hosts is explicitly not claimed.
8. One production four-rank optimizer smoke, local96/global384, reader→collate→
   actual PTv3/FM/GT2048 supervision→finite gradients. Separate smoke artifacts;
   smoke weights never initialize research training.

## Training and pre-registration

Initial weights: actual local `tmp/t38-r5/training/step003000.pt`, model-only
export; preserve its original TRAIN-only normalizer. New optimizer/step for the
half recipe; source local3000 remains the paired comparison. Existing model,
strict FP32, unknown goal/event, loss weights and GT2048 geometry remain intact.
Remote Torch2.0.1+cu118/graspenv requires attention API compatibility; code retains
FP32 and disables the older fused inference path without changing checkpoint keys.
Its efficient kernel rejects every non-null attention mask. Removing exactly the
invalid keys, grouped by valid key count, passes the existing ragged and new
arbitrary-hole output/gradient tests against independent math attention. Remote
CUDA regression:12 passed. No environment upgrade or half-precision fallback.

Measured engineering model preflight:101,353,749 parameters, all predictions
finite. Efficient Torch2.0 inference repeats differ in last bits; the fixed panel
therefore explicitly uses **FP32 math attention** (training remains efficient).
The same real model/input repeated under math attention is bit-exact for both hand
and object, max difference0mm, peak allocated665,339,392B. This is a backend
compatibility choice, not a model-quality improvement. Torch2.0 warns even on
unweighted integer CUDA bincount under deterministic-algorithm mode; that mode
uses warn-only there, with actual repeated panel predictions checked separately.
The existing NAS PointWorld dependency is linked read-only into the empty short
checkout submodule; its `ptv3.py` matches the local file SHA. No shared vendor code
or conda environment is modified.

World4 GPUs0/1/2/3, local96/global384,2 spawn workers/rank. Uniform source1/9,
then uniform mother, then uniform observed known phase, then uniform window;
at most8 draws/mother/rank/batch. Unknown phase used only if a mother has no known
phase. Source/mother support changes, so this is not the old6000-window recipe.
LR geometry/shared1e-5, flow1e-4,50-update warmup; original loss weights unchanged.

Half: up to3,000 new updates or6h. Full: explicitly restore the half model,
optimizer, step and four-rank RNG, add the complementary mothers, target remote
step40,000 or42h for the full stage. Source probability stays1/9; within-source
mother/window probabilities change and are declared in the transition receipt.
Half/full have different census/recipe/data identities and output directories.
All500-step checkpoints and panels remain; never combine these stages into one
unchanged recipe identity. Training wall budget at most48h across the two stages.

Fixed panel: same512 rows/124 mothers, exact original collated local tensors and
sample IDs, H4/K24, contact label threshold>0.25. The panel is TRAIN evidence.
Three EPEs: first8 hand, fixed contact-window hand, first8 object; counts
156,135 /288,960 /131,072.4000 paired mother-cluster bootstraps, seed170.

**Primary pass:** all three EPE means improve versus same-runtime half initial
weights and each paired cluster95%CI upper endpoint<0; successive complete
500-update mean-loss blocks are monotonically nonincreasing. Individual random
minibatch losses are not required to decrease every update. Improvement in only
some metrics is reported as partial; a nonmonotone loss block fails that criterion.
Panel at initial and every500 updates/final; full initialization has its own panel.

**Secondary:** same panel against actual local3000, report remote-minus-local mm,
paired cluster95%CI and winner/uncertain for every metric. Measured local3000:
**8.980819 /24.659350 /5.639402 mm**; respective cluster95%CI
[8.534392,9.407645] /[23.474065,25.768982] /[5.358705,5.909570].
Cross-Torch/GPU repeat differences are measured before research training; no
claim that environment differences are a dataset effect.

OOM, nonfinite loss/gradient, loader error, disk floor, artifact ceiling or
1800s training stall stops owned process groups, preserves evidence, and does
not restart. Monitor GPU utilization/memory and measured update ETA.

## Evidence and reproduction

Local transfer/budget evidence:
`outputs/ptv3-shared-wam-v6/remote-half-20261023-r1/`.
Remote plan/run commits and environments, `half/batches/`, `remaining/batches/`,
`*-census/manifest.json`, `*-gates/`, `*-recipe/`, `*-training/`, watchdog and
terminal receipts live under the NAS root above. All status facts must distinguish
measured, sampled, previous self-reported and projected results.

```bash
cd /home/wbcd/workspace/oyx_ws/Ref2Dex-agent
# Already running; never duplicate the one-shot campaign.
PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u \
 src/task/ptv3-shared-wam-v6/tools/run/remote_staged_data.py \
 --output /mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending \
 --mode half
# The controller has an exclusive one-shot lock and records each exact command.
PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u \
 src/task/ptv3-shared-wam-v6/tools/run/remote_staged_campaign.py \
 --output /mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending
```
