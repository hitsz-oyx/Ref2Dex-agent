---
schema: ref2dex.probe.v2
probe_id: P-20261023-nine-source-degradation
experiment_id: P-20261023-nine-source-degradation
date: 2026-10-23
task: ptv3-shared-wam-v6
branch: main
git_commit: recorded-in-receipts
classification: Decision
status: HOLD
run_id: diagnosis-C1-C2-C3-C4-20261023
---

# Nine-source degradation mechanism controls

Owner requests CPU supervision audit first, then C1 -> C2 -> C3 -> C4.
The exact preregistration, receipts, original source scale data and progressive
results are on NAS at
`/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/`.
Entry: `README.md`; immutable intent: `preregistration.json`.

Decision: distinguish inherited target/input scales, source weighting, shared
backbone damage and optimizer continuation. Cheapest authorized discriminators:
256 TRAIN windows/source for scale estimates (exact converted receipt census),
then six bounded500-update/40-minute controls. No model or threshold changes.
All arms initialize only local3000 weights, fresh optimizer, inherited statistics,
A unknown goal/event contract, flow_lr1e-4/shared_lr1e-5/warmup50/clip1/wd.01.
C1-C3 world4/local96/global384; C4 single-source1/10/0 world1/local96/global96.
No automatic restart. OOM/nonfinite/loader/disk faults preserve terminal evidence.
Existing half-training and remaining data are read-only inputs.

C1 requires the exact original curated6000 support metadata; source weights alone
would duplicate C2. Do not replace missing C1 support with full-data weighting or
with the frozen512-panel training set. The manifest referenced by the panel is
`outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json`;
this file is absent from this checkout. A path request is pending.

Actual model contract is2hands x21 joints x24 steps in start-object axes, not
11 world-coordinate points. A reports first8 actual training and world-axis
motion separately; no11-joint mapping is invented. Position inputs use fixed
physical scaling, not TRAIN mean/std standardization. Effective supervision
fractions are sampled point-frame fractions over both hand slots; a single-hand
source can legitimately yield0.5. Raw-world/normalized moments do not alone
prove a causal normalization failure. These limitations accompany owner H2 screen.

Run commands (remote short checkout, graspenv):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/audit/diagnose_source_supervision.py
# Existing audit output is exclusive: do not rerun into its full-census directory.
# Once original support metadata is supplied on NAS:
PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/run/run_diagnosis.py --curated <NAS-original-curated-manifest.json>
```

Report raw signed EPE deltas and relative changes for C4, preserving the owner's
50% threshold. No future repair or additional baseline is launched in this run.

A completed in379.912s; H2 scale screen positive, causal normalization attribution unresolved. C1 missing original support metadata; all B arms0 updates. CPU recipe/sampler checks passed; report snapshot: [audit](../../audit/nine-source-degradation-20261023.md). Initial scripts/report commit93c5ebc2; per-arm code identity recorded in recipe.
