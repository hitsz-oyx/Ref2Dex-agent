---
schema: ref2dex.probe.v2
probe_id: P-20261015-mse-multimodality-audit
experiment_id: P-20261015-mse-multimodality-audit
date: 2026-10-15
task: ptv3-shared-wam-v3
branch: main
git_commit: b3e6d009
claim_id: C1
hypothesis_family: HF-mse-conditional-multimodality
probe_index_in_family: 1
seed_pool: probe
seeds: [214]
decision_changed_if_positive: prioritize a matched frozen-backbone conditional FM versus refitted MSE head probe
decision_changed_if_negative: keep deterministic forecasting for now or improve conditional retrieval support before an FM investment
status: UNCLEAR
run_id: mse-multimodality-audit-20261015-r2
---

# Existing-weight conditional multimodality audit

Decision Probe for the MISSION short hand/object transition representation; follows user-supplied `docs/user/mse和flowmatching的选择.md`. Question: do similar observed H4 and supplied object-h24 goal G exhibit multiple supported future hand trajectories, and does the trained deterministic forecaster lie between those modes? Existing-weight diagnosis cannot establish that a newly trained FM would outperform MSE. FM itself regresses a conditional velocity field with squared loss; compare distributional versus deterministic prediction, not two mutually exclusive names for losses ([original FM](https://arxiv.org/abs/2210.02747)).

Use immutable model-only snapshot of the current remote strict recipe's best checkpoint (export captured step16000); current training keeps running. Earlier step8000 can be an undertraining sensitivity check after primary metrics, not an independent seed. The new DexYCB expanded recipe is excluded from primary analysis because the current weights were not trained on its revised data/statistics. Only TRAIN supplies retrieval futures; DEV supplies queries, not density/normalizer fitting. No heldout TEST is opened.

Bounded target: 128 DEV queries per actual DEV source (8sources,1024 total), up to2048 unique TRAIN bank windows/source for0/1/2/5/6/9. UniDex4 and HOI4D10 have disjoint TRAIN/DEV target geometries (1653/413 and229/57; intersections0), so their DEV diagnostics are retained but unusable TRAIN banks are omitted. Retain one fully future-observed11-point hand and identical hand-presence pattern; report all omissions. Source/target exact geometry and scene-object identities separate incompatible coordinate/hand regimes. Retrieval uses frozen masked pre-head shared z (scene/hand/goal summaries), plus raw-condition retrieval control. Never use future hand labels for retrieval. Check actual H4 hand history, current scene history and supplied goal in metres. Gates20mm and50mm; record actual distances and unsupported queries, prefer stricter results. One neighbor per capture, avoid overlapped windows masquerading as separate modes. k32 preferred, minimum8 distinct captures; do not force clusters on unsupported conditions.

Metrics: conditional future spread; prediction→TRAIN conditional mean, nearest real future, cluster centroid/medoid; prediction-vs-GT empirical3-NN future distance; first8/all24/h24 ADE, amplitude and adjacent-step speed/acceleration attenuation; point-cloud unsigned fingertip proximity proxy at8/24 (not signed mesh contact or physical grasp success). Primary comparison uses absolute hand positions in the same target-current anchor; displacement-space sensitivity checks avoid confusing different start poses with different dynamics. Multimodal candidates require >=8 distinct captures, two clusters each>=20% and>=3members, diagonal-GMM BIC improvement>=10, silhouette>=.5 and trajectory-centroid separation>=20mm. GMM is exploratory, not proof of conditional modes at an exact continuous input. Positive/negative synthetic controls test whether the metric can distinguish a between-mode mean from real mode samples and ordinary unimodal spread.

If >=30 supported distinct DEV captures show robust multi-mode evidence and >=20% show between-mode/low-density prediction (mean closer than modes by>=5mm, predicted empirical NN distance>1.5xGT), prioritize a small matched FM head probe. If conditional support fails or effects vanish under tight/within-source/raw-retrieval checks, mark UNCLEAR rather than equating finite-radius variation with intrinsic stochasticity. Ratios use GT motion floors, static cases are reported separately; mean predictions are positive mechanism controls, retrieved actual futures are executable-data controls. These are decision thresholds, not a validation claim or an architecture switch.

## Resource and safety Decision Note

Local0..5 occupied by existing three actor runs,6/7 must remain free per current CAMPAIGN. Remote4-card pretraining is this session's owned run. Use only its GPU0 for bounded no-grad inference (batch1/8 smokes passed; batch32 promotion only after memory and output-comparison smoke), allocator cap~3GiB, preflight free>=8GiB; stop audit on free<4GiB, OOM/nonfinite, own STOP,1h elapsed or2GB artifacts. Record GPU memory/utilization and parent training throughput; stop diagnostic if sustained interference or unsafe headroom appears. This colocates only with our own identified training process, never other/unknown jobs; no process is stopped or model/data/checkpoint overwritten. First small engineering smoke measures memory/rate before full bank. CPU loads/statistics use2threads. First batch8 smoke:144 windows in39.2s, peak775.7MiB, reserved1000MiB, parent1.425updates/s. Initial batch8 full cache deliberately stopped after121s to promote throughput; six completed DEV source archives retained for exact-input reuse, not a model failure. The replacement run is r2 with same frozen step16000/seed/recipe and prior-source reuse; combined cache GPU time remains within1h. New checkpoint exports<1GB, cache/results<2GB, free disk>=20GiB. No new long identity or additional training budget requested.

Future evidence: exact matched-condition repeated trials, mesh contact/SDF/penetration, calibration/mode reliability, trained FM vs MSE with matched data/backbone/compute, minADE@M vs single-sample/precision/coverage and cost, formal multi-seed validation and robot utility. Do not infer these from the current diagnostic.

Independent read-only implementation review: cache pooling, future-label invariance, four condition/retrieval controls and capture aggregation passed; three metric tests passed. Reviewer synthetic checks exclude starting-pose-only two-cluster artifacts and show duplicate windows do not change capture bootstrap. Bootstrap is conditional on this fixed TRAIN bank; input history position gates can still admit velocity/acceleration differences.

Batch32 promotion smoke passed:144 windows,41.8s including setup,9.06 TRAIN windows/s, peak1272.4MiB/reserved1504MiB; parent step18060 returned1.447updates/s after periodic step18000 checkpoint/eval interval. Identical sampled rows compared with batch8: mean per-point prediction difference0.115mm TRAIN/0.157mm DEV, max4.60/4.09mm. Near-threshold per-point behavior is subject to bf16/batch roundoff; trajectory-mean differences are much smaller, and formal claims need fixed precision. Final extractor PID1904003, b3e6d009, batch32,3300s plus prior121s andsmokes under combined1h, same3GiB cap. Analysis supervisor is CPU-only and starts only on complete cache. Output `.../mse-multimodality-analysis-20261015-r1`.

## Completed result and decision

Final aggregation56a1da34, final artifact `outputs/ptv3-shared-wam-v3/mse-multimodality-analysis-20261015-r4`; report [MSE/FM audit](../../mse-flowmatching-audit-20261015.md). GPU cache273.6s,11216 TRAIN/1024 DEV,570 original recordings. Window medians: amplitude0.718–0.921xGT, moving speed0.463–0.817xGT; ADE12.21–50.27mm. Every source median ADE beats zero-motion and constant-velocity baselines. At20mm no supported query among768 eligible;256 UniDex/HOI4D queries lack cross-split geometry support. At50mm only2 ARCTIC+2 H2O supported, no robust multimodal group; no architecture decision from these sparse neighborhoods. Keep MSE baseline; FM superiority not tested. Rerun later checkpoint on fixed panel or a matched small-head comparison is future decision evidence, not an omitted claimed result.

Implementation correction: H2O gap segments must merge by original take; file-level first analysis overstated10 supported H2O queries, corrected to2. An intermediate generic source_sequence rebinding merged UniDex relative episode names across independent batches; fixed by limiting rebinding toH2O, with regression. Intermediate analysis-r1/r2/r3 are superseded, no previous result is used to claim intrinsic stochasticity. Independent review and4tests passed. Final bootstrap also reports native group sensitivity, conditional on fixed bank.
