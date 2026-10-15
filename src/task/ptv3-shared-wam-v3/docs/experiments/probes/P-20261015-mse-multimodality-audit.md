---
schema: ref2dex.probe.v2
probe_id: P-20261015-mse-multimodality-audit
experiment_id: P-20261015-mse-multimodality-audit
date: 2026-10-15
task: ptv3-shared-wam-v3
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-mse-conditional-multimodality
probe_index_in_family: 1
seed_pool: probe
seeds: [214]
decision_changed_if_positive: prioritize a matched frozen-backbone conditional FM versus refitted MSE head probe
decision_changed_if_negative: keep deterministic forecasting for now or improve conditional retrieval support before an FM investment
status: RUNNING
run_id: mse-multimodality-audit-20261015-r1
---

# Existing-weight conditional multimodality audit

Decision Probe for the MISSION short hand/object transition representation; follows user-supplied `docs/user/mse和flowmatching的选择.md`. Question: do similar observed H4 and supplied object-h24 goal G exhibit multiple supported future hand trajectories, and does the trained deterministic forecaster lie between those modes? Existing-weight diagnosis cannot establish that a newly trained FM would outperform MSE. FM itself regresses a conditional velocity field with squared loss; compare distributional versus deterministic prediction, not two mutually exclusive names for losses ([original FM](https://arxiv.org/abs/2210.02747)).

Use immutable model-only snapshot of the current remote strict recipe's best checkpoint (export captured step16000); current training keeps running. Earlier step8000 can be an undertraining sensitivity check after primary metrics, not an independent seed. The new DexYCB expanded recipe is excluded from primary analysis because the current weights were not trained on its revised data/statistics. Only TRAIN supplies retrieval futures; DEV supplies queries, not density/normalizer fitting. No heldout TEST is opened.

Bounded target: 128 DEV queries per actual DEV source (8sources,1024 total), up to2048 unique TRAIN bank windows/source. Retain one fully future-observed11-point hand and identical hand-presence pattern; report all omissions. Source/target exact geometry and scene-object identities separate incompatible coordinate/hand regimes. Retrieval uses frozen masked pre-head shared z (scene/hand/goal summaries), plus raw-condition retrieval control. Never use future hand labels for retrieval. Check actual H4 hand history, current scene history and supplied goal in metres. Gates20mm and50mm; record actual distances and unsupported queries, prefer stricter results. One neighbor per capture, avoid overlapped windows masquerading as separate modes. k32 preferred, minimum8 distinct captures; do not force clusters on unsupported conditions.

Metrics: conditional future spread; prediction→TRAIN conditional mean, nearest real future, cluster centroid/medoid; prediction-vs-GT empirical3-NN future distance; first8/all24/h24 ADE, amplitude and adjacent-step speed/acceleration attenuation; point-cloud unsigned fingertip proximity proxy at8/24 (not signed mesh contact or physical grasp success). Primary comparison uses absolute hand positions in the same target-current anchor; displacement-space sensitivity checks avoid confusing different start poses with different dynamics. Multimodal candidates require >=8 distinct captures, two clusters each>=20% and>=3members, diagonal-GMM BIC improvement>=10, silhouette>=.5 and trajectory-centroid separation>=20mm. GMM is exploratory, not proof of conditional modes at an exact continuous input. Positive/negative synthetic controls test whether the metric can distinguish a between-mode mean from real mode samples and ordinary unimodal spread.

If >=30 supported distinct DEV captures show robust multi-mode evidence and >=20% show between-mode/low-density prediction (mean closer than modes by>=5mm, predicted empirical NN distance>1.5xGT), prioritize a small matched FM head probe. If conditional support fails or effects vanish under tight/within-source/raw-retrieval checks, mark UNCLEAR rather than equating finite-radius variation with intrinsic stochasticity. Ratios use GT motion floors, static cases are reported separately; mean predictions are positive mechanism controls, retrieved actual futures are executable-data controls. These are decision thresholds, not a validation claim or an architecture switch.

## Resource and safety Decision Note

Local0..5 occupied by existing three actor runs,6/7 must remain free per current CAMPAIGN. Remote4-card pretraining is this session's owned run. Use only its GPU0 for bounded no-grad batch1 inference, allocator cap~3GiB, preflight free>=8GiB; stop audit on free<4GiB, OOM/nonfinite, own STOP,1h elapsed or2GB artifacts. Record GPU memory/utilization and parent training throughput; stop diagnostic if sustained interference or unsafe headroom appears. This colocates only with our own identified training process, never other/unknown jobs; no process is stopped or model/data/checkpoint overwritten. First small engineering smoke measures memory/rate before full bank. CPU loads/statistics use2threads. New checkpoint exports<1GB, cache/results<2GB, free disk>=20GiB. No new long identity or additional training budget requested.

Future evidence: exact matched-condition repeated trials, mesh contact/SDF/penetration, calibration/mode reliability, trained FM vs MSE with matched data/backbone/compute, minADE@M vs single-sample/precision/coverage and cost, formal multi-seed validation and robot utility. Do not infer these from the current diagnostic.
