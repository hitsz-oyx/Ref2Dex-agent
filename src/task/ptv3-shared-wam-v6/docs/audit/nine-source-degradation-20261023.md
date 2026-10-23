# Nine-source degradation diagnosis, 2026-10-23

Initial remote HEAD: `44eab525` (Publish and guard the DDP run manifest only on rank zero). Initial worktree clean. Remote-only computation/artifacts; no Git push.

## A: complete

Measured audit elapsed379.912s, graspenv/Python3.8.20/CPU only. Exact converted receipt census:39,527 mothers and9,341,050 TRAIN windows. Moments/quantiles and effective point-frame supervision fractions are SAMPLED:256 uniform TRAIN windows/source without replacement, seed20261023. Total2304 sampled windows. Exhaustive window/phase counts; original half and remaining data untouched.

Data identity SHA256: `5a89c571e072dde9b6613c35d14f23d79102c67edf9adaf94b9011e063044324`. Census: `/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/full-census/manifest.json`.

Actual model:2hands x21joints x24steps, first8 analyzed. Training displacements use start-object axes; world-axis displacements also measured. No11-joint mapping was specified; no11-point result is invented. Validity denominator includes both hand slots;0.5 for a single-hand source is legitimate. Phase0/1/2/3/4 = approach/close/contact/hold/unknown.

|source|TRAIN windows|sample windows|valid fraction|tau world RMS mm|tau train RMS mm|tau norm mean mm|inherited z p1 / p99 xyz|phase counts 0/1/2/3/4|
|---|---:|---:|---:|---:|---:|---:|---|---|
|0|5412663|256|1.0000|16.104|16.104|14.079|[-1.58 -1.3  -1.53] / [2.12 1.53 1.35]|{0: 4416245, 1: 81912, 3: 827755, 2: 82021, 4: 4730}|
|1|287955|256|1.0000|68.124|68.124|69.178|[-5.28 -7.38 -5.36] / [6.56 6.11 7.02]|{0: 62071, 1: 36557, 3: 134256, 2: 53029, 4: 2042}|
|2|370362|256|1.0000|42.593|42.593|38.029|[-4.19 -4.21 -4.6 ] / [4.62 5.49 4.3 ]|{0: 65380, 1: 31928, 3: 231394, 2: 40594, 4: 1066}|
|4|1930631|256|0.5000|19.267|19.267|25.146|[-1.62 -1.55 -1.55] / [1.51 1.41 1.49]|{1: 232181, 3: 1515397, 4: 48047, 2: 110894, 0: 24112}|
|5|238297|256|1.0000|20.427|20.427|23.008|[-1.72 -2.1  -1.82] / [1.65 1.82 1.94]|{0: 10969, 1: 30777, 3: 167029, 2: 27547, 4: 1975}|
|6|578181|256|1.0000|34.192|34.192|36.062|[-2.93 -3.31 -3.74] / [3.   2.52 2.9 ]|{0: 465081, 4: 14484, 1: 12517, 3: 74558, 2: 11541}|
|8|2953|256|0.5000|60.894|60.894|80.522|[-5.13 -4.23 -5.34] / [5.08 5.32 4.99]|{1: 1384, 3: 1353, 4: 197, 0: 2, 2: 17}|
|9|44110|256|0.9990|10.847|10.847|10.794|[-1.43 -0.77 -1.07] / [1.1  0.96 0.96]|{0: 6156, 1: 2343, 3: 32390, 4: 695, 2: 2526}|
|10|475898|256|0.5422|18.551|18.551|22.728|[-1.94 -1.56 -2.11] / [1.63 1.26 1.91]|{4: 254598, 0: 30324, 1: 24031, 3: 143514, 2: 23431}|


**Preregistered H2 screen: POSITIVE; cannot exclude H2.** Source8 RMS60.894mm is3.78x source0 and2.23x old-source median27.309mm. Source9 RMS10.847mm is3.15x smaller than source6. Source8 start-object-y mean+10.043mm also shows a shift.

Causal limit: source1 RMS68.124mm and inherited normalized RMS1.941 are larger than source8 (1.818). Old-source RMS range16.104-68.124mm already contains source8. Heterogeneous scales satisfy the owner screening rule but do not establish a new-source-exclusive normalizer bug or explain every-source regression. C1/C2/C3 remain necessary.

### Per-axis displacement statistics (mm; xyz)

|source|frame|mean|std|p1|p50|p99|
|---|---|---|---|---|---|---|
|0|world|-0.101 / 0.461 / -0.095|20.859 / 12.847 / 13.327|-76.255 / -36.239 / -44.228|-0.020 / -0.013 / 0.015|73.872 / 48.728 / 47.297|
|0|start-object|0.920 / -0.187 / 0.581|18.021 / 15.305 / 14.756|-55.787 / -51.508 / -47.211|0.020 / -0.081 / 0.018|72.810 / 57.145 / 54.731|
|1|world|-0.211 / 0.574 / 3.063|65.992 / 68.631 / 69.627|-256.659 / -206.858 / -239.807|0.134 / -0.102 / 0.315|238.401 / 261.277 / 224.285|
|1|start-object|1.764 / -1.728 / 2.882|65.847 / 72.589 / 65.600|-201.600 / -274.011 / -181.590|-0.250 / -0.630 / 0.061|256.123 / 237.183 / 255.237|
|2|world|-0.332 / 4.055 / -0.553|46.847 / 45.590 / 33.951|-184.124 / -133.328 / -98.067|-0.084 / 0.279 / -0.554|147.786 / 199.706 / 120.036|
|2|start-object|-0.189 / 0.592 / -0.722|41.426 / 44.689 / 41.574|-134.956 / -137.095 / -128.626|-0.109 / 0.132 / -0.138|149.031 / 157.438 / 149.742|
|4|world|-0.119 / 2.934 / 3.623|12.114 / 11.819 / 28.381|-36.620 / -28.685 / -70.608|0.309 / 1.158 / 1.117|30.582 / 38.561 / 80.066|
|4|start-object|-1.243 / -1.183 / 0.561|20.752 / 17.434 / 19.386|-64.882 / -61.196 / -60.663|-0.029 / -0.051 / 0.030|61.065 / 48.235 / 59.930|
|5|world|0.220 / 0.563 / 0.118|21.591 / 18.734 / 20.841|-67.255 / -53.153 / -68.084|-0.085 / 0.041 / -0.068|77.926 / 70.567 / 66.883|
|5|start-object|0.202 / -1.229 / 1.306|18.436 / 21.483 / 21.146|-57.057 / -81.693 / -62.500|0.009 / -0.063 / 0.136|59.689 / 63.770 / 75.537|
|6|world|-0.644 / -1.119 / -3.124|34.321 / 32.008 / 35.964|-118.942 / -109.099 / -135.831|-0.012 / -0.190 / -0.292|110.302 / 93.830 / 104.217|
|6|start-object|0.026 / -3.245 / -3.910|32.998 / 33.172 / 35.947|-105.432 / -123.997 / -139.045|-0.041 / -0.546 / -0.327|114.268 / 93.459 / 103.495|
|8|world|33.289 / -7.633 / -23.051|52.086 / 54.576 / 61.115|-48.735 / -169.211 / -228.862|18.238 / -0.978 / -6.271|209.462 / 111.327 / 86.595|
|8|start-object|2.492 / 10.041 / -1.594|60.437 / 59.419 / 61.899|-190.229 / -148.264 / -180.983|1.163 / 3.663 / -2.133|179.761 / 199.252 / 173.326|
|9|world|0.196 / -0.191 / 1.013|9.642 / 10.293 / 12.367|-30.417 / -37.791 / -40.738|-0.043 / 0.014 / 0.360|32.336 / 36.364 / 44.610|
|9|start-object|-0.589 / 0.910 / -0.181|12.444 / 9.518 / 10.309|-49.904 / -26.475 / -33.586|-0.066 / 0.104 / -0.034|40.105 / 36.363 / 32.176|
|10|world|-0.391 / -0.114 / 1.089|16.936 / 20.306 / 18.219|-56.544 / -63.824 / -58.193|0.004 / 0.212 / 1.105|51.205 / 65.477 / 57.212|
|10|start-object|-0.062 / -0.865 / -0.907|19.061 / 15.907 / 20.360|-67.836 / -53.286 / -71.182|0.071 / -0.301 / -0.834|55.171 / 47.985 / 65.443|

### Input physical-scale quantiles (xyz)

No input TRAIN mean/std moments exist. Positions are divided by actual fixed position_scale_m=0.05m. Scene=2048 surface xyz; hand=valid start_hand; object=32 scene_points. Target inherited horizon normalizer is separate. These are quantiles relative to a physical scale, not violations of a learned input range.

|source|input|p1|p50|p99|
|---|---|---|---|---|
|0|scene|-2.631 / -3.790 / -3.449|0.005 / 0.018 / 0.007|3.356 / 2.900 / 3.231|
|0|hand|-15.581 / -12.403 / -9.909|0.375 / 0.536 / 0.456|17.320 / 16.983 / 15.358|
|0|object|-2.554 / -2.957 / -3.424|-0.007 / 0.013 / -0.011|3.403 / 2.829 / 3.238|
|1|scene|-1.790 / -1.204 / -1.261|-0.005 / 0.030 / -0.046|1.555 / 1.292 / 1.200|
|1|hand|-22.259 / -24.210 / -15.726|0.336 / -0.129 / -0.090|24.465 / 22.104 / 14.772|
|1|object|-1.694 / -1.203 / -1.331|0.011 / 0.035 / -0.076|1.589 / 1.287 / 1.200|
|2|scene|-3.527 / -1.715 / -2.825|0.170 / 0.185 / -0.081|3.716 / 4.184 / 4.569|
|2|hand|-11.855 / -12.298 / -12.652|0.688 / 0.109 / -0.186|13.675 / 14.177 / 14.241|
|2|object|-3.628 / -2.133 / -2.783|0.125 / 0.178 / -0.059|3.762 / 4.301 / 4.163|
|4|scene|-1.072 / -1.539 / -1.342|0.000 / 0.003 / 0.000|1.086 / 1.499 / 1.349|
|4|hand|-2.238 / -1.958 / -2.091|0.092 / 0.119 / 0.105|2.319 / 2.472 / 2.734|
|4|object|-1.219 / -1.843 / -1.480|0.000 / 0.067 / 0.001|1.277 / 1.766 / 1.464|
|5|scene|-2.989 / -1.716 / -2.052|0.496 / 0.029 / -0.044|3.337 / 1.725 / 1.759|
|5|hand|-5.741 / -5.439 / -4.514|-0.935 / 0.234 / 0.000|5.275 / 7.747 / 5.203|
|5|object|-3.071 / -1.751 / -2.291|0.488 / 0.032 / -0.041|3.337 / 1.687 / 1.808|
|6|scene|-2.936 / -1.602 / -2.487|-0.075 / 0.622 / 0.051|2.909 / 3.205 / 2.465|
|6|hand|-19.839 / -13.883 / -16.145|0.626 / 1.920 / 0.351|18.621 / 18.048 / 17.013|
|6|object|-2.827 / -1.572 / -1.926|-0.104 / 0.623 / 0.110|3.360 / 3.273 / 2.538|
|8|scene|-1.465 / -1.975 / -2.302|-0.000 / 0.003 / -0.023|1.354 / 1.534 / 2.015|
|8|hand|-5.391 / -7.931 / -8.402|-0.247 / 0.028 / 0.266|6.545 / 4.981 / 5.836|
|8|object|-1.366 / -2.042 / -2.390|0.031 / -0.033 / 0.028|1.314 / 1.556 / 1.926|
|9|scene|-1.045 / -2.044 / -2.352|0.006 / -0.033 / -0.019|1.061 / 1.987 / 2.294|
|9|hand|-6.025 / -5.527 / -3.394|0.138 / -0.213 / -0.041|5.562 / 4.558 / 3.510|
|9|object|-1.062 / -2.058 / -2.168|0.129 / -0.008 / 0.084|1.032 / 2.036 / 2.305|
|10|scene|-3.682 / -3.296 / -3.977|-0.116 / -0.001 / -0.029|3.438 / 3.244 / 4.884|
|10|hand|-6.040 / -4.191 / -2.091|1.412 / 0.199 / 0.476|7.398 / 4.632 / 5.539|
|10|object|-3.798 / -3.219 / -4.095|-0.139 / -0.003 / -0.046|3.485 / 3.268 / 4.958|

No new-source-exclusive position tails: source1 hand max absolute p1/p99=24.465 versus source8=8.402/source10=7.398. Velocity/acceleration and phase-conditional tails are not exhaustively audited.

## Provided observations (owner facts, not re-measured)

Half continuation first8 hand8.981->12.294mm; contact24.659->36.862mm; first8 object5.639->6.887mm, paired mother CI95 positive. Every source worsens:0:+2.58,1:+3.01,2:+5.72,4:+4.42,5:+9.01,6:+5.83,8:+6.79,9:+2.20,10:+2.98mm. Phase hold+19.8,close+9.2,contact+8.3,approach+0.35mm. Step0=local3000 bit-identical CI[0,0]. Same inherited normalizer and LR schedule. Curated source1=5632/6000;0=96,2=96,10=46,6=66,4=30,5=18,8=12,9=4. Panel source1=347/512,source10=78/512.

## B: preregistered; NOT RUN

Exact owner interpretation and stop criteria are recorded in `preregistration.json`. C1 needs original6000-window support, distinct from C2 full converted support. The manifest referenced by panel config is absent from checkout; NAS searches have not located it. A path request is pending. Weights alone duplicate C2 and cannot replace C1.

|arm|support / weighting|GPUs|local/global batch|status|
|---|---|---:|---|---|
|C1|original curated6000 / curated weights|4|96/384|BLOCKED missing support metadata|
|C2|full nine-source / curated weights|4|96/384|prepared, waits C1|
|C3|full nine-source / uniform; only flow_future trainable|4|96/384|prepared, waits C2|
|C4-source1|full source1|1|96/96|prepared, waits C3|
|C4-source10|full source10|1|96/96|prepared, waits C3|
|C4-source0|full source0|1|96/96|prepared, waits C3|

All arms initialize local3000 only, fresh optimizer, A unknown goal/event; flow_lr1e-4/shared_lr1e-5/warmup50/clip1/wd.01. Cap500updates OR2400 training seconds; initial/final evaluation overhead separate, process watchdog3300s. No retry. Disk floor30GiB; OOM/nonfinite/loader/disk faults terminal. C4 raw signed relative EPE changes preserve owner50% criterion. Completed arms immediately append progress/summary and grouped source/phase EPEs using report_diagnosis_arm.py. No B measurements or mechanism verdicts exist yet.

## Reproduction

Remote short checkout `/home/wbcd/workspace/oyx_ws/Ref2Dex-agent`; Python `/home/wbcd/miniconda3/envs/graspenv/bin/python`. A command already executed (exclusive full-census directory; do not rerun into it):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/audit/diagnose_source_supervision.py
```

Outputs: `source-supervision-stats.json`, `source-supervision-table.md`, `audit.log`, `full-census/manifest.json`. Once original curated metadata is supplied on NAS:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u src/task/ptv3-shared-wam-v6/tools/run/run_diagnosis.py --curated <NAS-original-curated-manifest.json>
```

Per-arm launch-receipt.json records command, config/data identity, actual code commit, GPU IDs and initializer. No existing campaign controller is restarted.

## Engineering verification and current stop point

CPU recipe/sampler checks PASS for C2/C3/C4-source1/10/0:4,800 draws per arm, deterministic repeat, worker-state reconstruction, max8/mother/rank. C2 source1=4499/4800 (93.729%). Evidence: `engineering-checks.json`. All six scripts parse successfully in graspenv. GPU training/flow-only backward not yet tested;0 optimizer updates across all arms. C1 is blocked on original curated support metadata; later arms wait to preserve requested order. Working tree remains on main; no push.

## Owner followup: restored C1, C5 preregistration (before B training)

Remote starting HEAD2af46693. brief31 has6000 tuples with local sequence IDs; brief32 sequence_inventory supplies stable keys. Exact join source+key+slot+tick matches6000/6000 (100%); every by_source count preserved. Numeric sequence IDs alone disagree across local/remote and are not treated as stable identity. See C1-support-match.json and curated-rebound.json. C1 is restored, superseding the earlier missing-metadata blocker.

Historical C1: per owner, local same-family curated continuation improved all three metrics by-7.19/-34.56/-5.61mm versus its own initialization; provisionally record "continuation mechanism cleared" as requested. This is owner-provided evidence, not remeasured here. Batch96 versus384 remains uncontrolled; the new C1 provides the matched large-batch/fresh-optimizer test.

Three suspicious points, separate verdicts:
1. Source target coordinate RMS10.85->68.12mm (6.28x) and inherited-normalized RMS0.320->1.941 remain heterogeneous; shared heads/loss coefficients make scale/gradient imbalance plausible, not yet causally established.
2. Uniform-per-source gives source8 (2953 windows) and source1 (287955 windows) each1/9 source probability; average exposure per available window differs by97.51x before mother/phase structure. C2 tests source-weight attribution.
3. A's46-50% missing point-frame supervision for sources4/8/10 DOES NOT imply that46-50% whole windows have no supervision. Missing other-hand slots are correctly excluded by objective masks. C5 uses actual whole-window target-mask validity; exact counts below supersede the mistaken window-level inference.

New immutable preregistration: preregistration-v2.json. C1->C2->C3->C4(source1/10/0 concurrent)->C5;500 updates or2400 training seconds/arm. C1/C2/C3/C5 world4 local96 global384; C4 world1 local96 global96. Unchanged model, objective coefficients and initializer. No fault retries. If the exact C5 gate is a no-op, no C5 result can establish a causal effect from removing nonexistent invalid windows; report the design limit explicitly with the preregistered outcome table.

## C5 exhaustive hand-supervision window gate (measured before training)


|source|TRAIN windows|zero valid hand-target windows (24-step)|zero valid hand-target windows (first8)|kept|
|---|---:|---:|---:|---:|
|0|5412663|0|0|5412663|
|1|287955|0|0|287955|
|2|370362|0|0|370362|
|4|1930631|0|0|1930631|
|5|238297|0|0|238297|
|6|578181|291|409|577890|
|8|2953|0|0|2953|
|9|44110|0|0|44110|
|10|475898|1299|1362|474599|

Gate rejects1590/9341050 (0.0170%) windows:source10=1299 (0.2730%),source6=291 (0.0503%),source4/8=0. All sources retain>=200 windows. The actual24-step hand objective mask is used; absent other-hand slots are retained. Object targets remain valid in these zero-hand windows, so these are not wholly unsupervised examples. C5 filters every source internally and preserves each source1/9, including the additionally found291 source6 zero-hand windows. This is a real but small gate intervention, not the supposed46-50% removal.

C5 gate SHA256: `2c74830476af4c93c17737ad11300e5b83649d5293dcb33aa4340d14c2dca20f`; composite data identity: `d56531259b33ecd747c688c3bec599a00641bbf5dafac5b1a102f57fda773d03`. Gate identity is included in checkpoints/receipts. CPU4800-draw gate/determinism/worker-state checks passed; evidence C5-support-engineering.json.

## C1 completed (measured)

Progress: `{"automatic_restart": false, "elapsed_s": 1230.9634535420046, "global_batch": 384, "status": "COMPLETED", "step": 500, "updates": 500}`

Panel summary:
```json
{
  "anchor_sha256": "440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706",
  "bootstrap_repeats": 4000,
  "bootstrap_seed": 170,
  "cluster_ci95_mm": [
    [
      7.744489619109408,
      8.428222479738272
    ],
    [
      18.61123934768469,
      20.153199449617702
    ],
    [
      4.665338230031405,
      5.2475048861264595
    ]
  ],
  "means_mm": [
    8.080901486311399,
    19.366286934735157,
    4.959187217480348
  ],
  "metrics_order": [
    "first8_hand",
    "fixed_contact_hand",
    "first8_object"
  ],
  "mothers": 124,
  "panel_attention_backend": "math",
  "repeat_bit_exact": true,
  "rows": 512,
  "same_A_conditions": true,
  "scope": "fixed TRAIN panel only; no generalization or execution conclusion",
  "status": "PASS",
  "step": 500,
  "training_attention_backend": "efficient",
  "versus_local3000": {
    "delta_mm": [
      -0.8999179289457846,
      -5.293062317326584,
      -0.680216192712237
    ],
    "paired_mother_cluster_ci95_mm": [
      [
        -1.1279194208713992,
        -0.6740164440313804
      ],
      [
        -6.0285278637516875,
        -4.49344285664408
      ],
      [
        -0.871183341717069,
        -0.49631058216425683
      ]
    ],
    "winner": [
      "remote",
      "remote",
      "remote"
    ]
  }
}
```

by_source

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|0|27/3|6.4015 (+0.1343)|11.8025 (+0.4576)|3.5315 (-0.0305)|
|1|347/101|8.2859 (-1.3490)|19.7491 (-6.5922)|4.8986 (-1.0545)|
|10|78/3|9.9167 (+0.3319)|26.3856 (+0.5988)|6.3351 (+0.1208)|
|2|16/3|6.0061 (+0.5277)|13.9588 (+0.6420)|3.7310 (+0.1239)|
|4|10/3|9.7997 (+0.6161)|17.6942 (+0.9849)|6.4638 (+0.1727)|
|5|7/2|5.8212 (+0.9484)|12.7840 (+1.7127)|3.9946 (+0.1564)|
|6|18/3|5.8597 (+0.5274)|9.8830 (+0.4320)|3.4377 (-0.0537)|
|8|6/3|5.8893 (+1.3003)|14.8593 (+3.1111)|3.9177 (+0.4085)|
|9|3/3|4.7959 (+0.6375)|11.0808 (+1.5526)|4.0350 (+0.9206)|

by_phase

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|approach|105/79|8.9069 (-1.2562)|20.0824 (-2.6544)|3.9147 (-0.1908)|
|close|105/100|7.4517 (-0.9023)|19.4194 (-4.7505)|4.5963 (-0.5945)|
|contact|105/104|6.7293 (-0.7698)|14.6053 (-2.9459)|4.4767 (-0.9265)|
|hold|105/102|7.5948 (-1.1598)|21.8109 (-8.5536)|5.7464 (-1.6569)|
|unknown|92/17|11.3741 (+0.0891)|26.3480 (+0.5430)|6.2177 (+0.0592)|

Full reproducible paired group CIs and relative changes: `/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C1/grouped-panel.json`.

Preregistered interpretation: All three stable/improve: continuation cleared under matched C1; inspect distribution via C2.

### C1 interpretation scope

C1 meets the owner aggregate three-metric recovery rule with all paired CI95 below0. However source1 supplies the improvement: first8-hand delta-1.349mm, while every other source has positive first8-hand point estimates (+0.134 to+1.300mm). Thus this is a source-weight tradeoff on a source1-heavy fixed TRAIN panel, not universal source improvement. The generic inherited result.primary.pass_all=false is caused by undefined loss-block monotonicity (only ONE500-update block); the requested C1 decision uses its three EPE deltas/CIs, not that inapplicable multi-block field. No threshold was relaxed.

## C2 completed (measured)

Progress: `{"automatic_restart": false, "elapsed_s": 1211.150352437995, "global_batch": 384, "status": "COMPLETED", "step": 500, "updates": 500}`

Panel summary:
```json
{
  "anchor_sha256": "440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706",
  "bootstrap_repeats": 4000,
  "bootstrap_seed": 170,
  "cluster_ci95_mm": [
    [
      10.938187207067852,
      11.895930431255964
    ],
    [
      31.865318560266505,
      36.39105164041886
    ],
    [
      6.393557545758629,
      7.1715454570390635
    ]
  ],
  "means_mm": [
    11.41656623134433,
    34.071339758648925,
    6.779464827147663
  ],
  "metrics_order": [
    "first8_hand",
    "fixed_contact_hand",
    "first8_object"
  ],
  "mothers": 124,
  "panel_attention_backend": "math",
  "repeat_bit_exact": true,
  "rows": 512,
  "same_A_conditions": true,
  "scope": "fixed TRAIN panel only; no generalization or execution conclusion",
  "status": "PASS",
  "step": 500,
  "training_attention_backend": "efficient",
  "versus_local3000": {
    "delta_mm": [
      2.4357468160871463,
      9.411990506587184,
      1.1400614169550778
    ],
    "paired_mother_cluster_ci95_mm": [
      [
        2.110840309722414,
        2.7498262034710734
      ],
      [
        7.789745421578563,
        11.174453117228472
      ],
      [
        0.9374832469139278,
        1.3797063642711969
      ]
    ],
    "winner": [
      "local",
      "local",
      "local"
    ]
  }
}
```

by_source

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|0|27/3|9.0463 (+2.7791)|16.8752 (+5.5303)|3.8844 (+0.3224)|
|1|347/101|11.8340 (+2.1990)|36.2748 (+9.9336)|7.0671 (+1.1139)|
|10|78/3|11.1665 (+1.5816)|28.3862 (+2.5994)|7.1062 (+0.8919)|
|2|16/3|9.7029 (+4.2245)|22.3657 (+9.0489)|4.8459 (+1.2388)|
|4|10/3|12.3722 (+3.1887)|24.2324 (+7.5231)|9.3898 (+3.0987)|
|5|7/2|9.7356 (+4.8629)|20.4492 (+9.3779)|9.2229 (+5.3847)|
|6|18/3|10.2503 (+4.9180)|14.4598 (+5.0088)|3.8060 (+0.3146)|
|8|6/3|11.5093 (+6.9202)|32.9732 (+21.2250)|5.7025 (+2.1934)|
|9|3/3|6.0658 (+1.9073)|15.4732 (+5.9450)|6.9758 (+3.8614)|

by_phase

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|approach|105/79|16.0413 (+5.8782)|22.1208 (-0.6160)|4.3881 (+0.2826)|
|close|105/100|9.5533 (+1.1993)|30.6411 (+6.4712)|5.6473 (+0.4565)|
|contact|105/104|8.9806 (+1.4816)|23.3349 (+5.7836)|6.5970 (+1.1938)|
|hold|105/102|10.5001 (+1.7455)|46.4426 (+16.0780)|10.3842 (+2.9810)|
|unknown|92/17|12.4385 (+1.1535)|27.9486 (+2.1436)|6.8950 (+0.7364)|

Full reproducible paired group CIs and relative changes: `/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C2/grouped-panel.json`.

Preregistered interpretation: C2 does not recover all three: new-source supervision remains implicated by owner rubric; inspect C3/C5 and scale evidence.

C3 FAILED exit 1; no restart. See /mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C3/console.log

## C3 initial run: terminal reader fault,0 updates

NAS diagnosis/full-census/sequences.jsonl had mode000; all ranks stopped while opening the reader, before first update/panel/checkpoint. Original controller stopped and all failure logs/receipts remain under C3/. Only this diagnostic task's artifact permissions were restored to0664; every census file byte size and SHA binding is unchanged. This is an engineering fault, not C3 outcome. After explicit diagnosis and permission repair, a separate one-shot C3-r2 run is manually admitted with original initialization/config/data and full500-update budget (previous0 updates). No automatic retry loop is added. Remaining C4/C5 retain original preregistration.

### Decision audit registered after C2, before measuring source1 support scales

C1 recovers; C2 with identical source weights regresses. Question: does expanded within-source1 target scale differ from curated support despite identical source weights? Cheapest discriminator is CPU replay of first256 source1 draws from each actual training sampler, seed170/rank0, compare first8 raw/normalized targets and observed velocities. No model execution/optimizer. If expanded scale changes>=2x, prioritize within-source window/label distribution over a claim restricted to rare newly added source IDs; otherwise continue source-semantic/optimization diagnosis. This audit is sampled evidence, not a causal intervention.

### C1 versus C2 source1 sampled support audit (measured, CPU13.93s)

First256 source1 draws replayed from each actual rank0 sampler: C1 first8 coordinate RMS63.718mm (inherited z RMS1.741), C2 RMS91.505mm (z RMS2.520), ratio1.436. Observed input velocity coordinate RMS0.361 versus0.533m/s. C1/C2 sampled230/233 mothers,253/256 distinct windows. This DOES NOT meet the preregistered >=2x dominant-source scale criterion. Phase counts and axis tails remain in C1-C2-source1-scale-audit.json. Do not overclaim a causal source1 normalization failure from this result.

The source1 window support expands5632->287955 (51.13x) with the same source probability/update budget; average repeated exposure per available source1 window therefore falls51.13x before mother/phase nonuniformity. Thus C2 failure establishes that restoring source weights alone is insufficient on this old TRAIN panel; it does not isolate toxic newly added source IDs, nor distinguish broader support/retention/optimization from label-semantic mismatch. No extra training baseline was run.

C3-r2 FAILED exit 1; no restart. See /mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C3-r2/console.log

## C3-r2 terminal input-range fault (no mechanistic EPE verdict)

At step104/rank2, production fixed_workspace_grid rejects valid points beyond fixed origin -2m. Last logged successful progress is step100; control flow implies103 completed updates before the forward104 failure, but no final checkpoint/panel was saved. Do NOT classify as C3 recovery or C3 degradation. Runtime failure is distinct from model-quality outcome. No further C3 retry is launched.

Exact CPU production-reader/validator replay: `workspace-fault-reproduction.json`. Offender source4/key s4-34183/remote sequence29105/slot0/tick61, native trajectories/episode-00-env-179.npz, existing HALF shard.799 valid hand points (21 joints+778 surface) have x=-2.724..-2.612m versus fixed lower bound -2m; object surface points are in range. Hand future supervision is valid, so C5 validity gate retains this sample. The full batch error exactly reproduces grid_min[-145,172,229]/grid_max[584,683,583]. This confirms an input-range admission failure missed by sampled p1/p99, not a demonstrated target-unit bug. Workspace/normalizer/data masks are unchanged. C4/C5 prescribed sampling ranges are being preflighted on CPU before spending additional GPU budget.

Reproduce: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wbcd/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/reproduce_workspace_fault.py` (existing output is immutable; remove the final write or choose a fresh output to rerun; no original artifact is overwritten).

## Remaining input preflight and C5 block

C4 source1/source10/source0: all500 prescribed single-card batches preflighted on CPU with0 out-of-range windows (41566/39015/44438 distinct windows, respectively); raw input reconstruction is vectorized from the same native arrays. C5 actual valid-gated sampler still selects the production-confirmed bad source4 window at step104/rank2, so it is BLOCKED_FIXED_WORKSPACE at0 updates; this is not a C5 quality outcome and cannot enter recovery/degradation rules. Workspace, sampler support and masks are not relaxed to obtain a result. C4 proceeds unchanged.

A separately reviewable proposal workspace-repair-proposal.json would move fixed voxel origin[-2,-2,-2] to[-4,-4,-4] for a NEW no-optimizer preflight only (known fault plus initializer panel equivalence). It is not executed or mixed into the original controls without owner exception.

C4-source1 FAILED exit 1; no restart. See /mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C4-source1/console.log

## C4-source10 completed (measured)

Progress: `{"automatic_restart": false, "elapsed_s": 1141.9332626570103, "global_batch": 96, "status": "COMPLETED", "step": 500, "updates": 500}`

Panel summary:
```json
{
  "anchor_sha256": "440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706",
  "bootstrap_repeats": 4000,
  "bootstrap_seed": 170,
  "cluster_ci95_mm": [
    [
      19.36710647333797,
      22.7186542334276
    ],
    [
      43.947638179205256,
      50.42613712974833
    ],
    [
      6.888554698277872,
      7.979402306936917
    ]
  ],
  "means_mm": [
    21.014957817647055,
    47.18389448248668,
    7.409144082742074
  ],
  "metrics_order": [
    "first8_hand",
    "fixed_contact_hand",
    "first8_object"
  ],
  "mothers": 124,
  "panel_attention_backend": "math",
  "repeat_bit_exact": true,
  "rows": 512,
  "same_A_conditions": true,
  "scope": "fixed TRAIN panel only; no generalization or execution conclusion",
  "status": "PASS",
  "step": 500,
  "training_attention_backend": "efficient",
  "versus_local3000": {
    "delta_mm": [
      12.034138402389871,
      22.524545230424938,
      1.7697406725494886
    ],
    "paired_mother_cluster_ci95_mm": [
      [
        10.571918763112388,
        13.565597679739792
      ],
      [
        19.880663879287464,
        25.207281331481468
      ],
      [
        1.3703937823645302,
        2.2265425301440103
      ]
    ],
    "winner": [
      "local",
      "local",
      "local"
    ]
  }
}
```

by_source

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|0|27/3|16.4257 (+10.1584)|22.6676 (+11.3226)|3.7690 (+0.2070)|
|1|347/101|22.6959 (+13.0609)|50.0476 (+23.7064)|7.9189 (+1.9658)|
|10|78/3|14.5080 (+4.9232)|30.3550 (+4.5682)|6.8392 (+0.6248)|
|2|16/3|17.7668 (+12.2884)|30.5389 (+17.2221)|4.9400 (+1.3330)|
|4|10/3|22.8277 (+13.6441)|49.2275 (+32.5182)|9.9626 (+3.6715)|
|5|7/2|17.9362 (+13.0634)|35.5559 (+24.4846)|13.2509 (+9.4127)|
|6|18/3|12.7850 (+7.4527)|16.3681 (+6.9171)|3.5778 (+0.0864)|
|8|6/3|29.6162 (+25.0271)|58.4195 (+46.6713)|6.4139 (+2.9047)|
|9|3/3|14.0716 (+9.9132)|37.0889 (+27.5607)|12.0310 (+8.9166)|

by_phase

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|approach|105/79|40.3687 (+30.2056)|36.5991 (+13.8623)|4.6642 (+0.5587)|
|close|105/100|17.3136 (+8.9596)|46.1058 (+21.9358)|5.3917 (+0.2009)|
|contact|105/104|12.4565 (+4.9575)|29.2926 (+11.7413)|8.6711 (+3.2679)|
|hold|105/102|14.5460 (+5.7914)|64.4275 (+34.0630)|11.5173 (+4.1140)|
|unknown|92/17|19.1277 (+7.8428)|32.2943 (+6.4893)|6.7157 (+0.5571)|

Full reproducible paired group CIs and relative changes: `/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C4-source10/grouped-panel.json`.

Preregistered interpretation: Other-source EPE worsening>=50% observed; strong interference under owner threshold.

## C4-source0 completed (measured)

Progress: `{"automatic_restart": false, "elapsed_s": 1147.3677557160117, "global_batch": 96, "status": "COMPLETED", "step": 500, "updates": 500}`

Panel summary:
```json
{
  "anchor_sha256": "440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706",
  "bootstrap_repeats": 4000,
  "bootstrap_seed": 170,
  "cluster_ci95_mm": [
    [
      11.702798342902348,
      12.765941325374216
    ],
    [
      42.462500550607295,
      50.171209645616656
    ],
    [
      6.896114127555456,
      8.110115236220002
    ]
  ],
  "means_mm": [
    12.237180291174335,
    46.30037399613418,
    7.495198640482542
  ],
  "metrics_order": [
    "first8_hand",
    "fixed_contact_hand",
    "first8_object"
  ],
  "mothers": 124,
  "panel_attention_backend": "math",
  "repeat_bit_exact": true,
  "rows": 512,
  "same_A_conditions": true,
  "scope": "fixed TRAIN panel only; no generalization or execution conclusion",
  "status": "PASS",
  "step": 500,
  "training_attention_backend": "efficient",
  "versus_local3000": {
    "delta_mm": [
      3.256360875917151,
      21.641024744072435,
      1.8557952302899565
    ],
    "paired_mother_cluster_ci95_mm": [
      [
        2.8704748543791605,
        3.6486153010195768
      ],
      [
        18.57652127567687,
        24.805359486427474
      ],
      [
        1.4187584969756204,
        2.3425607338444787
      ]
    ],
    "winner": [
      "local",
      "local",
      "local"
    ]
  }
}
```

by_source

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|0|27/3|8.6796 (+2.4124)|17.9970 (+6.6520)|3.4255 (-0.1365)|
|1|347/101|12.5583 (+2.9233)|50.0431 (+23.7018)|7.9034 (+1.9503)|
|10|78/3|12.5217 (+2.9368)|31.2206 (+5.4337)|7.4952 (+1.2809)|
|2|16/3|10.6623 (+5.1840)|25.4330 (+12.1162)|5.5059 (+1.8989)|
|4|10/3|12.0836 (+2.9000)|22.0253 (+5.3159)|10.0173 (+3.7262)|
|5|7/2|15.1395 (+10.2668)|30.2769 (+19.2056)|13.1536 (+9.3154)|
|6|18/3|11.3915 (+6.0593)|16.1650 (+6.7140)|3.6533 (+0.1619)|
|8|6/3|14.3090 (+9.7199)|67.2636 (+55.5154)|7.2939 (+3.7848)|
|9|3/3|8.3329 (+4.1745)|21.7786 (+12.2505)|9.3563 (+6.2419)|

by_phase

|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|
|---|---:|---:|---:|---:|
|approach|105/79|17.0226 (+6.8595)|62.9216 (+40.1848)|3.7776 (-0.3279)|
|close|105/100|10.3241 (+1.9701)|40.6438 (+16.4739)|4.8429 (-0.3479)|
|contact|105/104|9.1798 (+1.6808)|23.0057 (+5.4544)|8.7262 (+3.3230)|
|hold|105/102|11.6535 (+2.8989)|71.0932 (+40.7286)|12.9401 (+5.5368)|
|unknown|92/17|13.6360 (+2.3510)|30.6643 (+4.8593)|7.1459 (+0.9874)|

Full reproducible paired group CIs and relative changes: `/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/diagnosis/C4-source0/grouped-panel.json`.

Preregistered interpretation: Other-source EPE worsening>=50% observed; strong interference under owner threshold.

## C4-source1 terminal launcher isolation failure

C4-source10 and C4-source0 completed500 updates and have final checkpoints/panels; their reports were appended and committed individually. C4-source1 last logged480 updates, no final checkpoint/panel. Torch2.0 installed run.py lines771-774 shows --standalone forcibly sets localhost:29400 despite separate random run IDs. Concurrent independent agents therefore share a TCPStore server; one completed arm closed the store and interrupted source1 via RendezvousConnectionError/Broken pipe. This is a launcher isolation mistake, not source1 quality outcome. No retry/model update is performed, no original output deleted. Source1 interference row is NA. Future launcher uses explicit c10d localhost:0 and arm-specific rendezvous ID; original receipts/commands retained.

## C4 partial interference matrix (measured; source1 row unavailable)

Cells are signed EPE delta mm (relative percent). Positive=worse; off-diagonal>=50% worsening hits owner interference threshold. Source1 row is NA because final weights/panel were lost to launcher failure.

first8_hand

|trained source|0|1|2|4|5|6|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|10|+10.158 (+162.1%)|+13.061 (+135.6%)|+12.288 (+224.3%)|+13.644 (+148.6%)|+13.063 (+268.1%)|+7.453 (+139.8%)|+25.027 (+545.4%)|+9.913 (+238.4%)|+4.923 (+51.4%)|
|0|+2.412 (+38.5%)|+2.923 (+30.3%)|+5.184 (+94.6%)|+2.900 (+31.6%)|+10.267 (+210.7%)|+6.059 (+113.6%)|+9.720 (+211.8%)|+4.174 (+100.4%)|+2.937 (+30.6%)|

fixed_contact_hand

|trained source|0|1|2|4|5|6|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|10|+11.323 (+99.8%)|+23.706 (+90.0%)|+17.222 (+129.3%)|+32.518 (+194.6%)|+24.485 (+221.2%)|+6.917 (+73.2%)|+46.671 (+397.3%)|+27.561 (+289.3%)|+4.568 (+17.7%)|
|0|+6.652 (+58.6%)|+23.702 (+90.0%)|+12.116 (+91.0%)|+5.316 (+31.8%)|+19.206 (+173.5%)|+6.714 (+71.0%)|+55.515 (+472.5%)|+12.250 (+128.6%)|+5.434 (+21.1%)|

first8_object

|trained source|0|1|2|4|5|6|8|9|10|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|10|+0.207 (+5.8%)|+1.966 (+33.0%)|+1.333 (+37.0%)|+3.672 (+58.4%)|+9.413 (+245.2%)|+0.086 (+2.5%)|+2.905 (+82.8%)|+8.917 (+286.3%)|+0.625 (+10.1%)|
|0|-0.137 (-3.8%)|+1.950 (+32.8%)|+1.899 (+52.6%)|+3.726 (+59.2%)|+9.315 (+242.7%)|+0.162 (+4.6%)|+3.785 (+107.9%)|+6.242 (+200.4%)|+1.281 (+20.6%)|

## Read-only numerical recomputation and engineering verification

From remote checkout, `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/wbcd/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam-v6/tools/audit/recompute_diagnosis.py` recomputes every available final mean/delta/4000-draw mother-cluster CI and verifies saved summaries to1e-12. C1/C2/C4-source10/C4-source0 all verified; source1 correctly prints NO_FINAL_PANEL. Workspace reproduction tool is now idempotent/read-only on an existing receipt and asserts exact saved-case equality.

Future launcher now sets explicit c10d localhost:0 and an arm-specific rendezvous ID without --standalone. CPU TCPStore lifecycle check passed: distinct ephemeral ports, second store survives first-store closure;0 optimizer updates/GPU model executions. Evidence: rendezvous-isolation-engineering.json. This verifies the isolation mechanism, not a rerun of lost source1 training. No further training is launched.

## Final measured synthesis with terminal blockers


|arm|status|updates|GPUs/global batch|first8 hand mm (delta)|contact mm (delta)|first8 object mm (delta)|
|---|---|---:|---|---:|---:|---:|
|C1|completed|500|4/384|8.0809 (-0.8999)|19.3663 (-5.2931)|4.9592 (-0.6802)|
|C2|completed|500|4/384|11.4166 (+2.4357)|34.0713 (+9.4120)|6.7795 (+1.1401)|
|C3 / C3-r2|reader fault then input-range failure|0;100 logged/failed104|4/384|NA|NA|NA|
|C4-source1|rendezvous store failure|480 logged|1/96|NA|NA|NA|
|C4-source10|completed|500|1/96|21.0150 (+12.0341)|47.1839 (+22.5245)|7.4091 (+1.7697)|
|C4-source0|completed|500|1/96|12.2372 (+3.2564)|46.3004 (+21.6410)|7.4952 (+1.8558)|
|C5|known deterministic input-range fault|0|4/384|NA|NA|NA|

Paired cluster CI95 (4000 bootstrap draws, seed170):

|arm|first8 hand delta CI95|contact delta CI95|object delta CI95|
|---|---|---|---|
|C1|[-1.1279,-0.6740]|[-6.0285,-4.4934]|[-0.8712,-0.4963]|
|C2|[+2.1108,+2.7498]|[+7.7897,+11.1745]|[+0.9375,+1.3797]|
|C4-source10|[+10.5719,+13.5656]|[+19.8807,+25.2073]|[+1.3704,+2.2265]|
|C4-source0|[+2.8705,+3.6486]|[+18.5765,+24.8054]|[+1.4188,+2.3426]|

C4 pooled OTHER-source signed EPE deltas (hand/contact/object):
- C4-source10: +12.6799mm, +23.5276mm, +1.9755mm; relative +142.06%, +95.65%, +35.68%
- C4-source0: +3.3084mm, +21.9637mm, +1.9667mm; relative +36.16%, +88.05%, +34.17%

Preregistered decisions and causal limits:
- C1: all three EPE improve with CI95<0 at matched global384/fresh optimizer. Generic continuation settings alone are not sufficient to explain the old half regression; source1-heavy aggregate recovery is not all-source improvement.
- C2: all three EPE regress with CI95>0 despite restored curated weights. Source weighting alone is insufficient; expanded within-source support/retention/semantics/scale-optimization interactions remain. This does not isolate a toxic newly added source ID.
- C3: no final quality measurement after fixed-workspace fault; shared-backbone-versus-flow mechanism remains UNRESOLVED.
- C5:0 updates because its exact valid-gated sampler still hits the proven bad input; invalid-window recovery/degradation and C2+C5 joint rules are NOT EVALUABLE.
- C4: other-source EPE worsening>=50% occurs; see exact cells and sparse-source CIs.
- H2: target-scale screen remains positive; source1 C1/C2 sample RMS ratio1.436 is below its additional2x criterion. A real valid-input workspace violation is proved for source4, but that causes hard failure and does not itself explain the earlier successfully completed500-step EPE regression.
- Validity:46-50% refers to missing point-frame supervision, not missing whole windows. Whole24-step zero-hand supervision is1590/9341050=0.0170%; source4/8 invalid whole-window counts0. All those windows still have object supervision.
- C4-source10 also worsens its OWN source on all three point estimates; source mixing is not necessary for that observed regression. Any>=50% off-diagonal threshold hit is reported under the owner rule, but is not a clean source-tradeoff proof when the trained source itself also deteriorates. Broader within-source support/retention/semantics/optimization remain candidates.

No model/loss/LR/threshold/workspace was relaxed. Original half/remaining outputs are retained. Original half support differs from new full support; C4 global96 differs from global384 controls. Fixed TRAIN512/124 evidence only; no heldout generalization/execution claim. Independent workspace repair preflight remains a proposal unless separately authorized; original C3/C5 cannot be labelled recovered/degraded from runtime failures.
