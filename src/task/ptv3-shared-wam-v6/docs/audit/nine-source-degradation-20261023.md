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
