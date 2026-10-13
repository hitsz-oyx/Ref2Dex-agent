---
schema: ref2dex.probe.v2
probe_id: P-20261013-pointwam-repro-patch-fit
experiment_id: P-20261013-pointwam-repro-patch-fit
date: 2026-10-13
task: pointwam-repro
branch: main
git_commit: 10e7b6f9
claim_id: C1
hypothesis_family: HF-pointwam-repro-human-forecaster
probe_index_in_family: 1
seed_pool: probe
seeds: [110]
decision_changed_if_positive: keep this 1cm patch wiring and test frozen scene features on the same packs
decision_changed_if_negative: repair the reduced forecaster before spending budget on Mosaic3D or more epochs
status: UNCLEAR
run_id: pointwam-repro-patch-fit-20261013-r4
---

# Reduced patch forecaster on qualified human clips

Result: UNCLEAR. Zero-init fit hand EPE 54.8 to 29.5 mm, but fit scene 9.0 to 15.1 mm and dev hand 69.3 to 87.1 mm.
Decision: Stop scaling this substitute. Do not add epochs or Mosaic3D until scene and held-out hand can improve together.

Class: Decision. The 48-clip contract had already passed. This asks whether the
reduced 1 cm / 5³ forecaster can lower fit displacement error from current
scene and hand points without training on dev clips. It does not test
Mosaic3D, Recap-CLIP, Volt-H, or robot transfer. Sources ran from the dirty
worktree copied to the remote run directory; `10e7b6f9` is only the parent commit.

## Protocol

- Fit 23 clips train; dev 25 evaluate. Packs stay `training_allowed=false`.
- At most 1024 currently valid scene points, chosen without looking at future motion.
- Paper motion-weighted per-coordinate loss on scene and hands. Static control is zero displacement.
- AdamW `1e-4`, weight decay `0.01`, clip `5`, seed `110`, fp32, GPU0.
- Fit gate: both fit EPEs at most 90% of the untrained model. Dev gate: hand EPE
  no worse than static, scene EPE at most 105% of static.
- GPU3 reconstruction was not touched. Peak allocated memory on these runs was about 35 MiB.

## Runs

r1 wrote a nonfinite loss in 0.6 s and took 0 steps. Invalid targets are stored
as NaN, and `NaN * 0` poisoned the scene term. That run is not evidence about
the model. The mask now ignores nonfinite pairs. Remote directory
`pointwam-repro-patch-fit-20261013-r1`.

r2 kept the random final layer for 200 steps / 3.7 s. Fit scene/hand EPE fell
from 160.6/277.3 mm to 20.9/106.5 mm, still far above static 9.0/54.8 mm.
Dev ended at 24.6/117.4 mm versus static 11.6/69.3 mm. Under the predeclared
rule this is UNPROMISING for scaling: the budget improved a bad initialization
and did not beat zero displacement.

r3 zeroed the output heads, so step 0 matched static, then ran 1000 steps / 15.9 s.
Fit hand EPE 54.8 to 29.5 mm. Fit scene EPE 9.0 to 15.1 mm. Dev hand 69.3 to
87.1 mm and dev scene 11.6 to 18.4 mm. Train loss fell because the hand term
is much larger than the scene term. Fit gate failed, so the label is UNCLEAR.

r4 divided each term by its detached static loss and repeated the 1000 steps.
Fit hand still improved, 54.8 to 30.6 mm, while fit scene rose 9.0 to 17.0 mm
and dev hand rose 69.3 to 91.0 mm. Balancing the two coefficients does not
remove the scene or held-out regression. Same UNCLEAR gate.

## Decision and limits

Do not continue epochs, enlarge this transformer, or attach Mosaic3D on the
claim that this wiring is ready. The repeated pattern is fit-hand movement
with worse scene error and worse dev hands. Full 12,000-point clouds, frozen
encoders, language, and robot transfer remain future evidence. Local numbers
are in [summary.json](../../../../../../outputs/pointwam-repro/pointwam-repro-patch-fit-20261013/summary.json).
Remote r4 result is
`/mnt/ugreen_nas/storage/oyx_storage/pointwam-repro-patch-fit-20261013-r4/result.json`.
