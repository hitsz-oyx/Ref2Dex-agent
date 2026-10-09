# P-20261009-pointworld-main3-sharedstats

- `task`: cm-pointflow-effect-pretrain
- `branch`: agent/pointworld-ref8-fix
- `git_commit`: 40229e47adef1dcabaf7a8ef668fbff0ff440f5d
- `class`: Decision
- `status`: RUNNING
- `question`: Does a three-domain shared train-union forward normalization provide a clean fixed statistical interface for repaired PointWorld training?
- `hypothesis`: One shared OakInk2+GRAB+ARCTIC train-union normalization preserves physical comparability across domains better than retaining OakInk2-only forward statistics, while the existing mixed-domain loss scales remain valid.
- `decision_note`: For this run, choose shared multisource stats because the feature-group constants required by fixed physical-scale normalization have not been specified. Defer physical-scale normalization as a separate design route rather than inventing scales during a 50000-step comparison.
- `run_id`: pointworld-main3-sharedstats-20261009
- `output`: `outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/training-r1/`

## Protocol

Use the frozen registered main three-source corpus with unchanged source roots,
source weights, category mapping, 30Hz history/horizon, and the previous
50000-update temporal configuration (`microbatch=64`, `accumulation=2`,
`learning_rate=1e-4`, `motion_floor=.1`, validation microbatch 2). The derived
manifest only changes normalization identity and copies the frozen source index
arrays; it does not change data rows.

Forward statistics are estimated from 4096 train windows drawn with the
registered source weights across OakInk2, GRAB, and ARCTIC. Physical loss
statistics are independently estimated from 2048 three-source train windows.
Both files are train-only and hash-bound to their source/data contracts. The
fixed voxel origin is `[-2,-2,-2]` m with PTv3 bounds `0..65535`.

The run starts with explicit random initialization. The old step-50000
checkpoint is not imported because it was trained with OakInk2-only forward
statistics and the pre-ref8 batch-relative voxelizer. No existing output is
overwritten.

## Inputs

- data: `outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data/`
- forward stats: `outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json`
- loss stats: `outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/loss_stats.json`
- config: `configs/pointworld_multisource_sharedstats_50000_wm24.json`

The derived manifest hash is `babaf17fcb3e414472406247adb9d7abbf95bdbea86158372df4f053ab479c58`,
the shared normalization-source hash is
`fedfd19e0d14a5185d5f2710ce619b942815b415fe1a0099bf6317b78027017a`, and the
forward-stat hash is
`536ca8e4a628dfbaf3965aba959d3cd3c9e2f56222437fb07232322c58d318d4`.

## Limits and stop conditions

This is a single 50000-step training Probe, not a formal cross-domain
validation. It does not compare fixed physical-scale normalization or make a
zero-shot robot claim. Stop and preserve a bounded checkpoint if source/hash
drift, nonfinite loss, GPU conflict, out-of-bound coordinates, or the campaign
deadline occurs. Checkpoint selection uses the repaired moving/static macro
score and preserves both components.

## Result

### Interim snapshot (training still running)

This snapshot was recorded at approximately step `42984/50000`; it is not a
final result and does not close the Probe.

- run state: `TRAINING`; two-rank DDP on GPUs `0,1`; no nonfinite loss, OOM,
  source/hash drift, or fixed-workspace violation observed;
- latest training record: loss `0.1948`, gradient norm `2.041`,
  `0.715 s/step`, elapsed `32388 s`, learning rate `1.4320e-5`;
- latest completed validation: step `42750`, moving-anchor h24 point EPE
  `23.785 mm`, static-object h24 point EPE `3.949 mm`, balanced macro score
  `13.867 mm`;
- best validation point seen so far: step `33000`, moving-anchor h24
  `24.393 mm`, static-object h24 `3.098 mm`, balanced macro score
  `13.745 mm`;
- current checkpoints: `outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/training-r1/train-action/best.pt`
  and `latest.pt`; `final.pt` and the final result manifest are pending;
- resource snapshot: GPU0/GPU1 memory `24211/24576 MiB` and
  `22703/24576 MiB`; utilization remained active and the reserved-memory
  watermark stayed stable.

The curve has entered a validation plateau after roughly step 33000, with
normal sampling fluctuations around the best score. Continue to step 50000,
then replace this interim section with the final checkpoint and result
manifest audit before assigning the Probe conclusion label.
