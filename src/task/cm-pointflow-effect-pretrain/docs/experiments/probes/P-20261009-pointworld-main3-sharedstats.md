# P-20261009-pointworld-main3-sharedstats

- `task`: cm-pointflow-effect-pretrain
- `branch`: agent/pointworld-ref8-fix
- `git_commit`: 285fd74517e1a363866572bcaefa39fa6bb840b9
- `class`: Decision
- `status`: COMPLETED
- `conclusion`: PROMISING (single-seed Probe only; not a formal cross-domain claim)
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
deadline occurs. During this run, checkpoint selection used the then-active
moving-anchor/static-object macro score and preserved both components. The
post-run contract repair uses complementary all-object moving/static strata;
it is covered by the tests below and does not trigger a retrain.

## Result

### Interim snapshot (superseded by final audit)

This snapshot was recorded at approximately step `42984/50000`; it is kept as
an intermediate trace and is superseded by the final audit below.

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

The curve entered a validation plateau after roughly step 33000, with normal
sampling fluctuations around the best score.

### Final run audit

The run completed at step `50000` in `37754.35 s` with two identical DDP
parameter hashes. The run's original selection contract chose the historical
`anchor/cat0` moving stratum and `static_objects/cat-1` static stratum; its
saved `best.pt` is therefore step `33000`, with `24.393 mm` moving-anchor,
`3.098 mm` static-object, and `13.745 mm` macro score. The final checkpoint's
balanced validation is `23.689 mm` moving-anchor and `4.321 mm` static-object.
On the symmetric post-run population, the same final checkpoint is
`21.896 mm` moving objects and `4.321 mm` static objects, or `13.109 mm`
macro.

For the post-run contract, selection is now defined symmetrically as
`moving_objects/cat-1` and `static_objects/cat-1`. Recomputing that score from
the recorded validation history gives a best observed point at step `40000`
(`21.951 mm` moving objects, `3.902 mm` static objects, `12.926 mm` macro),
but no new checkpoint is produced and no retraining is implied by this
selection-contract repair. The three saved checkpoints retain the shared
forward-stat hash
`536ca8e4a628dfbaf3965aba959d3cd3c9e2f56222437fb07232322c58d318d4` and
loss-stat hash
`07a258088ae52083f4b2f1547b5d46f9d4472db18794afaf7c989e9357b815ea`;
`final.pt` and `result.json` are present. The full Task CPU contract suite
passes (`80 passed, 6 skipped`), and the real CUDA model-level
batch-composition/gradient/checkpoint contract passes (`1 passed`). The
`motion_floor` remains `0.1`, and no HOCap or static-loss ablation was run, so
the Probe does not close the static-dynamics question.

## 2026-10-10 branch and worktree rename Decision Note

The user requested the development branch name `cm-pointflow-effect-pretrain`,
a matching worktree directory, and publication to origin. The existing local
and remote branch at `a620375771babbf358f29e6ac575628869f339d8` is an ancestor
of the completed ref8 route at `8bd733c4ee011d94d6ac6f0ebf41f99c2bf5308d`;
there is no divergent history. Training is complete and no training process
uses the old worktree.

Rename the previous local branch to
`cm-pointflow-effect-pretrain-before-ref8-20261010` to preserve its identity,
and rename `agent/pointworld-ref8-fix` to `cm-pointflow-effect-pretrain`. Move
the worktree to
`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-wt-cm-pointflow-effect-pretrain`, repair
Git registration, and retain the old `Ref2Dex-agent-wt-ref8` path as a symlink
for existing session paths and artifact references. Historical experiment
branch/commit fields above continue to identify the actual training code.

Cost is Git metadata and a small documentation commit, without training,
artifact copying, or checkpoint rewriting. Verify unchanged artifact location,
submodule identity, uncommitted reference content, and worktree registration,
then push normally with the new upstream. Stop on a branch collision,
unexpected worktree changes, or a non-fast-forward remote update; do not force
push. Preserve the old remote branch. User authorization covers renaming and
publication; no additional resource or permission expansion is required.
