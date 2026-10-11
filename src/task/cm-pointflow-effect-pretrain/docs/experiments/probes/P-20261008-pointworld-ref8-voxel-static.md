# P-20261008-pointworld-ref8-voxel-static

- `task`: cm-pointflow-effect-pretrain
- `branch`: agent/pointworld-ref8-fix
- `git_commit`: c65d308e9be85e15892efef520df86d33d4c3a0d
- `class`: Decision
- `status`: COMPLETED
- `conclusion`: PROMISING (implementation repair retained; no model-quality claim)
- `question`: Does removing batch/action-dependent voxel origins materially change the existing 50000-step PointWorld endpoint, and what are its moving/static object errors under the repaired evaluator?
- `hypothesis`: The old endpoint's outputs can change because the legacy origin is selected from all batch points; the repaired fixed-origin path should be invariant to partners and candidate actions. Static and moving errors must be reported separately before deciding whether a short motion-floor ablation is worth the budget.
- `decision_changed_if_positive`: If the endpoint is materially sensitive to origin choice, treat old quality numbers as implementation-specific and prioritize repaired-code evaluation before any motion-floor training. If sensitivity is negligible, keep the repair as a correctness fix and decide the next probe from the static/moving split.
- `cheapest_probe`: The checkpoint-fixed validation panel (192 windows), one frozen old action checkpoint, same model weights, no fitting; compare legacy voxelization against the repaired fixed workspace and report `moving_objects`/`static_objects` metrics.

## Protocol

Use the read-only checkpoint
`outputs/cm-pointflow-effect-pretrain/pointworld-main3-20261007/train-action/latest.pt`
(step 50000), the checkpoint's saved balanced validation indices, and the
registered main three-source `MixedWindows` validation corpus used by that
checkpoint. The legacy arm is reproduced in-process by restoring the pre-ref8
`coords.amin(0)` voxelizer; the repaired arm uses
`grid=floor((x-[-2,-2,-2])/0.01)` with PTv3/Hilbert bounds `0..65535`.
Evaluation uses the existing action arm, microbatch 2, fixed TF32/BF16
settings, and no optimizer state. The static mask is exactly the complement of
the existing moving rule (translation `>2mm` or rotation `>.02rad` at any
horizon). Outputs go under the worktree-local run directory and do not touch
the historical run.

## Limits

This is a single-seed implementation Probe, not a retraining result or a
scientific validation. It does not compare motion-floor values and does not
claim that either voxelizer is better on held-out data. A repaired run that
fails fixed workspace bounds is an implementation blocker.

## Result

Attempt r1 stopped before model inference because the checkpoint-fixed panel has
192 windows; the failed manifest is preserved. Attempt r2 stopped after the
dataset identity check found that those indices belong to the registered
three-source mixed corpus rather than OakInk2-only windows. Attempt r3 used the
correct corpus but exposed that the first fixed origin was too tight: a valid
panel coordinate produced `grid_min=[19,-12,50]` under `[-1,-1,-1]`. No fixed
arm metrics were written; the failure manifest is preserved. Attempt r4 is the
bounded rerun with `[-2,-2,-2]`, selected from the panel scan (global range
approximately `[-1.067,-1.152,-1.025]` to `[1.008,1.182,0.682]` m).

Attempt r4 completed on GPU0 with the frozen step-50000 checkpoint and all 192
saved validation indices. The repaired arm was in bounds under `[-2,-2,-2]`.
Both raw metric maps and target-output deltas are in
`outputs/cm-pointflow-effect-pretrain/P-20261008-pointworld-ref8-voxel-static/attempt-r4/`.

The balanced selection score (moving-anchor h24 plus static-object h24, macro
averaged) was 15.573 mm for the legacy batch-relative arm and 15.508 mm for the
fixed workspace arm. The components were:

| arm | moving-anchor h24 | static-object h24 | macro score |
| --- | ---: | ---: | ---: |
| legacy `coords.amin(0)` | 28.053 mm | 3.094 mm | 15.573 mm |
| fixed `[-2,-2,-2]` | 28.744 mm | 2.272 mm | 15.508 mm |

The same checkpoint's predictions were materially sensitive to the old
voxelizer: maximum absolute translation difference was 93.748 mm and maximum
rotation difference 1.801 rad; means over returned target rows were 2.312 mm
and 0.0193 rad. This is an implementation-sensitivity result, not evidence
that either arm has better held-out quality. It does justify retaining the
fixed contract and re-evaluating any future checkpoint under it before
starting the deferred `motion_floor` ablation. No training was launched and no
historical checkpoint was overwritten.
The panel and an additional 2,000-window validation sample found no points
outside this contract; the sampled range was approximately
`[-1.428,-1.303,-1.330]` to `[1.297,1.324,1.508]` m. The bounded audit is
recorded in `outputs/cm-pointflow-effect-pretrain/P-20261008-pointworld-ref8-voxel-static/workspace_scan.json`.
