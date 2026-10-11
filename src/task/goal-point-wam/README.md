# goal-point-wam

This Task implements the approved no-video Goal-Point-WAM route on the frozen
OakInk2/GRAB/ARCTIC main-three corpus.

The scene encoder consumes only current/history scene points. The trajectory
branch receives a teacher endpoint goal made from the observed current hand
and the future step-24 hand. The world branch receives the executed future
hand trajectory and predicts the existing per-object rigid effect; it never
receives goal tokens. RGB, video, robot H328, PD, policy and evaluator losses
are outside this phase.

Use the existing mixed manifest and train-only normalization statistics:

After retiring the former worktree on 2026-10-11, use the path-relocated
`data-main-worktree` view below. Index/source hashes and normalization identity
are unchanged; the original `data` manifest retains its historical bytes.

```bash
PYTHONPATH=src/task/goal-point-wam/src:src/task/cm-pointflow-effect-pretrain/src \
python3 src/task/goal-point-wam/tools/run/train_three_domain.py \
  --data outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree \
  --stats outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json \
  --init-pw outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/training-r1/train-action/final.pt \
  --output outputs/goal-point-wam/<run_id> \
  --require-cuda
```

`--smoke --steps 4` runs the engineering-only contract, gradient, checkpoint
and goal-isolation check. Probe outputs belong under
`outputs/goal-point-wam/<run_id>/` and must not overwrite the historical
PointWorld outputs.

Fresh runs require `--init-pw`, or explicit `--random-init` for a control.
`--resume` is separate and verifies WAM code/data/config/stats identities.
The repaired 50k-step PW final checkpoint initializes all PTv3 backbone and
18-to-128 scene projection tensors. WAM goal/action/world/trajectory heads and
its differently structured object projection remain fresh. Temporal PW weights
are reused in the scene-only spatial encoder; this is weight transfer, not a
reproduction of PW's action-fused temporal forward. World memory now retains
all 528 executed-action tokens and their validity masks.
