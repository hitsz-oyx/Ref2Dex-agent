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
  --output outputs/goal-point-wam/<run_id> \
  --require-cuda
```

`--smoke --steps 4` runs the engineering-only contract, gradient, checkpoint
and goal-isolation check. Probe outputs belong under
`outputs/goal-point-wam/<run_id>/` and must not overwrite the historical
PointWorld outputs.
