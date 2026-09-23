# V1.34 来源平衡的跨轨迹 CmLite 一步模型

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

使用 V1.33 的五个固定训练源与哈希；只增加
`train_cmlite.py --source-repeat 16 --source-repeat 16 --source-repeat 1
--source-repeat 1 --source-repeat 1`，其余参数不变。
训练分区内复制而不复制内部 holdout 的逻辑有单测。
独立评估为 s3 Cm-off e160 seed76 与 s1 自训练 PPO e140
seed76，分别收集首 episode 转移；详见 experiment card。
