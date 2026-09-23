# V1.34 来源平衡的跨轨迹 CmLite 一步模型

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- evaluation code commit: `61fb966`
- run_status: `RUNNING`
- official actor checkpoint: `null`

使用 V1.33 的五个固定训练源与哈希；只增加
`train_cmlite.py --source-repeat 16 --source-repeat 16 --source-repeat 1
--source-repeat 1 --source-repeat 1`，其余参数不变。
训练分区内复制而不复制内部 holdout 的逻辑有单测。
独立评估为 s3 Cm-off e160 seed76 与 s1 自训练 PPO e140
seed76，分别收集首 episode 转移；详见 experiment card。

首轮独立采集已启动：GPU 5 是 Cm-off s3 e160
`eval_s76_e160_full`，GPU 6 是自训练 s1 PPO e140
`eval_s76_e140_full`，均为 64 env、提前终止关闭、保存
`transitions.pt`。每轮的输入、checkpoint SHA、资源、精确命令及
终态由评估目录 `run_manifest.json` 记录。训练预算固定为
30 epoch、batch2048、seed42、单 GPU6，在采集完成释放 GPU 后启动。
