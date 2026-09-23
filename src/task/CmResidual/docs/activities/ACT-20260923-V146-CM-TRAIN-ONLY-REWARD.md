# V1.46 冻结 CmLite 训练期奖励对照

- date: `2026-09-23`
- branch: `agent/v146-cm-train-reward`
- run_status: `RUNNING`
- training code commit: `d0922d6`
- official actor checkpoint: `null`

冻结源、smoke/正式训练、heldout 重复评测矩阵
与停损条件见 experiment card。训练时按长任务
规范记录代码 commit、source/Cm SHA、输入 manifest、
GPU5/6、固定 e300 checkpoint、预算与异常停机。

两臂 e262 smoke 均 `COMPLETED`，日志确认各从
SHA `16fd261b…` 的 e260 自训练权重加载、学习率
`1e-5`、e261/e262 连续训练、e262 checkpoint
已写出；Cm-on config 锁定 V1.37 SHA，奖励日志
有限且真实接触门有效（一个早期 horizon 的门
比例0.36、正奖励比例0.12），Cm-off 的 config
明确没有模型。两臂训练 reward 约45并非完整
episode 抓取指标。正式实验从原 e260重新启动，
不会使用 smoke 权重。
