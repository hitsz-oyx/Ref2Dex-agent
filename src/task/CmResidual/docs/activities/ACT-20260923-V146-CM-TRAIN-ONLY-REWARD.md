# V1.46 冻结 CmLite 训练期奖励对照

- date: `2026-09-23`
- branch: `agent/v146-cm-train-reward`
- run_status: `COMPLETED`
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

正式两臂各从原 e260 恢复并完成至 e300，e280/e300
checkpoint 齐全、run manifest 均 `COMPLETED`。
固定 e300 Cm-off SHA256
`36ff2ac7ffd4433b5f60b32dce7603cff5db24a0bab9866b9ab469d1ce645929`，
Cm-on SHA256
`278fc7a1b65837de7495d0e5da12a78ab177937a14210ae733134940deac15b2`。
训练末端 reward 约1030/747，但不是严格抓取率，
不按 reward 改选 epoch。下一步按固定 seeds104–108、
每臂每 seed 两次、GPU5/6交叉的矩阵做首个完整
episode 评估；评估 actor 不加载 Cm。

固定 seeds104–108、两臂各两次共20 run 全部
`COMPLETED`，无技术失败。Cm-on依次两次合计
41/50/63/58/50，Cm-off为81/81/74/89/74；
总计262/640对399/640，seed聚类95%区间
−27.34至−14.84pp。Cm 有益与稳定两个
预注册门槛均失败。平均接触占比0.368对0.615、
接触抬升0.101对0.217m，确认抓握退化。
分析脚本 `analyze_v146_eval_matrix.py` 逐项核验
来源/完整轨迹/固定矩阵，结果写在父矩阵目录。
