# V1.39: 自训练 s3 策略的继续训练与接触课程回启

- experiment_id: `EXP-20260923-V139-S3-CURRICULUM-CONTINUE`
- branch: `agent/multitrajectory-v129`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（s3 每 seed ≥90% 稳定门槛；探索集有进展）
- official actor checkpoint used: `no`

## 假设与冻结设计

V1.35 的 s3 自训练 Cm-off 策略从 e160 到 e180，seed77
严格抓取从 e170 的29/64 升到 e180 的35/64；另在新
seed79 达42/64。当前 Cm 奖励和一步候选选择均未优于
Cm-off，但多轨迹稳定性目标仍远未达到。本实验只测试
更强的**自训练、无 Cm 在线干预基线**，不用于证明 Cm。

两臂固定同一 s3 Cm-off e180 checkpoint，SHA256
`863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c`，
同一 s3 `corrected_manifest_r2.json`、训练 seed70、64 env、
horizon32、minibatch256、学习率`1e-5`、相同几何/抬升奖励、
20-epoch 保存频率，各继续80 epoch至 e260。

- 标准继续臂保留原 `curriculum-anneal-start=40/end=80`；
  e180 后近接触重置比例为0，与源训练一致。
- 接触课程回启臂取消 anneal，近接触重置比例0.5、参考抬升
  重置比例0.25，并设 `curriculum-backtrack-start=180/end=220`。
  前期在接触/抬升附近复习，随后将接触重置窗口逐步扩展到
  从轨迹开始至首次接触；评估起点仍完全不变。

先每臂 e182 接线 smoke，确认同源 SHA/学习率/课程日志；
正式两臂从原 e180 重新启动，不用 smoke 权重。独立 seed81
完整轨迹/无提前终止评估源 e180 及两臂 e200/e220/e240/e260
各64首 episode；按严格成功数选择臂及 checkpoint，平局
按平均最大接触抬升、较早 epoch。若最佳新 checkpoint 未
比同 seed 源 e180 **至少多10/64**，预注册停损，不扩大。
否则冻结所选单策略在未见 seeds82–84 各评估64 episode；
s3 稳定门槛仍为每 seed ≥90%，不能以选择集 seed81 代替。
从相同源做两臂只能比较这两个课程配方，不能把多试验
选择偏差当作总体效果。

GPU5/6 各一臂，最多两卡并发；预计总训练数分钟、输出
远低于300GB。源/输入 SHA 漂移、外部占卡、恢复错误或
数值异常即停。此 run 是显式新微调实验，不静默修改旧 run。

## 完整轨迹结果

两臂 80 epoch 正常完成，e200/e220/e240/e260 checkpoint
齐全。回启课程在 e190/e200/e210/e220 的进度日志依次为
0.25/0.5/0.75/1.0，证明课程实际生效。seed81、每枚
64 首 episode、提前终止关闭的严格成功数：

| checkpoint | 源 e180 | 标准继续 | 回启接触课程 |
| ---: | ---: | ---: | ---: |
| 180 | 35/64 | — | — |
| 200 | — | 18/64 | 25/64 |
| 220 | — | 24/64 | 2/64 |
| 240 | — | 31/64 | 36/64 |
| 260 | — | 30/64 | **45/64** |

严格成功数选中回启课程 e260，checkpoint SHA256
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`；
其 seed81 平均最大接触抬升0.24319m，源0.12428m。
恰比源多10/64，按预注册规则进入未见 seed 门禁：

| 未见 seed | 回启课程 e260 严格成功 | 平均最大接触抬升 |
| ---: | ---: | ---: |
| 82 | 40/64 (62.5%) | 0.20658 m |
| 83 | 44/64 (68.8%) | 0.29177 m |
| 84 | 42/64 (65.6%) | 0.19043 m |
| 合计 | 126/192 (65.6%) | — |

三个 seed 都未达到90%，所以**没有完成不同轨迹稳定抓取**。
这说明从完全自训练 s3 源延长并回启课程可得到明显更强的
候选，但比较标准继续与回启课程的因果效应仍只基于一个训练
seed和一个选择 seed，不能据此宣称普遍优势。更重要的是
seed81 各 checkpoint 表现剧烈波动，训练 reward 上升不能
替代严格评估。

训练与评估 `run_manifest.json`、`config.json`、`train.log`、
checkpoint、`eval_s{81,82,83,84}_e*_full/results.json` 均留在
`outputs/Dexplore/agent_v139_s3_{standard,backtrack}_s70_e260`；
源 e180 的 seed81 门禁留在 V1.35 Cm-off run。没有官方
actor 或外部项目修改。
