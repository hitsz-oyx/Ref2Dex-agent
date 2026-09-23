# V1.39: 自训练 s3 策略的继续训练与接触课程回启

- experiment_id: `EXP-20260923-V139-S3-CURRICULUM-CONTINUE`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
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
