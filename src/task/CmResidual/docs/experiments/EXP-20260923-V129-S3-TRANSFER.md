# V1.29: s3 轨迹迁移与 CmLite 适配

- experiment_id: `EXP-20260923-V129-S3-TRANSFER`
- branch: `agent/multitrajectory-v129`
- code_commit: `557702bd41738fee24006d949575bb136df99711`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（此从零 s3 PPO 配方与预设 checkpoint 网格）
- official actor checkpoint used: `no`

## 问题与预注册门禁

V1.28 只证明 `s1_airplane_lift` 的跨 seed 抓取。本实验先看第二条 GRAB
轨迹 `s3_airplane_lift`：单独从零训练的策略能否抬升，以及在相同 PPO
配置下，适配后的 CmLite 奖励能否超过 Cm-off。独立 seed 的严格成功定义
保持不变：物体相对初始位置抬升至少 3 cm，且手物接触连续至少 5 个控制步；
每个并行环境只统计首个完整 episode。与 V1.28 固定路由评估一致，
正式门禁关闭提前终止，让 episode 运行到轨迹自然终点；评估 JSON 显式记录
`early_termination_disabled`。原有 s3 PPO 诊断保留默认提前终止，两者
不能直接比较成功率。

评估口径修订于两条 seed-70 训练运行期间、任何 seed-71 checkpoint
评估之前；修订原因是审计发现 V1.28 路由评估显式关闭提前终止。

训练 seed 固定为 70，每臂 64 env、horizon 32、200 epoch，除 CmLite
奖励外，其余几何奖励与课程参数相同。先检查 epoch 100、140、180、200
在 seed 71 上的 64 个 episode；按严格成功数选各臂 checkpoint，平局依次
比较平均最大接触抬升和较早 epoch。若两个臂的候选均为 0/64，判定当前
训练配方未通过继续扩大的门禁。否则冻结选择，在未见 seeds 72–74 上
各评估 64 个 episode。多轨迹稳定抓取要求 s3 和既有 s1 在每个未见 seed
均至少 90%；Cm 的收益要求同样评估预算下 Cm-on 优于 Cm-off。一个训练
seed 的比较只能作为机制线索，不能单独证明总体因果收益。

## 输入与已完成诊断

重建输入 `outputs/CmResidual/agent_v129_s3_coordfix/corrected_converted_r2`
由 raw GRAB 的 `s3_airplane_lift` 和本地 DExplore converter 形成，
`corrected_manifest_r2.json` 记录源文件哈希与右手/物体相对坐标修正。修正
后的参考 tensor 与既有几何导出的右手/物体相对位置平均绝对差为
`3.24e-8 m`；接触帧最近右手关键点到物体中心的平均距离 s3 为
`0.0444 m`，s1 为 `0.0406 m`。这些只检查输入几何，不证明仿真可抓。

旧的、只见过 s1 DAgger 转移的 CmLite
`outputs/CmLite/V1.27/train_dagger_s5910_5929_val_s5909/best.pt`
在 s3 epoch-130 策略真实转移上：运动 EPE `3.537 mm`，零位移基线
`2.363 mm`，接触 precision `0.057`。所以旧 Cm 在 s3 上的奖励预测
失准，不能拿训练 reward 当抓取证据。

旧 CmLite 奖励的 s3 从零 PPO `agent_v129_s3_cmlite_s66_e200` 完成
200 epoch。在**默认提前终止开启**的独立 seed 67 评估中，epoch
50、100、130、170、200 均为严格 `0/64`；epoch 100 的平均接触
占比 `0.19155`，epoch 200 为 `0.10561`。这项诊断尚未通过与
V1.28 相同的无提前终止口径复核，不能据此断言完整轨迹下同样失败。
其中 epoch 100 的 64 个 episode 全部由提前终止结束，37 个在参考
接触帧 50 前结束，进一步说明需要重评。
默认提前终止下的参考关节动作 lead 1 对照也是 `0/64`、无手物接触。
该对照不能证明环境物理不可抓，因为 episode 可能在接触前结束，且
s1 的同类参考动作也未直接成功。

## 新 CmLite 离线门禁

将手腕世界位置换成相对物体的位置，并保留原 49D 宽度、107,012 参数
的 `relative_wrist_v1` 输入。旧 checkpoint 默认继续使用 `absolute_v1`。
训练数据为自训练 s1 DAgger 转移 seeds 5910/5929，加 s3 自训练 PPO
epoch 130 seed 67、epoch 100 seed 68 的真实转移。s1 seed 5909 与
s3 seed 69 留出；两种 feature mode 使用完全相同的数据和训练超参。

在 s3 seed 69 的**首个 episode 转移**上，绝对版/相对版的运动 EPE
分别为 `1.787/1.739 mm`，零位移基线 `4.428 mm`；接触 F1 分别为
`0.977/0.979`。s1 seed 5909 的运动 EPE 分别为 `2.059/2.132 mm`，
零位移基线 `3.311 mm`。相对版在这两个留出集没有一致优势，后续 PPO
使用它是检验跨场景平移表征的假设，而非已证实改进。
在 s3 seed 69 的 1226 个首 episode 运动转移上，固定状态并以 seed 129
打乱动作后，相对版 EPE 从 `1.739` 上升到 `3.752 mm`，预测向量平均
改变 `2.971 mm`；这支持模型实际利用了动作输入，但不等于在线奖励有效。

面对未参与 Cm 训练的 s3 epoch-170 策略，绝对版/相对版运动 EPE 为
`2.294/2.281 mm`，零位移基线 `4.838 mm`；该样本只有 9 个真实
接触，precision 仅 `0.200/0.174`。在线奖励仍有误导风险，尤其是
策略分布迁移到新状态时。

注意：导出的完整 transition tensor 包含其他环境在等待最慢环境期间的
重复 episode。上述 seed 69 门禁显式用 `done` 的逐环境前缀筛选首个
episode（2233/9280 条转移），避免把重复 rollout 算作独立样本。

## 成对 PPO 终态

- `agent_v129_s3_relcm_s70_e200`：GPU 5，CmLite
  `b7aa7630e31c820802cb95d81c490d4f27be8e74c1c9e1e400b20fcd849a9a38`。
- `agent_v129_s3_cmoff_s70_e200`：GPU 6，匹配的 Cm-off 对照。
- 两臂均由同一 launcher 产生 `run_manifest.json`、`config.json`、
  `train.log` 和每 10 epoch checkpoint；均正常完成 200 epoch。3-epoch
  接线小测也分别完成。

seed 71、每个 checkpoint 64 个首 episode、提前终止关闭的严格结果：

| epoch | 相对 Cm-on | Cm-off | Cm-on 平均最大接触抬升 | Cm-off 平均最大接触抬升 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0/64 | 0/64 | 0.00045 m | 0.00050 m |
| 140 | 0/64 | 0/64 | 0 m | 0.00080 m |
| 180 | 0/64 | 0/64 | 0.00034 m | 0.00024 m |
| 200 | 0/64 | 0/64 | 0.00027 m | 0 m |

两臂所有预定候选均为 0/64，触发预注册停损，不再用 seeds 72–74
扩大此从零训练配方。旧 s3 CmLite run 的 epoch 100/200 用相同完整轨迹
口径复核，也均为 0/64。因此“离线预测改善会让此 scratch PPO 配方抓取
s3”被反驳；不能据此否定 Cm 的其他使用方式。

## 下一条线索

将完全由本项目从零训练、在 s1 达到 35/64 的 PPO epoch-140 actor
直接放到 s3（不更新参数），seed 71 完整轨迹得到 **2/64** 严格成功、
19/64 出现至少 1 cm 接触抬升，平均最大接触抬升 `0.01057 m`。
相同 s1 DAgger BC 在 s3 为 0/64，接触占比 `0.00367`。这表明 s3
物理环境存在可抬升状态，值得以该自训练 PPO 为起点进行成对微调；
结果不支持直接宣称跨轨迹稳定抓取。
