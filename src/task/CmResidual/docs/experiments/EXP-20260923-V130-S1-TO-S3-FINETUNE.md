# V1.30: 自训练 s1 策略到 s3 的成对微调

- experiment_id: `EXP-20260923-V130-S1-TO-S3-FINETUNE`
- branch: `agent/multitrajectory-v129`
- run_status: `COMPLETED`
- conclusion: `REFUTED` (本配置下 CmLite 奖励的增益假设)
- official actor checkpoint used: `no`

## 假设与冻结设计

V1.29 的 s3 从零 PPO 两臂在完整轨迹门禁均为 0/64，但 s1 自训练 PPO
epoch 140 直接迁移 s3 可达到 2/64，且 19/64 出现至少 1 cm 接触抬升。
因此测试：从同一 s1 自训练 checkpoint 出发，在 s3 上继续 20 epoch，
相对坐标 CmLite 奖励是否比 Cm-off 更能保留并改善该迁移能力。

源 checkpoint 为 `agent_v125_norm_cmlite_anneal_s45_e200` 的
`GRAB_00000140.pth`，SHA256
`5d1f50a21409df5a09f0a471d115d2f06eedce6bc53855a81acf3cc268e6e1a7`。
源 manifest 为 `COMPLETED`，未使用官方 actor；恢复必须同时匹配绝对路径
和 SHA256，绝不从目录中自动选择“最新”checkpoint。

两臂固定同一源策略、s3 `corrected_manifest_r2.json`、训练 seed 70、
64 env、horizon 32、同样的几何奖励和接触课程。学习率固定 `1e-5`，
从源 epoch 140 继续到 epoch 160，保存 150/160。Cm-on 使用 V1.29
相对手腕 CmLite checkpoint
`b7aa7630e31c820802cb95d81c490d4f27be8e74c1c9e1e400b20fcd849a9a38`；
Cm-off 仅移除该奖励。GPU 5/6 各运行一臂，上限两卡。

先做每臂 epoch 142 的接线小测，检查 checkpoint/RMS/optimizer 是否
按同一源恢复。正式 20 epoch 后，在独立 seed 71 关闭提前终止，评估
源 e140 和两臂 e150/e160 的 64 个首 episode。若至少一臂严格成功
超过源策略的 2/64，或平均最大接触抬升超过源策略 `0.01057 m`
的 1.25 倍，再考虑延长到 epoch 200；否则停损并回到轨迹/动作表征。
即使 pilot 通过，跨轨迹稳定抓取仍需未见 seeds 72–74 每个至少
90% 的独立门禁。

此 matched resume 比较能检验“在同一初始化上加入 Cm 奖励”的局部效果；
CmLite 预训练额外使用了 s3 转移，故最终因果表述必须保留这项数据来源。

## 结果与判定

两臂均从同一 e140 源准确恢复并完成到 e160。原定的 seed 71
完整轨迹门禁（提前终止关闭、64 个环境的第一个 episode）中，两臂
均通过继续训练的 pilot：Cm-on e150 为 0/64、e160 为 13/64；
Cm-off e150 为 5/64、e160 为 28/64。随后冻结 e160 checkpoint
在未见 seeds 72、73 做配对复评：

| seed | Cm-on e160 | Cm-off e160 | Cm-on 平均最大接触抬升 | Cm-off 平均最大接触抬升 |
| --- | ---: | ---: | ---: | ---: |
| 71 | 13/64 | 28/64 | 0.05827 m | 0.07901 m |
| 72 | 7/64 | 30/64 | 0.04049 m | 0.06730 m |
| 73 | 10/64 | 25/64 | 0.02309 m | 0.06577 m |
| 合计 | 30/192 (15.6%) | 83/192 (43.2%) | — | — |

严格成功定义为物体至少抬高 3 cm 且手与物体接触连续至少 5 步。
这三个 seed 上 Cm-on 均低于 Cm-off，且离跨轨迹每 seed ≥90%
的目标很远。因此，**当前相对手腕 CmLite + 系数 5 的目标进展奖励
不支持“Cm 促进 s3 迁移”**；不能以两臂相对未微调源策略 2/64
均有提升，代替匹配对照的结论。这里仅否定这一模型、奖励与微调配方，
不否定 Cm 世界模型的预测能力或其他用法。

结果位于两臂 `outputs/Dexplore/agent_v130_s3_transfer_{relcm,cmoff}_s70_e160`
的 `eval_s{71,72,73}_e160_full/results.json` 和各自的
`run_manifest.json`。当前不延长 Cm-on 同一奖励到 e200；下一步先
排查源 checkpoint 的可迁移性，再另行预注册 Cm 接入方式。
