# V1.30: 自训练 s1 策略到 s3 的成对微调

- experiment_id: `EXP-20260923-V130-S1-TO-S3-FINETUNE`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
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
