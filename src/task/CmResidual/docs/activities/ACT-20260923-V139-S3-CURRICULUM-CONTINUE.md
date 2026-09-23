# V1.39 自训练 s3 策略继续训练与接触课程回启

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- training code commit: `791381d`
- run_status: `RUNNING`
- official actor checkpoint: `null`

共同源为 V1.35 自训练 Cm-off s3 e180，源 run manifest
`COMPLETED`、checkpoint SHA256
`863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c`。
两臂均 Cm-off，不涉及官方 actor。GPU5 标准继续，GPU6
回启接触课程；先 e182 工程 smoke，之后从原 e180 各跑到
e260。评估/停损规则见 experiment card。

Cm-off 启动器原先未转发 `curriculum-backtrack`；已在
`791381d` 补齐参数配对/互斥校验、环境变量与配置记录，
`15 passed`，没有改变预注册课程。GPU5 的标准继续
`agent_v139_s3_standard_smoke_s70_e182` 与 GPU6 的课程回启
`agent_v139_s3_backtrack_smoke_s70_e182` 已从同一源 e180
SHA 启动两 epoch 接线 smoke，各自 manifest 初态 `STARTED`。

两臂 smoke manifest 均为 `COMPLETED`，日志确认同一 e180 源
与 `1e-5` 学习率；回启课程前两个 epoch 手物接触占比约
0.426/0.585，标准继续约0.173/0.331，与课程差异方向一致。
正式 `agent_v139_s3_standard_s70_e260`（GPU5）及
`agent_v139_s3_backtrack_s70_e260`（GPU6）已分别从原
e180 checkpoint 启动，预算80 epoch至 e260，保存
e200/e220/e240/e260。各 run manifest 为 `STARTED`，
不从 smoke checkpoint 继续。

正式两臂均正常完成到 e260，run manifest `COMPLETED`，
e200/e220/e240/e260 checkpoint 均存在；回启课程日志在
e190/e200/e210/e220 依次记录 0.25/0.5/0.75/1.0。
GPU 5/6 已释放并开始 seed81 完整轨迹门禁：GPU5 先评估
共同源 e180，GPU6 顺序评估回启课程臂四枚 checkpoint；
源评估完成后 GPU5 评估标准继续臂四枚。所有评估
`run_manifest.json` 记录原始策略 checkpoint SHA、输入与口径。
