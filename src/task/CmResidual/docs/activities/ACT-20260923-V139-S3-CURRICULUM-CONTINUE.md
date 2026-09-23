# V1.39 自训练 s3 策略继续训练与接触课程回启

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

共同源为 V1.35 自训练 Cm-off s3 e180，源 run manifest
`COMPLETED`、checkpoint SHA256
`863443513a746155f2b04662fe4dddd8c7b5a672d3e73533c1ff555d90c0237c`。
两臂均 Cm-off，不涉及官方 actor。GPU5 标准继续，GPU6
回启接触课程；先 e182 工程 smoke，之后从原 e180 各跑到
e260。评估/停损规则见 experiment card。
