# V1.35 适配 CmLite 的 s3 在线成对微调

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

源为本地自训练 Cm-off s3 e160 actor，源 run manifest
`COMPLETED`，checkpoint SHA256
`ce62d6efb0c9a200583635eb942b45a07d972cfae098b8edb103e805a82cb8f1`。
新 CmLite 为 V1.34 平衡模型，SHA256
`1146025ca88b35f39a5fad7f5f899a88cca34b774fa9abe0c0d6f5d72519e1b8`。
两臂训练 seed70、s3 重建输入、64 env/horizon32/minibatch256；
Cm-on GPU5、Cm-off GPU6，各先 e162 smoke，再从同一源正式
训练至 e180。下一步评估 seed77 的 e170/e180 网格；详情与
停损规则见 experiment card。
