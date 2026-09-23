# V1.45 固定路由重复检验

- date: `2026-09-23`
- branch: `agent/v145-route-replication`
- run_status: `COMPLETED`
- evaluation code commit: `9fa83c8`
- official actor checkpoint: `null`

按照 experiment card 固定 seeds99–103、每臂
两次、GPU5/6交叉配置。全矩阵完成后才汇总
跨 seed 指标；不使用逐 env_id“修复数”做
因果结论。

10轮/20个 run 均完成，无外部占卡或技术失败。
父运行记录在
`outputs/CmResidual/agent_v145_replication/run_manifest.json`，
分析 JSON 同目录；五 seed 路由两次总计499/640，
单专家431/640，聚类95%区间+7.66至+14.69pp。
预注册路由相对增益门槛通过，但稳定门槛全败
（最高53/64<58/64）。粗略整次运行时长中位
62.5s对60.0s；两臂均未使用 Cm 在线推理。
