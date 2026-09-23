# V1.43 首次抓握增强的配对一步反事实审计

- date: `2026-09-23`
- branch: `agent/v143-first-grip-counterfactual`
- run_status: `COMPLETED`
- evaluation code commit: `250efd8`
- official actor checkpoint: `null`

固定输入、配对有效性与模型门槛见 experiment card。
评测器只增加可选记录功能；V1.40/V1.41 默认
行为不变。两臂都完成严格首 episode 后，再离线
验证干预前状态是否同一状态，绝不把不匹配状态
当成反事实样本。

seed97 off/always 并发GPU5/6，seed98 为控制
跨卡因素在GPU5顺序执行。四个 run 均 `COMPLETED`
且每臂首次可干预记录64/64。配对审计 JSON
`outputs/CmLite/V1.43/audit_first_grip_s97_98.json`
给出严格同状态 seed97仅1/64、seed98为0/64，
低于预注册最低样本数，判 `INCONCLUSIVE_PRESTATE_MISMATCH`。
不从唯一有效对解释模型排序；下一步需测同策略
同 seed 重跑的一致性。
