# V1.43 首次抓握增强的配对一步反事实审计

- date: `2026-09-23`
- branch: `agent/v143-first-grip-counterfactual`
- run_status: `PLANNED`
- official actor checkpoint: `null`

固定输入、配对有效性与模型门槛见 experiment card。
评测器只增加可选记录功能；V1.40/V1.41 默认
行为不变。两臂都完成严格首 episode 后，再离线
验证干预前状态是否同一状态，绝不把不匹配状态
当成反事实样本。
