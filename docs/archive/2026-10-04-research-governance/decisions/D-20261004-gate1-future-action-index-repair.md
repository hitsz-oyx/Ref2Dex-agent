# Gate 1 future-action index repair

**当前需要决定的问题。** 之前使用 fast assembler 生成的 `future_action` 是否真的
对应同一 episode 的决策动作，因而能否作为 `V_HF`/`V_HFEI` 对照。

**关键证据。** fast assembler 曾以 episode-local `future_pos` 直接索引全局交错的
`merged["action"]`。这在多环境 shard 中会把 future action 取自其他 episode；旧
future-action control 结果因此标记为 `INVALID_IMPLEMENTATION`。direct `V_H`/`V_HEI`
没有读取该字段，不受此 bug 影响。

**root 选择的行动及理由。** 在 commit `fcfde1c` 中改为
`merged["action"][idx[future_pos]]`：`idx` 保留 episode-local 到 shard-global 的
映射，legacy post-step shift 只作用于物理量，不作用于决策动作。同时加入交错 episode
的 slow/reference 与 fast/vectorized 回归测试，覆盖 pre-step 和 legacy timing。

**验证。** 两种 timing 的 slow/fast tensor contract 测试均通过（2 tests passed）。
重新组装的 e260 h16/h32 与 e420 h32 base/interaction audits 均为 `errors=[]`；
interaction 维度为 80→119，row key、target 和 episode split 不变。

**修复后的诊断结果。** 这些仍是探索性 control，不是正式 Validation：

| 数据 / horizon | `V_HFEI` vs `V_HF` 点估计（五个 seed） | episode CI 排除零 | source-run cluster CI 排除零 |
| --- | --- | ---: | ---: |
| e260 / h16 | `+19.1,-0.6,+14.0,+14.8,+15.3%` | 4/5 | 4/5 |
| e260 / h32 | `+29.0,+17.4,+18.9,+31.2,+33.8%` | 5/5 | 5/5 |
| e420 / h32 | `+3.9,+6.6,+16.8,+4.8,+12.0%` | 3/5 | 4/5 |

随后又完成 deterministic quaternion-sign canonicalization；h32 的最终 qfix direct /
control 结果与边界见 [quaternion-sign repair memo](D-20261004-gate1-quaternion-sign-repair.md)。
因此 future-action 修复没有改变 e260 h32 的 local `PROMISING` 边界，也没有把 e420
h32 变成稳定跨 actor 证据。旧 control JSON 和旧文档中的 future-action 数字不再作为
证据；最新 qfix artifacts 位于 `tmp/gate1_split_rng/*qfix*ctrl*.json` 及对应 cluster
bootstrap JSON。正式 Validation 仍需预注册 actor/outcome cluster、success/drop 覆盖和
固定表示合同；不启动 Cm。
