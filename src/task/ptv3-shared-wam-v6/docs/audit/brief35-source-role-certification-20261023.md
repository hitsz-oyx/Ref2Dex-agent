| 可能结果 | 含义 | 决定 |
|---|---|---|
| 6,000 全可认证：先存在 source artifact + role/reset 绑定 | goal 是合法部署前输入 | 构造 cache/receipt 并报告认证；本轮明确约束只做认证、训练 0 updates，后续启动另按授权执行 |
| 部分可认证，覆盖 C 较大（如 ≥80%） | 条件合法性与覆盖度冲突 | 报 C、缺失构成、按 C 训练 / 补记录两条量化代价，等机主裁决 |
| 部分可认证但 C 很小（<50%） | 现有录制不足以支撑合法条件训练 | 不训练，报补记录清单与规模，等裁决 |
| 全部确证来自对应 rollout 自身未来 | 现有 goal 条件非法 | 不训练；统一诊断缺部署前意图绑定；仅列 unknown-goal 零填充为明确配方变更选项，不启用 |
| 无法判定：缺 artifact / role / reset 等证据 | 证据链不足，不能将“不确定”当“非法”或“合法” | 报逐窗口缺项与最小补证清单，不改生产代码 |

## 预先冻结的认证口径

Experiment `P-20261023-source-role-certification`，run `brief35-source-role-20261023-r1`。
本轮为 Blocker / Decision 审计：区分已知参考计划与对应执行自身未来，决定能否构造合法条件 cache。最便宜方法是沿 r7 有序 6,000 元组追查已绑定 metadata、producer、原始 artifact 和执行记录，不跑模型或仿真。

不能把 r9 的“纯观测历史不可恢复指令”外推成“所有参考轨迹不合法”。旧录制可以成为新执行的事前计划，但必须证明它与目标执行的来源角色、内容及 reset 绑定；文件存在 / mtime 更早 / 同名对象 / producer 推断都不足以补上该绑定。当前 reader 使用未来 endpoint 的代码事实与“是否存在合法候选 waypoint 来源”分开记账。

合法证据至少包含：窗口目标执行身份、source artifact 内容身份与可读引用、明确 source_role、source 对目标执行独立的 provenance、source 冻结/声明于目标 reset 之前的证据、引用该 source 的 waypoint 派生链。确证未来泄露须有对应目标执行的同身份未来读取证据；缺一条合法证据不能自动判成泄露。

训练 0 updates，无 GPU；写盘上限 1 GiB，预计审计 manifest / CSV / receipt 不超过 64 MiB，写前实测空间；不改 production、配方、goal_missing_policy 或 config status。不新增采集，不删除历史产物。逐窗口结果与聚合数字待本次全量追查后填写。
