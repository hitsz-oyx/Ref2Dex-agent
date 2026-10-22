| 结果解读表：读数 / 可能结果 | 含义 | 决定 |
|---|---|---|
| 固定 τ 面板前 8 帧 vs 旧 B1（9.357826 mm），母序列聚类 paired CI | 短期预测变化；B1 是旧配方参考，多项条件来源同时变化，不能归因为事件消融 | 改善且 contact / overall 无明确退化才进入执行器验证；跨 0 则不宣称优势 |
| contact window vs 旧 B1（19.834665 mm），同一固定面板 / 聚类 CI | 接触附近预测是否改善，不能单独代表 approach 或任务成功 | 与前 8 帧、整体联合判断，不以 contact 单项放行 |
| 整体 vs 旧 B1（23.409973 mm），同一固定面板 / 聚类 CI | 完整预测时域的可比读数；旧 B1 仅作旧配方参考 | 不新增基线；保留分阶段结果，尤其 approach（B1 29.897742 mm） |
| 执行器内实际任务进展 / 完成率改善、持平或退化 | 模型误差是否转化为任务进展，τ 读数本身不保证成功 | 改善才支持下一阶段；持平或退化不宣称部署收益；本轮不新开执行器实验 |
| 训练发生 OOM / 非有限 loss 或 gradient / 磁盘告警 | 运行或基础设施失败 | 停止并报告，保留现场；不静默重启、不换参数、不消耗第二次训练 |
| 任一输入、实现、资源 preflight 失败 | 尚未具备按冻结配方启动的条件 | 训练 0 updates，列出最小修复，不以旧 endpoint 或 GT 事件填坑 |

## 冻结配方（先于代码与启动落盘）

Experiment: `P-20261023-tau-event-free`；run_id: `tau-event-free-20261023-r1`。
承接 brief31 / brief32 的冻结结论，不重复信号普查。Gate A 已失败，默认选择训练与部署统一 unknown/invalid 事件。该选择不把 unknown 当作已知 NONE，不删除或伪造调度器计划、反馈状态与执行器控制信号。

| 事件概念 | 真正模型张量与改动 | 不变项 |
|---|---|---|
| type | `event_type`: int64，占位 0，必须同时 `event_valid=False` | 原三类编码与参数形状保留；0 不表示已知 NONE |
| delta30 | `event_delta_steps`: float32，统一 0 | 30 Hz 时间基准与 K24 不变 |
| valid | `event_valid`: bool，统一 False；模型五维 event features / event residual 全零 | 沿用 `event_features` → `EventGoalResidual` 的 invalid mask |
| known-NONE | `valid & (type == 0)` 恒 False | 不增加一个假“无事件”的有效输入 |
| recovery | 不是独立模型输入列；不能通过事件包恢复有效状态 | 调度器 recovery 控制逻辑不变，不用 GT 补 runtime 状态 |
| release | 不再有有效 RELEASE 输入；type 的 2 编码保留 | 释放控制与接触预测监督不变，不移除未来真值标签 |

训练与推理必须使用同一 unknown 包常量及同一模型 invalid mask；仅配置声称 unknown、仅训练随机 dropout、或仅运行时默认缺省事件，均不构成合同通过。历史 checkpoint 未声明此策略时保持历史配方身份。新配方须在真实 collate 后与部署入口证明事件包一致。

完整保留 curated r7 的 **6,000** 窗口与其有序 `(source_id, sample_id, window_start_frame)` 清单；母序列、物体、阶段组成及 sampler 不作为本轮研究变量。保留手物轨迹监督、contact geometry loss 与接触预测目标、模型架构及执行器接口。goal 保持 `spatial_path_waypoint`，不能将观测未来 endpoint 作为 waypoint 冒名输入。原 `tau_curated_data_only_v1.json` 的 status 不变。

旧 B1 只作旧配方参考。新配方改变事件条件来源，不能把新旧模型误差变化解释为“事件消融的因果效应”。source-ceiling A0 不消费事件包，其测量语义不因本选择改变。

## Preflight 与启动记录

状态：`PREFLIGHT_PENDING`，训练 **0 updates**。冻结 HEAD、输入 / 配方身份、实测资源、checkpoint 增长及预算、测试与启动裁决将在审计 receipt 中填写。本文件先于任何配方实现写入。

GPU 7 属执行器，4/5/6 属 mutant 审计；不能利用占用间隙抢卡。必须核实空闲且可用资源；全满则排队。磁盘必须实测，不删除历史产物。训练时若触发预先冻结的内存、磁盘或数值阈值，停止并报告，不自动恢复。

## 复现与写盘

待 preflight 工具与 receipt 冻结后填写。训练启动证据尚不存在，不能把配方冻结或测试通过表述为已经开训。
