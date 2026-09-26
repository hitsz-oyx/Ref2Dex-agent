# V1.28：无官方策略的单轨迹跨 seed 稳定抓取

- timestamp: `2026-09-23`
- branch: `agent/ppo-stability-v128`
- code commit: `318afee`
- run_status: `COMPLETED`
- official actor used for training/evaluation: `no`
- strict gate: 每个未见 seed 的 64 个首 episode 成功率均不低于 `90%`
- conclusion: `SUPPORTED`（仅限 `s1_airplane_lift` 单轨迹跨 seed；多轨迹与跨手仍未验证）

## 方法

所有 PPO expert 都来自同一条从零训练、使用 CmLite reward 的 run，不加载官方 DExplore actor。
参考动作 DAgger BC checkpoint 也明确记录 `official_policy_checkpoint: null`。固定路由只读取 episode
初始 reference frame，一整个 episode 内不切换策略：

- frame `0–20`：BC；`21`：epoch 140；`22–24`：epoch 150；
- frame `25–29`：BC；`30–31`：epoch 130；`32–34`：epoch 170；
- frame `35`：epoch 140；`36–37`：epoch 120。

路由先在 seeds `49–53` 形成第一版，再在已经揭盲的 seeds `54–58` 加入 epoch 150/170；最终规则
在形成后冻结，只在全新 seeds `60–64` 上做正式门禁。每个环境仅统计第一个完整 episode，成功定义为
物体相对初始高度 `dz >= 0.03 m` 且手物接触连续至少 5 个控制步。

## 严格验证

| seed | success | rate | mean max lift |
| ---: | ---: | ---: | ---: |
| 60 | 61/64 | 95.31% | 0.2332 m |
| 61 | 62/64 | 96.88% | 0.1965 m |
| 62 | 63/64 | 98.44% | 0.3158 m |
| 63 | 59/64 | 92.19% | 0.1502 m |
| 64 | 62/64 | 96.88% | 0.2082 m |
| pooled | **307/320** | **95.94%** | — |

五个 seed 均通过预设 `>=90%` 门禁。产物位于
`outputs/CmResidual/agent_v128_robust_router2_s{60..64}_n64`，每个目录包含 checkpoint 哈希、固定
route、逐 episode 指标和终态 manifest。

## Cm 证据边界与否定结果

- 同一 CmLite-PPO run 的 epoch 140 在 seeds `49–53` 为 `185/320=57.81%`；
  同训练 seed 的 Cm-off 所选 epoch180 在 seed49 为 `19/64=29.69%`。
  两臂 epoch 不同，seed49 参与了 checkpoint 选择，且 Cm-off 没有同样的五 seed 网格；
  这只是早期探索性正向信号，不能证明 Cm reward 的稳定因果增益。
  V1.28 选中 PPO 专家的 episode 共 141/149 成功，但它们与 BC 处理的起始帧不同，
  也不能构成 Cm-on/off 比较。
- 但当前成功系统还包含 reference-action BC，且没有做“相同路由、只移除 Cm reward”的完整消融，
  因此不能把 `95.94%` 全部因果归于 Cm。
- `318afee` 实现了 CmLite 对 BC/e120/e130/e140 动作的逐步选择；seed59 只有
  `22/64=34.38%`，而固定整段路由为 `58/64=90.62%`。一步预测逐步拼接不同闭环策略会破坏时序一致性，
  该在线仲裁假设被反驳，产物为 `agent_v128_cmlite_ensemble_s59_n64`。

## 延迟边界

固定路由在单环境部署时每个控制步只执行被选中的一个 actor；路由是一次整数 frame 查表，不做 Cm
在线推理，也不同时前向所有 experts。因此控制环的推理路径与单个 BC/PPO actor 同阶。当前实现启动时仍
加载全部 checkpoint，内存与冷启动尚未优化；蒸馏成单 actor 是后续多轨迹阶段的工程目标。
