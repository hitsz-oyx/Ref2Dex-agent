# 无官方策略的单轨迹跨 seed 稳定抓取

- experiment_id: `EXP-20260923-V128-ROBUST-SINGLE-TRAJECTORY`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`

假设：由无官方 actor 的 BC 与 CmLite-reward scratch PPO 时点组成的固定整段策略路由，能在未用于规则
选择的 seeds `60–64` 上分别达到至少 `90%` 的严格抬升成功率。

结果为 `61/64、62/64、63/64、59/64、62/64`，合计 `307/320=95.94%`；五个 seed 均通过。
因此“`s1_airplane_lift` 单轨迹可在不同随机 seed 下稳定抓取”得到支持。该结论不覆盖其他 GRAB
轨迹、其他物体、MANO 手型或单一低延迟网络；Cm 的独立因果增益也仍需要 matched route ablation。

逐步 CmLite expert-action selector 在 seed59 仅 `22/64`，显著低于固定整段路由的 `58/64`；这反驳
当前一步模型可以直接逐控制步混合闭环专家的假设。

