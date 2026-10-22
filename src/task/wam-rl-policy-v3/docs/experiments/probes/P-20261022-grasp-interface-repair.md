---
schema: ref2dex.probe.v2
probe_id: P-20261022-grasp-interface-repair
experiment_id: P-20261022-grasp-interface-repair
date: 2026-10-22
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-grasp-interface-repair
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259]
decision_changed_if_positive: retain precise servo and reachable consumption; same-R ablation decides whether tau content helps on the fixed panel
decision_changed_if_negative: keep RL HOLD and repair measured control; do not claim stable engineering or pretrained tau benefit
status: RUNNING
run_id: grasp-v10-stat40-20261022-r1
---

# v10 修复与同 R 的 n40 验收

## 结果解读表

| 每子集每臂40次结果 | 结论 | 下一步决定 |
|---|---|---|
| A1 seen≥32/40且显著优于A2、CI不重叠 | 精确控制目标通过；本面板τ内容有贡献 | 保留消费链，完整任务/RL资格仍另验 |
| A1 seen≥32/40但A2等效 | 控制改善，τ独有收益不成立 | 使用更简单控制，不宣传预训练贡献 |
| 限位下降但仍不能抓住/到位 | 安全改进不足以建立未见资格 | 保留失败，修抓姿/闭合；RL HOLD |
| seen仍<32/40 | 未达到机主80%目标 | 明确失败和剩余控制阻塞，不称完成 |

Result: PENDING frozen 160 episodes, no training updates.
Decision: RL HOLD; no v6 retraining.

完整方法、开发失败、diff与复现：[修复报告](../../grasp-interface-v10-repair-20261022.md)。

这是 Blocker：先修真实控制资格，才能开始样本效率RL。最便宜方法为已有轨迹定位+有上限工程触发，再对冻结R/WAM做零更新物理复验。

严格终点沿用v9：600 controls/20秒末S2c当前hold（接触耦合、抬升≥5cm、离桌连续≥1秒），XYZ误差≤15mm、无物理中止。HOLD无release。真实安全阈值不改、失败全部入分母。
A2同R/相位/伺服，仅全部24帧τ位移置零；valid在共同状态不变，keypose也零内容，禁止R旁路。初态按v9配对原seed，seen40与原未见面板40分别报Wilson/Fisher、全量持握/限位/误差分布。八物体已用于诊断，不再声称本轮是新未见holdout。预注册与全部结果在`outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/`。
