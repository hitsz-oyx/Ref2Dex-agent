# Decision Memo — 单对象任务对齐物理门失败后（2026-09-24）

Decision: 结束当前 Cm policy-utility campaign 并收敛为负结果，还是启动
明显更高成本的时序 credit-assignment 新架构？

Key evidence: 单对象 e140 actor 上，持续 grip+lift 相对 lift-only 的 H20
接触加权抬升效应为 -8.20mm，95% CI [-14.08,-2.49]mm；此前同一 option
在多对象 actor 上也仅 +0.15mm。当前 V1.3 action-aware token 不优于
action-blind/shuffled；reward、ranking、critic、auxiliary、单步/两步动作
均未给出跨训练 seed 的稳定 matched gain。已有多个有效 Probe 排除了局部路线。

Option A — 停止新增计算，保留 95.94% 单轨迹 self-trained baseline，并将
Cm policy utility 记为“尚未建立/当前配方被否定”。成本 0；下一步整理可复现
负结果和研究边界。若论文必须声称 Cm 增益，则需要修改最终 claim。

Option B — 新建时序级 Cm/option critic：以整段接触—承重结果训练 recurrent
或 trajectory-token 表示，再进行 matched on/off/placebo。预计设计实现 4–8h，
训练与 matched Probe 至少 2 GPU 数小时，且 deadline 风险高。成功才可进入
Validation；失败则回到 Option A。

AI recommendation: A。当前证据已超过继续局部探索的停止阈值；Option B
不是邻近修复，而是一条新的高成本研究项目，在现有 deadline 下缺少足够正向
物理先验。

Decision outcome: 用户于 2026-09-24 确认 **Option A**。当前 campaign
停止新增训练、仿真和模型 Probe，保留所有 checkpoint、运行 manifest 与
实验报告。`MISSION.md` 中的 Cm policy-utility 目标仍未达成；这次决策
没有把未经证实的 Cm 增益改写为已证实的 claim，也没有把局部失败外推为
“所有 Cm 都无效”。阶段证据索引见 `docs/CM_CAMPAIGN_CLOSEOUT_20260924.md`。
