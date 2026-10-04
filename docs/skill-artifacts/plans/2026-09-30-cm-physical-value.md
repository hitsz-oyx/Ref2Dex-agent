# Airplane 物理价值路线实施步骤

用户已经批准设计并明确要求开始工作。当前工作分支为 agent/cm-physical-value，目标为完成设计中的有效三臂真实策略 Probe，正向则优先正式 Validation。实现步骤由 root 在已授权范围内执行，不新增设计选择审批。

1. 实现可独立检查的保持与掉落状态机、历史窗口、后继参考上下文及候选动作合同。
2. 实现完整 episode 采集，保留真实 reward、终止原因和 reset 前后继；先工程 smoke，再按公共数据预算采集。
3. 实现动力学 ensemble、动作无关诊断、三档固定训练及基于真实完整 return 的 V/Q 预训练。
4. 将独立 teacher 的候选监督接入 PPO，环境动作与 logprob 保持原样；真实 rollout 更新 V/Q。
5. 实现 actor-only 全首回合评估及三臂初始化、配对、预算审计，执行固定矩阵。
6. 记录 Probe 判定与失败环节；正向后冻结方法并另定 Validation 矩阵。

必要检查为 episode 隔离、后继与终止、reward 相等、quaternion 处理、保持和掉落状态、reference 推进、PPO 行为合同、独立 RNG、初始化 hash 与完整评估矩阵。旧 Probe 证据与用户修改不覆盖。
