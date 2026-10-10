# Trajectory policy

用户 [ref1](docs/user/ref/ref1.md) 指定的新架构研究边界：纯测量历史H→独立高层
trajectory actor→latent c→24步11点手轨迹→每步读取实时状态的executor→native command。
旧behavior只提供初始化和可衰减训练先验，不作为永久在线动作基座；后续检验
PointWorld动作条件物理表征是否改善真实RL策略训练。

当前阶段：已实现固定48维轨迹decoder，微型合同测试通过；准备一次D/R覆盖Probe，
尚未训练高层actor或接入WM。consequence-evaluator后续实验按用户
要求暂停；保留其实现、修复和运行证据。继承根级Mission/Campaign/AGENTS，包括
无真实未来q/物体参考、phase/clock或触觉策略输入；不新建分支、不push。

- [架构建议与最小验证路径](docs/research/20261010-ref1-architecture.md)
- [RLT/PPO/chunk原始来源核对](docs/research/20261010-trajectory-policy-source-check.md)
- [48维decoder覆盖Probe](docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md)

本Task成功仍需自训练操纵策略与matched Cm-on/off训练收益，不能由decoder重建、
单个成功视频或冻结控制器收益替代。具体Probe先写Task-local实验卡，再运行。
