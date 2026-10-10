# Trajectory policy

用户 [ref1](docs/user/ref/ref1.md) 指定的新架构研究边界：纯测量历史H→独立高层
trajectory actor→latent c→24步11点手轨迹→每步读取实时状态的executor→native command。
旧behavior只提供初始化和可衰减训练先验，不作为永久在线动作基座；后续检验
PointWorld动作条件物理表征是否改善真实RL策略训练。

当前阶段：固定48维轨迹decoder首轮覆盖Probe已完成，局部UNPROMISING。
密集几何轨迹4/4长时终末抓持；四节点重建两组均0/4终末持有，输入/FF/命令审计通过。
最优四节点拟合仍有最差窗口32mm位置/31mm FF RMS，证实这种时间形状难以近似
当前可执行前缀。固定PCA D48把手点RMS降到5.61mm，但启动掌部max23.62mm/
FF RMS18.50mm未过几何screen，按协议未仿真；不作为新的抓持负证据。
下一步让decoder拟合目标直接约束前缀手几何与腕部速度，
尚未训练高层actor或接入WM。consequence-evaluator后续实验按用户
要求暂停；保留其实现、修复和运行证据。继承根级Mission/Campaign/AGENTS，包括
无真实未来q/物体参考、phase/clock或触觉策略输入；不新建分支、不push。

- [架构建议与最小验证路径](docs/research/20261010-ref1-architecture.md)
- [RLT/PPO/chunk原始来源核对](docs/research/20261010-trajectory-policy-source-check.md)
- [48维decoder覆盖Probe](docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md)
- [固定D的前缀最优拟合诊断](docs/experiments/probes/P-20261010-decoder-prefix-fitting.md)
- [固定PCA时间基底覆盖](docs/experiments/probes/P-20261010-lowrank-trajectory-decoder.md)

本Task成功仍需自训练操纵策略与matched Cm-on/off训练收益，不能由decoder重建、
单个成功视频或冻结控制器收益替代。具体Probe先写Task-local实验卡，再运行。
