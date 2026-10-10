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
最终pose/FF metric D48离线手点RMS0.919mm、掌部max4.386mm、FF RMS2.805mm通过，
但原生两组长时终末held为2/4、0/4；GT与dense各4/4。8行都先抓住，6行随后失持。
实际live q/object重编码下前缀手点RMS8.017mm；输入/命令/PD与独立实现审查通过。
按预声明停止48D重建搜索，改用保留24步独立腕姿/手指的直接D288，先核对既有dense
执行轨迹的接口等价，再进入H→c初始化及真实奖励RL。
`457c65f` D288工程回放通过68窗口x4条dense计划：native q max5.96e-7、
手点3.58e-7m、速度1.97e-5、腕FF2.35e-6；GPU4约0.975s、Torch峰值6.85MiB。
这是同一既有成功计划的编码/解码一致性，未新增物理rollout或学习证据。
已补充H328纯测量历史合同、独立Gaussian actor与有界监督初始化入口
`tools/run/fit_history_actor.py`：全部源轨迹保留，train-only normalization，按行留出；
首轮6504样本BC初始化2500updates完成14s；留出目标比常量低80%，但实际纯H两组
均0/4稳定抓持（GT4/4、dense3/4），启动手点RMS56.36mm。实际H/c/D/native链审计
通过，局部UNPROMISING；当时尚未PPO训练或接入WM。后续startup-balanced2500update/13s把启动手点RMS降到7.395mm，但掌部max15.61mm，
仍未过5/10mm初始化screen，按协议未仿真。停止BC权重搜索，下一步固定D/R做真实奖励
trajectory PPO；balanced500仅作为未验证warm start，不作为成功策略。
首轮真实任务PPO `4da53a9` 已完成24updates/48992交互/264s，训练采样曾有
486帧且终末持有，但最终冻结比较GT/dense各3/4，warm/PPO均0/4形成抓持。
全H、概率/价值/奖励/GAE、随机c物理解码及实际执行链审计通过，局部UNPROMISING；
尚未形成可用轨迹baseline或WM收益。已有rollout显示短credit导致长held样本启动
优势全负；固定快照lambda1回放把实际启动样本概率提升从1/7变5/7，KL受控。
lambda1固定预算Probe已完成265s/48992交互，9/12长held探索行启动Apositive，
但冻结warm/final仍各0/4，GT/dense各3/4；全链审计通过，局部UNPROMISING。
固定同H接触前XYZ mean改动<=.03893mm（per-coordinate RMS），探索std1mm。
下一步先冻结随机采样warm/final比较，分辨随机部署收益与偶然探索；尚未启动。
不把offline信用或训练reward改善当作抓取收益，暂不再改训练或接WM。consequence-evaluator后续实验按用户
要求暂停；保留其实现、修复和运行证据。继承根级Mission/Campaign/AGENTS，包括
无真实未来q/物体参考、phase/clock或触觉策略输入；不新建分支、不push。

- [架构建议与最小验证路径](docs/research/20261010-ref1-architecture.md)
- [RLT/PPO/chunk原始来源核对](docs/research/20261010-trajectory-policy-source-check.md)
- [48维decoder覆盖Probe](docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md)
- [固定D的前缀最优拟合诊断](docs/experiments/probes/P-20261010-decoder-prefix-fitting.md)
- [固定PCA时间基底覆盖](docs/experiments/probes/P-20261010-lowrank-trajectory-decoder.md)
- [pose/FF metric覆盖与停止48D搜索](docs/experiments/probes/P-20261010-metric-trajectory-decoder.md)
- [独立H→c初始化与完整执行](docs/experiments/probes/P-20261010-history-trajectory-actor.md)
- [启动平衡初始化与转入RL的决策](docs/experiments/probes/P-20261010-startup-balanced-actor.md)
- [首次真实任务PPO及冻结比较](docs/experiments/probes/P-20261010-trajectory-ppo.md)
- [只改信用时间尺度的lambda1任务Probe](docs/experiments/probes/P-20261010-trajectory-ppo-long-credit.md)

本Task成功仍需自训练操纵策略与matched Cm-on/off训练收益，不能由decoder重建、
单个成功视频或冻结控制器收益替代。具体Probe先写Task-local实验卡，再运行。
