# Decision Memo：验证 Cm 物理效应的短宏动作转换

当前问题是：Cm 的一步 candidate consequence 是否在现有 one-tick candidate +
nine-tick fixed-Cup 合同中被 continuation 冲掉，因此看不到任务价值。

证据显示候选 PD 命令有明显动作覆盖，单候选 value adapter 仍受高方差限制；原生
uncertainty fallback 在 one-tick 合同下 score 为负。选择一个不同的执行合同，而不是
扫描原 selector 阈值：在新的 contact states 上，Cm 只选冻结模型的 raw top candidate，
连续执行 3 tick，再接 7 tick fixed-Cup continuation；与固定 Cup 以 `p=.5` 随机 A/B。
Cm 仍只负责物理后果预测，raw top 是一次固定的动作接口测试，不是成功率预测器。

支持门为两侧各至少 48 个窗口和 20 个 episode；score 的 90% 组 bootstrap 下界须为
正，retained/contact/clearance 下界不得低于 `-0.05`。任一门失败就关闭该宏动作
配方，不扫描宏长度、seed 或 score 阈值，不启动 PPO；通过才进入小规模策略学习。
