# D-20261001-native-player-collector-repair

日期：2026-10-01；owner：root；分类：Blocker（科研采集器的工程合同）。

**问题与证据。** 当前策略 V 诊断尚无真实转移。R1/R2 被资产 cwd 和预算包装器阻塞；R3 的真实环境初始化暴露 `reward_shaper` 从配置字典变成运行时对象的问题，R3b 在首个 Gaussian draw 前失败。只读对照本机 rl_games `ModelA2CContinuousLogStd.Network.forward` 发现：原生路径先 `norm_obs`，网络返回 `logstd` 后取 exp，再采样；现采集器直接调用 raw a2c_network，误把 logstd 当 sigma，critic-only 路径也遗漏相同归一化。旧简化 mock 没覆盖此 API，CPU 合约验收不等于真实 player 接线已证。

**行动与理由。** 停止本轮额外仿真，保留所有零数据失败和变更 hash。由固定 agent_rl CPU-only 修复原生 player actor/critic 路径，对实际安装的 rl_games wrapper 做等价与反例检查，包括非恒等 running statistics、负 logstd、同 RNG 的一次采样、critic-only 不消耗 RNG、裸 tensor/字典观测；不改模型、reward、seed、动作裁剪或实验条件。以真实共用循环测试补充原 mock，先修复采集器，而非据此诊断 V 或调整 HF08。

**成本与下一步。** CPU 修复上限 2 线程/15 分钟/0.1 GiB，无 GPU/Isaac/新转移。原恢复轮绝对截止 epoch1790836597 不变，原运行 manifest 和旧 SHA 保留。验收后依据 owner 的实际累计资源证据，另行选择最多 10 分钟、1 GPU 的冻结面板采集；不得沿用重置计时器。原 1200 秒诊断尝试按终态关闭，新的工程恢复授权须显式记累计时间；所有诊断数据/cache/log合计仍限 2 GiB，整个诊断累计上限仍服从 CAMPAIGN 的 60 分钟 Probe 范围。输入漂移、非有限、采样/RNG/归一化不等价或资源冲突停止。

**边界。** 本轮失败均未产生有效科学 Probe，不构成 V 不充分或 Cm 负效应的证据。HF08 UNPROMISING/PAUSED、utility slot1/1和 Cm policy utility OPEN 保持。不重训、不扫参数、不升级 Validation，不改变研究问题/claim，不突破 CAMPAIGN，不需新增外部授权。

**终态审查补充。** R3/R3b自有进程均已退出，80项报告内声明artifact匹配；0数据行。报告/终态card字段仍用R3b追加前hash（6690字节），owner实际最终card为7672字节，两者不符，因此暂不导入card，也不接受整份manifest无条件完整性声明；精确MD/JSON仅作为失败审计归档，待原owner补正最终card/hash。R3/R3b实际GPU环境初始化分别约32.66/32.57秒。报告“checkpoint非正sigma”只记录异常表象；原生API证据表明是collector误用raw logstd，不能据此认定checkpoint无效。
