# WAM-RL Policy v2：scheduled current565 → BC warmstart PPO

本 Task 承载新的执行器观测路线：当前 UniDex Shadow 状态300维 + WAM 世界系
8×11×3 手轨迹264维 + 历史有效位1维，共565维。包名为 `wam_rl_policy_v2`。

任务 goal → 独立 scheduler → 冻结 WAM-v4 → tau → 全量BC初始化的PPO actor →
24维原生动作。前6拍无完整H4时 tau置零、有效位为0，仍由BC执行。
空间 scheduler waypoint 尚不是经验证的0.8秒可达目标；phase尚未作为学习条件。

- `tools/run/train_current_bc.py`：全量条件缓存及BC训练。
- `tools/run/parallel_scheduled_cache.py`：互斥分片多GPU缓存。
- `tools/run/closed_loop_scheduled_shadow.py`：native闭环及批量PPO试跑。
- `tools/audit/check_control_pipeline.py`：真实模型、归档实测输入接口检查。
- [控制合同及审计](docs/research/2026-10-18-control-interface.md)。
- [全量BC结果](docs/experiments/probes/P-20261018-full-scheduled-current-bc.md)。
- [批量PPO试跑结果](docs/experiments/probes/P-20261018-scheduled-batch-ppo.md)。

新输出使用 `outputs/wam-rl-policy-v2/<run_id>/`。

2026-10-19 起，新运行默认 `--execution-protocol corrected --goal-type position_only`。
终止后的下一拍只对重置环境执行官方零动作初始化；该拍不参与PPO loss、KL或优势归一化，
真实终止transition保留，后续策略动作使用新观测。SE(3)任务显式选择`--goal-type se3_pose`。
PPO checkpoint绑定WAM权重/归一化、实际逐物体几何及调度/终止合同；旧checkpoint或
有意换条件必须传`--checkpoint-migration-reason`，产生独立迁移记录。原生官方复现默认
legacy时序和native终止；匹配WAM对照应显式统一corrected时序和common_workspace终止。
历史held60保持“曾连续保持”的原义，新增最终保持、曾保持后失守、最终任务成功与终止原因。
详见[执行边界修复](docs/research/2026-10-19-execution-boundaries.md)。
已完成产物仍在 `outputs/wam-rl-policy/full-scheduled-current-bc-20261018-local-r1/`，
初始策略直接读取其中 `fit-fast/best.pt`，无需重新生成缓存或训练BC。
WAM模型继续复用 `outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1/`。
checkpoint schema和数值合同保持兼容，Task目录v2不代表重新编号WAM模型或实验。

旧Task的scheduled专用入口只保留兼容跳转；其旧观测训练入口继续保留。
通用scheduler、WAM adapter、PPO数学与BC模型在本Task保留迁移时实现，
后续新观测路线在此维护；不依赖旧Task的Python包。数据准备/模型Task仍作为显式依赖。

验证（graspenv）：

```bash
PYTHONPATH=src/task/wam-rl-policy-v2/src python -m pytest -q src/task/wam-rl-policy-v2/tests
python src/task/wam-rl-policy-v2/tools/run/closed_loop_scheduled_shadow.py --help
```

迁移验证：20项本Task测试、9项旧入口兼容测试通过。真实WAM+父BC的12拍归档输入
检查通过（GPU6，11.1秒，PyTorch峰值417MiB）；BC均值保持误差2.38e-7、
PPO log-prob回放误差0。记录见
`outputs/wam-rl-policy-v2/migration-interface-20261018-r1/report.json`。
这是模型接口检查，不是native物理成功率或批量PPO训练结果。

批量PPO首轮已完成：16环境、4次更新、11520训练交互；详见上述实验卡。
当前只确认训练接口可用，尚未证实性能提升或跨物体泛化。


当前训练入口以用户指定的官方UniDex PPO为准：
`tools/run/launch_mixed_ppo.py` 启动六卡同一策略的混合物体训练，复用固定上游
PPO.run/update/storage及原配置，增加BC均值/WAM观测和多卡同步。
[协议与审计](docs/experiments/probes/P-20261018-official-scheduled-mixed-ppo.md)。
此前自定义PPO试跑保留为工程历史，不作为当前训练底座。

在线性能选项（默认关闭，运行时随环境变量继承到冻结worker）：

- `REF2DEX_BATCH_PIPELINE=all`：批量connector、GPU输入构造与tau转换；
  `legacy/tau/inputs/scheduler`用于单项对照。
- `REF2DEX_DEEP_OPT=all`：进一步批量姿态验证/交互几何、合并WAM有限性检查与状态传输；
  `cpu/sync/none`用于单项对照。配合`REF2DEX_BATCH_PIPELINE=all`使用。
- `REF2DEX_PROFILE_PPO=1`记录CPU profile；`REF2DEX_PROFILE_CUDA=1`仅记录第4轮CUDA时间线，
  用于短时诊断，不用于正常训练计时。

合并检查不跳过有限性判断：在predict返回前统一检查全部谓词，context之外保持原行为。
当前主训练使用自己的冻结副本；修改环境变量或源码不会热更新已运行的训练。
新启动器默认独立分配通信端口，显式指定已占用端口会在启动worker前失败。

## Inspire：沿用Shadow训练路线（2026-10-19用户修订）

正式适配使用共享`NormalizedCommandMLP`，隐藏层1024/1024/512/512、ELU；
全量660条DExplore示教（589 TRAIN、71 DEV），先current BC再条件BC，各50000更新。
输入为无手部ref的physical current312 + world tau264 + valid1，共577维；
12独立命令展开18原生命令，30Hz H4和tau。current721只取无ref实测slice9:257，
在线直接从真实刚体构造对应248维，并在原生进程做等价/参考扰动检查。
已知完整物体SE3路径是任务指令，由同一个scheduler给WAM局部目标，actor不读hand-ref。

- `tools/run/launch_inspire_full_bc.py`：冻结代码、全量提取、GPU6/7不重叠分片生成小型条件缓存、两阶段Shadow配方BC。
- `tools/run/launch_inspire_shadow_ppo.py`：冻结代码、启动模型/物理双运行时、资源与checkpoint预算。
- `tools/run/launch_inspire_mixed_ppo.py`：每卡同置两运行时，2–6 rank同步一个策略，50物体/589 TRAIN motion分摊、全局geometry绑定、rank0滚动保存。
- `tools/run/inspire_shadow_ppo.py`与`src/wam_rl_policy_v2/inspire_ppo_bridge.py`：直接调用Shadow同一`prepare_official`和UniDex PPO引擎；算法、GAE、网络容量、默认PPO超参相同。
- `tools/run/inspire_physics_host.py`：仅DExplore资产/初始化/物理/PD，不构造或训练DExplore actor，不使用其ref模仿奖励及终止。
- [全量同路线实验](docs/experiments/probes/P-20261019-inspire-shadow-parity.md)。

完整物体路径任务在环境adapter中定义有序进度、跟踪误差与终点门控；
这不是把Shadow静态终点成功率原样解释成整段任务完成。保持代理另行记录。
Inspire冻结WAM使用FP32：真实输入的bf16 batch64/single差超过1mm，FP32回归通过；
网络、权重未改变。缓存与在线推理精度写入同一个conditioning合同。
新Inspire训练修正上游adaptive KL与其平方scale_tril分布的失配；采样分布及官方其他PPO
超参保留，优化合同与旧checkpoint迁移单独记录，详见后期退化诊断实验卡。

Inspire吞吐选项由`launch_inspire_mixed_ppo.py`显式指定：
`--wam-executor-mode hand_shared_tf32 --wam-batch 256`保留FP32几何编码，
仅共享Transformer的矩阵乘法允许TF32，读取相同的手轨迹头，省去执行器不用的scene/C读出。
默认仍为`full_fp32`/128；不改变PPO、奖励、频率或环境数。
数学设置在预测返回或异常时恢复，不影响Actor/Critic更新；启动及每300次预测用真实状态
对照完整FP32前向，仍执行1mm门槛。推理模式写入execution identity，切换须显式
`--performance-migration-reason`，仅允许该项差异，其他身份不匹配仍拒绝续训。
参考路径初始化批量验证相同SE3条件并保持路径数组逐位一致。
`tools/audit/benchmark_inspire_inference.py`提供50 TRAIN物体的固定H4计时/精度检查；
在线日志的`rollout_timings`区分物理RPC、WAM、scheduler、indexed reset及传输。

旧`inspire_scheduled_probe.py`/`inspire_executor.py`是已归档的current986/小型单序列工程试跑，
不作为全量路线的BC或PPO入口；其失败结果不代表上述完整Shadow配方。
