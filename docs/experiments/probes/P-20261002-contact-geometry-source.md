# 接触几何与动作组合：源采集合同

本卡当前是工程准备，尚未运行科学Probe，不消耗旧HF18 slot、不改旧失败门。
问题：为动作—相对几何作用模型取得真实混合动作监督，同时保留随机候选
排序和重复参考噪声的可辨识性。新路线在原生工程通过后才登记科学预算。

每次合格pre-state生成8个执行程序：0为原生base反馈；1为rotation_cup；
2–7为6个随机组合。组合分6块：XYZ、index、中指、pinky、ring、thumb。
各块对6专家非负权重和为1；thumb共用一组权重，原生耦合由真实执行器实施。
随机权重是独立U(0,1)经(-log(U))^3再归一化，明确不称Dirichlet。
先生成整个候选集，再用独立RNG均匀分配10槽；槽8重复base、9重复cup。
实际option0/1概率.2，其余.1；重复槽只测随机噪声，不声称逐状态配对oracle。

组合权重和腕部旋转目标在窗口开始固定；每一步重新调用六专家反馈，组合
新的XYZ/手指命令，cup和组合均保持窗口起点旋转，执行10步后重新观察。
不输入未知未来命令；未执行候选只保存参数与起点动作，没有反事实真值。
独立proposal seed=assignment_seed+30000，allocation seed=assignment_seed。

触发：首episode、joint归一化净力存在代理连续3tick、中心rest>=5mm、
warmup>=10、剩余episode>11tick；不要求已经离桌，不筛选未来成功或当前vz。
initial-clear定义rest+3cm且meshCLR>=2mm，early/clear各最多4窗口/episode，
总8窗口，完成窗口后cooldown6。96环境、最多650tick、240秒/native phase。
所有状态保存，整初始motion/start组用SHA12651固定fit<50/cal<70/held>=70；
当前仍是探索数据，不能称未被研究路线观察的正式Validation。

保存全候选系数、初始动作、allocation/actual propensity、10步全6专家bank、
pre-native观测1442、真实全18PD目标、pre/future状态/原始力/mesh间隙及done。
另保存pre和post掌部/五指端的真实world位置与速度，以及全10步post-native
观测1442，便于独立验证几何标签，包括最后一帧而非只验证有next-pre的前9步。
force只作存在代理/观测，不当作周期平均力。未发生完整10步或跨reset的窗口
不能作为训练标签；采集失败不通过删坏片段来形成正向结果。

所有专家冻结；Cm/V/PPO均不训练。工程必须核对native PD映射、随机槽合并
概率、组合边界、固定旋转、逐步专家回放、真实几何与标签，再接纳科学数据。
离线动作工程只证明可生成命令，不能替代原生仿真工程或候选优势证明。
后续科学采集/拟合卡在运行前固定支持与信息/排序门和全成本；GPU默认一张，
整slot<=60min/8GiB，包括工程/失败/审计。GPU访问已恢复，先执行有界原生工程。
