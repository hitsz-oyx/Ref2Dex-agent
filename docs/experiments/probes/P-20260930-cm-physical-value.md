# P-20260930-cm-physical-value

Family: HF08；probe_index_in_family: 1/1；类型：Decision Probe。
状态：工程 preflight；conclusion：未形成；分支 agent/cm-physical-value。

用户已批准 [固定设计](../../superpowers/specs/2026-09-30-cm-physical-value-design.md)，并授权本轮暂时跳过代理工作流，root 直接推进研究。期限 2026-10-03 23:59 Asia/Shanghai。具体代码提交、命令、输入 hash 与阶段状态由原生 run_manifest 保存；工程 smoke 不消费科学 Probe slot。

本实验区分：完整真实交互数据上的短期动作条件物理模型与长期价值结合，能否改善训练所得 actor，超过普通 PPO 及同形式的直接 Q 候选监督。正向后优先正式 Validation；完整有效负向则停止该实现；合同失败或预算未完成为 UNCLEAR。最便宜的有效检验是固定公共预训练池及单一三臂矩阵，不对数据规模分别做 RL sweep。

固定 source 为自训练 source_e260，SHA256 16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f。三 motion 为 canonical airplane s3/s7/s9，参考文件 hash 在 hf02_temporal_canonical_route.json，全部臂使用同一 actor/critic/normalization/optimizer 初态，使用相同任务 reward。无六专家路由。候选监督保留 PPO 原执行动作及 logprob；独立评估只运行 actor。

公共采集 seeds283/284，每次目标500000转移、64环境，达到目标后完成活跃 episode。保存逐步实际 action、pre/post物理状态、真实 reward 分量、终止原因、reset前后继及参考上下文。完整 episode 20%开发 holdout、80%fit；规模100k/500k/1M是公共池目标，最大档实际fit约800k，全部报告实际行数，不将holdout计入拟合。每档固定1000更新/模型、三动力学成员、V与Q；最大档增加同容量动作无关诊断。最大档开发loss选模型，不按策略评价挑模型。

三训练臂 plain_off/direct_q/cm_value，seeds286/287，均追加160epoch（e260→e420）、64环境、horizon32，每臂每seed327680新交互。在追加0/40/80/160epoch固定 actor-only 评价；seeds288/289，每次96完整首回合，三motion各32，全部从帧0开始。主判定仅终点，每臂384episode；曲线面积为辅指标。

PROMISING：cm_value对两对照均≥+5pp持续抬升成功率，各训练seed对两对照均非负；全episode中成功后掉落比例对任一对照不恶化超过5pp。稳定成功为高度≥初始+3cm且满足固定接触代理，连续45控制步（1.5s）。另报历史五步、最长保持、平均抬升、接触及条件掉落率。Probe标签不构成正式科学结论。

硬上限：同时最多2GPU，总阶段累计≤6h，新增产物≤20GiB；采集≤90min，模型预训练≤120min，六臂训练≤90min，矩阵评估≤60min。使用空闲GPU，不触碰未知进程。预算不足不得换seed、缩短矩阵或改门槛拼成负向结论。

## 工程证据

r5 位于 src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r5。12项合同/模型/数据检查通过。smoke采集3252条完整转移、6episode；source actor-only评价完成6episode；128行档2更新仅用于接线。三臂单epoch训练均完成，保存e261；共同初始 actor hash d97f44d9b6cd5281faa63917021a94204658b5c461abc23017c7f4cd48061eb9。direct_q/cm_value保存的行为/RNG完整性及actor监督梯度检查均通过。科学百万转移采集尚未启动。

早期工程失败r1–r4及smoke_train_r1均保留原日志；分别修复参数缩写、player batch初始化、CUDA设备编号、obs包装及import路径，不用于科学判定。outputs是指向baseline的共享软链；新正式产物使用当前工作树research/output，launcher拒绝越界路径。r1仅生成少量启动日志/配置，失败于模拟器启动前，证据不删除。

三组保存e261的 actor-only 首回合评价已全部完成（每臂6episode，start_frame全部0），工程闭环通过。科学执行登记run_id r6，代码4e389cc，物理GPU1，最大六小时阶段累计预算；launcher stage all依次执行采集、模型预训练、六臂训练和固定评价矩阵。运行记录位于当前工作树 src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r6/run_manifest.json。启动状态RUNNING/采集，尚无科学判定。

r6 在第一次非空 reset 因环境编号为list而失败，耗时75.409s，无完整科学数据池或策略结果。只保留FAILED/工程证据；修正为Tensor IDs，并对player起始执行显式全量reset。补充审计发现r5初始未reset导致全部motion_id=0，因此r5评价只证明checkpoint加载，不能证明三轨迹覆盖。r7扩大smoke到4000目标转移、覆盖重复reset，并加入首回合motion均衡硬检查；继承r6的75.409s科学耗时和分阶段成本，slot仍为HF08 1/1，seed不变。当前状态r7工程检查RUNNING，科学矩阵等待此检查通过后启动。修正代码35ee1b8。

r7扩展工程检查COMPLETED：采集6592条/12episode，三motion各4；起始帧包含0和5/16/17/27/33/34，已覆盖跨回合reset。source actor-only评价6episode，三motion各2，全部start_frame0。12项合同检查再次通过。r7科学stage all从公共采集开始；实施代码64ba312，准确阶段提交以manifest为准。

r7 mixed-data预训练工程smoke已在空闲GPU2完成（2更新/模型，非科学规模）：6592完整转移、fit4166、holdout2426、12episode、excluded_rows0，V/Q及四个动力学诊断模型均能训练保存；native manifest/results在r7/smoke_models_mixed。该模型不用于科学策略训练或模型挑选，GPU2任务已结束。科学采集session仍为22846，GPU1，固定完整数据预算不变。

科学collect_s283已完成：520833转移、960完整episode、excluded_rows0；fit415925行/768episode，holdout104908行/192episode。三motion转移174480/175912/170441，四噪声档均覆盖（约12–14.5万行/档），动作界[-1,1]，维度std约0.139–0.190。逐shard hash与数据合同及完整return检查通过，collection_audit.json保留原输入。零起始帧501/960，符合随机混合采集。采集策略有8个稳定保持标签、71个历史五步标签、6个成功后掉落标签；这些属于带额外噪声的采集池描述，不是actor-only主评价或策略收益结果。父phase945.223s，native采集924.866s；第二批collect_s284已自动启动，同一session22846。

公共采集全部完成：1041599转移、1920完整episode；逐shard hash、episode连续性、完整return及fit/holdout隔离审计见r7/full_collection_audit.json。两批合计15个稳定标签、147个历史五步标签、12个成功后掉落标签，仅作带噪声采集池描述。第二批父phase1060.912s，两批加r6失败科学耗时累计2081.544s，未超过采集90min上限。正式fit已启动，固定三档各1000更新，不依小档loss改策略矩阵。新增独立模型分量/完整return审计工具audit_cm_physical_value_models.py（a562a14），只作开发集诊断，已通过三轨迹工程smoke，不参与模型挑选或改变主gate。

96环境actor-only工程吞吐为97.789s，串行48点预计超过评价60min墙钟预算。按已授权两GPU并行，不改矩阵与判定；配对三臂在同GPU，GPU映射跨training-seed互换。launcher已SIGSTOP暂停后续派发，native fit继续完整运行，PID和命令核对在r7/launcher_handoff.json，处置见D-20260930-hf08-evaluation-throughput.md。阶段wall与GPU计算成本分别报告，不增加Probe slot。

正式预训练已COMPLETED，native1534.601s；三档实际fit100206/500169/831811。完整fit原生manifest及各checkpoint hash复核通过；旧launcher仅在native已退出且模型完整后SIGTERM结束（session22846 exit143是主动调度交接，不是模型失败）。fit父阶段保守上界2764.470s含root交接等待，科学阶段已记累计4846.014s，不遗漏该等待成本。新paired两GPU训练调度c9c49f0/session93879，seed286三臂GPU1、seed287三臂GPU2；全48点评价另用并行调度，新增4项固定gate/配对/缺失点检查通过。原研究变量、160追加epoch、每臂每seed327680交互及gate不变；墙钟与设备分配时长单独报告。


plain_off与direct_q四臂科学训练均已完成e420、327680新交互，初始模型hash一致；两个cm_value臂继续固定训练。新evaluation_handoff.py/session17061仅等待六臂终态、复核epoch/frame/行为与梯度标记后启动既有48点评价，不改变矩阵。当前运行训练调度加载的是c9c49f0，其中GPU时长汇总误将group wall纳入子臂合计；交接脚本会保留原值并更正为六个native臂时长之和，墙钟账本和科学结果不变。代码68bb31f已修正未来运行。

最大档预训练模型开发审计（固定2048行、component_audit_largest_with_baselines.json）：物体位置坐标RMSE0.027890m，高于copy-state的0.009263m及恒定线速度的0.004684m；姿态测地RMSE1.663830rad，高于copy-state的0.193119rad。该恒速基线的旋转仍沿用当前姿态，不包含角速度积分。Cm的joint-position/velocity、object-velocity和contact分量优于copy-state，但不能用总loss概括为物理转移已准确。V/Q对完整return的RMSE24.428/25.808低于fit-mean常量45.480，仍不证明未执行候选动作的排序或policy收益。以上是带噪source分布的开发诊断，不改变固定闭环或主gate；新增online-checkpoint审计仅比较Cm物理分量，明确不将source-policy MC return用于宣称当前策略V/Q校准。
