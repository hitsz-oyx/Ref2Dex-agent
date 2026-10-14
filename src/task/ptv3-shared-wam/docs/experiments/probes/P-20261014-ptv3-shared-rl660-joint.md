---
schema: ref2dex.probe.v2
probe_id: P-20261014-ptv3-shared-rl660-joint
experiment_id: P-20261014-ptv3-shared-rl660-joint
date: 2026-10-14
task: ptv3-shared-wam
branch: main
git_commit: b22978c1
claim_id: C1
hypothesis_family: HF-ptv3-shared-rl660-joint
probe_index_in_family: 1
seed_pool: probe
seeds: [124]
decision_changed_if_positive: evaluate the jointly adapted forecaster and Chunk-R on held-out complete robot motions
decision_changed_if_negative: distinguish teacher data quality and retargeter mismatch before changing the shared architecture
status: RUNNING
run_id: rl660-joint-large-20261014-r1
---

# 全660条DExplore轨迹：同执行补录与共享预测器/Chunk-R联合微调

Result: RUNNING。扩容phase已到9460/19000步（累计10460/20000）。训练日志分段均值4.73→1.80；固定dev动作命令误差改善，手/物体轨迹误差进入平台。当前best=step4500、latest=step9000；尚无新增闭环抓取证据。
Decision: 用户明确选择补录完整机器人执行，再联合微调WAM+Chunk-R；不只做airplane。

Class: Decision. 服务于自训练机器人执行底座与可用交互预测。检验固定共享goal架构
在多物体机器人执行上的手/物体/命令预测是否可用；改善后进入独立闭环，不改善则
先区分数据失败轨迹、控制合同与适配缺陷，不把BC loss当成RL增益。

## Decision Note

- 用户授权停止原80k，改用完整660条RL重定向数据，并确认补录实际手点、状态和命令。
- 既有660导出共有166337帧、50类物体，仅qpos373:391和object198:205是实际执行，
  其余人体字段复制自几何参考；没有实际PD命令，不能直接用于完整WAM+R。
- 采用原660名单、对应几何计划及已有official actor作为只读数据生成器；模型只从
  自训练三域38k初始化，Chunk-R从头学习，official actor权重不进入最终模型。
- 采集两张空闲卡，首次4/5、资源冲突后转5/7，每批16个独立motion，最多3小时、4GB；不影响0–3上的其他工作。
  单批最多900秒，失败、非有限值、命令合同不符或资源超界时停所属进程，保存失败记录。
- 新数据仍按完整序列划分，按物体分层约10%dev，每物体至少一条。全部660为采集名单，
  TRAIN/dev之间不共享同一序列的窗口。准入质量策略在正式训练前固定并报告，不默认为
  所有official rollout都是成功抓取。
- 用户已确认排除明显失跟/掉落窗口。固定门为当前及未来24步的实际物体位置与
  已知参考位置差均<=0.10m；只作为明显失跟准入，不等同于成功抓取判定。
  每序列记录raw/admitted/excluded_windows及固定tick列表；TRAIN-only归一化也仅使用
  准入窗口。没有准入窗口的序列保留原记录并报告，不能声称660条都贡献了训练标签。
- 微调计划：2卡、每卡128/global256、20000 updates，保留既有全forecaster+R与GT-tau
  anchor、K4；不改goal、标签坐标或架构。一次GPU工程检查后启动；max_seconds=86400，
  硬上限4GB训练产物/400GB共享总量，使用已有监督器。无新权限边界。
- 成功后选择dev best做独立native执行；失败保留best/latest和质量分组，先定位标签/控制。

## 采集与数据合同

工具：[collect_rl660_batch.py](../../../tools/run/collect_rl660_batch.py)、
[collect_rl660.py](../../../tools/run/collect_rl660.py)。固定legacy Torch2.0.1/Isaac runtime，
GPU pipeline、30Hz，从真实frame0重置；每个环境对应一个motion，禁止隐式hard-object
oversampling。记录q/dq、11个真实native rigid-body link点、动态物体SE3/velocity、
实际送入pre_physics的规范化命令和实际PD targets。控制t对应state[t+1]，保留terminal。
用独立pd_from_action复算同tick转换，逐步检查捕获命令完全一致、PD差<=1e-6。
参考只作为已知计划和质量检查；监督手/物体均来自本次同一执行，不混用旧660人体坐标。
每物体canonical512点取自实际URDF collision frame，保留32同ID query。

原始成员与字段审计：`outputs/ptv3-shared-wam/rl660-source-audit-20261014-r1/summary.json`。
补录产物：`outputs/ptv3-shared-wam/rl660-corpus-20261014-r2/`；只在完整660回执、唯一成员
和一致controller检查通过后发布COMPLETED/training_allowed=true。

## 工程记录

- `engineering-rl660-collection-20261014-r1`未完成，旧runtime player构造参数不兼容，
  原始日志保留；修复为legacy flat config后重新独立运行，不视作方法负证据。
- `engineering-rl660-collection-20261014-r2`：GPU4，airplane432帧和mug334帧，42.92秒。
  逐步命令/PD检查通过，所有状态与命令有限，实际GPU观察约4213MiB。
  物体reference RMSE分别0.079m/0.865m；mug存在明显失跟，数值合同通过不等于成功示范。
- pilot只用于接线，TRAIN=airplane、dev=mug；不作为多物体方法收益证据。
  全量runtime与模型工程检查结果后续在本卡补充。
- `engineering-rl660-joint-20261014-r2`：真实两轨迹单卡128四步完成；手/scene损失均到PTv3、interaction、goal；predicted-action到forecaster与R，GT-anchor到R且forecaster梯度为0。
- `engineering-rl660-ddp2-prefetch-20261014-r1`：两卡每卡128/global256、双worker预取，四步及best/latest保存完成；最后步0.847秒，data wait1.13ms，rank0峰值分配3642MiB。只证明执行合同和负载，不证明学习。
- 全部33个Task测试通过，包含整窗质量、TRAIN-only归一化与断点回执检查。best清理回归先失败再通过。

## 初始化与checkpoint保存缺陷

预训练停在最后日志39100，完整latest为step38000，objective1.6881；已有best与latest
均指向38k。历史最低22k/objective1.5545没有被独立保留，已不可恢复。初始化必须说明
是best available，不声称历史真正best。原20k、38k和全部curve/日志保留。
已补回归检查重现best被恢复点清理的缺陷；新训练验证创新低时立即保存完整checkpoint，
原子维护best/latest，清理时保护alias所指的登记文件。不能据此追回历史权重。

## Limitations / future evidence

单seed离线Probe，不证明机器人抓取、多物体泛化或最终Cm因果增益。660来自official
actor作为数据生成器，不能称全部是自训练RL演示。重采集与原导出不要求逐位一致，
关键是本次手/物体/动作配对和控制时钟正确。后续必须核对质量分组与真实闭环；
近10%序列dev不是物体不相交测试，不能据此声称unseen-object泛化。

## 全量采集与恢复记录（2026-10-14）

- 全量r1因旧共享asset bank缺少banana/cylindermedium mesh失败，原日志保留。
  修复提交5e18ca18：每个批次独立只读资产视图，使用既有完整50物体asset root；
  不改共享软链接、历史专家bank或外部项目。r2前13批共208条已完成。
- r2第14批在初始化前检测GPU4上的外部进程3149043，安全停止；其命令属于
  `/home2/jiao/gs_slam`，不是本任务进程，不停止它。这是资源冲突，不是非有限
  轨迹、PD合同错误或方法负证据。原GPU5上的未完成第13批保留后重试。
- Decision Note：复用已有208条完整回执，保持原成员/每批seed/控制合同/几何资产；
  未完成批次写新resume01目录。转空闲5/7，仍为两卡且不超过原10800秒累计
  采集预算；不覆盖任何轨迹或失败日志。不改变质量门或训练方案，无新授权边界。
- `finetune_after_collection.py` 有界等待完整660并逐序列检查有限值、同执行动作、
  准入tick一致性，输出每split的准入/零窗口序列统计后启动
  `rl660-joint-20261014-r1`（GPU5/7、global256、20000步）。资源占用或采集失败
  时停止交接，不抢占他人进程；训练由已有监督器独立管理best/latest、24小时与4GB预算。

恢复阶段的supervisor PID3180629已完成全660条；交接PID3181015已完成审核并
启动独立训练监督器PID3234992。交接记录
`tmp/rl660-joint-20261014-r1-handoff/status.json`为TRAINING_OBSERVED。

## 全量准入与实际训练启动

- collection run：`rl660-corpus-20261014-r2`，660条/50类/166337帧，累计采集
  1879.72秒（包含恢复前有效运行时间，不含两次运行之间的人工修复间隔）。
- 质量门当前到未来24步最大物体位置误差<=0.10m：raw150497窗口，
  admitted122685，excluded27812（18.48%）。无零窗口序列，最少准入7窗口。
  每序列数量和固定tick已存corpus manifest；不同序列的准入长度不等。
- 完整序列TRAIN/dev为589/71条，准入110777/11908窗口；没有跨split同序列
  窗口，逐序列有限值、实际动作一致性和准入tick复算均通过。归一化只用TRAIN。
- 正式run：`outputs/ptv3-shared-wam/rl660-joint-20261014-r1/`，
  GPU5/7，128每卡/global256，20000更新，loader4 workers/rank、预取2、cache128、
  pin_memory。初始化自best available step38000，全WAM+新R联合学习，保留GT-tau anchor。
- 早期观察step240：0.595s/update，430.6 samples/s，data wait 3.32ms，peak allocated 3926MiB；物理占用约4793/4787MiB。
- 18秒10次GPU采样：GPU5平均53.0%（15–82%）、GPU7平均46.8%（25–79%），
  无新增显存异常。早期按3–4小时预留验证/checkpoint开销；后续按正式吞吐修正。
- 训练/梯度目前有限；early training loss仅说明优化运行，不能作为抓取改善证据。
  验证每500步，普通checkpoint每1000步，新best立即保存并保护best/latest。

## 2026-10-14 用户授权容量与吞吐优化

Decision Note：当前52M模型在每卡128时仅约4.7GiB，数据等待约0.5%，
主要实现候选包括逐样本nonzero取token与小算子/同步/双卡通信。用户明确要求
扩大模型、batch并全面优化。先在完整step1000之后停止本任务，保留best/latest，
原指标及监督器SIGTERM退出记录保留；不修改正在运行的src。
固定660准入/坐标/目标，保留PTv3、128D观察编码器、输出头和已学R；共享trunk
尝试从128D×4扩大到768D×8，按整数复制权重并均值投影、附加恒等残差层，
先证明迁移初始输出保持，再实际测试128/256/384吞吐、显存和训练梯度。
这不是更大模型有效性的matched比较，也不声明扩容改善抓取。
工程优化限定两卡5/7、最多一小时、每次短测试；无OOM/合同错误且吞吐/负载
可接受时，从1000步权重迁移后进行剩余19000更新。扩容重置optimizer，记录
新phase/global batch/累计样本量，不伪称精确恢复旧训练。原86400秒训练与
400GB共享产物预算仍有效，无新增权限边界。

### 性能与容量工程结果

- 第1轮`engineering-rl660-scale-20261014-r1`保留：768D×8无重计算在256
  OOM，benchmark旧版OOM清理也报UnboundLocalError，原日志/已完成case不覆盖。
  修复清理变量后，第2轮各case正常结束。纯token向量化未见明确收益，转查padding。
- 原PTv3逐样本padding使单步profile出现8448次item、7772次stream synchronize；
  向量化保持所有pad/unpad/cu_seqlens并通过边界/短样本/零长度行索引回归。
  优化大模型profile中stream synchronize降到139次；两组profile的模型/batch不同，
  数量用于定位开销，不能作为同算子耗时比。
- `engineering-rl660-scale-20261014-r2`，真实固定source1000、单GPU5、3步预热+8步测量：

| 配置 | batch/GPU | step秒 | samples/s | peak allocated MiB |
|---|---:|---:|---:|---:|
| 原128D×4，原padding/token、普通AdamW |128|0.503|254.7|3654|
| 同模型，向量化padding/token |128|0.358|357.3|3687|
| 同模型，向量化+融合AdamW+TF32 |256|0.339|755.0|6704|
| 768D×8，重计算+上述优化 |256|1.294|197.8|9940|
| 768D×8，重计算+上述优化 |384|1.896|202.5|14211|

- 768D按6份复制时FP32除以6会有权重舍入；第一版真实pipeline的10微米门失败，
  第二版同输入FP64严格1e-9门也失败（差5.06e-7）。未把它们记录成精确保持迁移。
  最终采用512D×16，按4份复制/除以4（binary-exact）；parameter101636370。
- `engineering-rl660-transfer-20261014-r3`，8个实际TRAIN窗口：相同输入共享trunk
  FP64最大差1.35e-14；完整GPU pipeline scene/hand最大差0.0404/0.0717mm，
  normalized action差5.33e-6，GT-anchor差0。保持0.1mm pipeline数值门并明确
  dtype/kernel数值差异；这不是预测准确性或闭环效果验证。
- `engineering-rl660-scale-20261014-r3`：512D×16且重计算，batch256时1.371秒/
  186.7samples/s/9561MiB，batch384时2.013秒/190.8samples/s/13672MiB。
  384只多约2.2%吞吐，单update耗时约多47%，因此正式选256/GPU、global512。
  总batch翻倍、参数约翻倍；更大容量有真实计算成本，不能宣称整体训练比小模型快。
- Task全部47项CPU接口/合同检查通过；实际GPU迁移审核已通过。双卡大batch
  工程检查随后补充。扩容optimizer重置，固定sample_step_offset1000，新phase
  19000更新：本轮累计更新目标仍20000，但global batch512会增加总样本曝光。

### 扩容后的短更新稳定性门

Blocker：大batch四步工程检查虽完成，但其3-window dev不能直接与原512-window
curve比较；按相同64个dev窗口复算，区分样本选择与扩容更新扰动。这项检查会
决定是否降低迁移/新增层学习率，最便宜方法是现有两个checkpoint的GPU离线前向。
`engineering-rl660-expansion-dev-20261014-r1`：原1000步objective5.0929、hand3.5259；
扩容直接旧LR四更新后9.0109、7.4050，出现破坏性扰动。原结果保留，不能只因
工程finite而启动长期训练。初始函数已对齐；宽矩阵复制后Adam更新的有效幅度与
12个新恒等层的累积扰动需要分开控制。
配置改为：scene1e-5、其余旧编码器/head/R5e-5、前4层迁移interaction2.5e-5、
新增12层1e-5。不改标签、样本窗或架构，optimizer仍明确重新初始化。
重新四步后用相同64窗口检查objective/hand/action均不超过parent1.2倍，
通过才进入正式19000更新。这个工程稳定性门不证明更大模型有效或更好抓取。

- 降低分组LR后的`engineering-rl660-large-ddp-20261014-r2`四步双卡global512完成，
  所有关键梯度合同和best/latest完整保存通过；工程权重不进入正式训练。
- 同64 dev的`engineering-rl660-expansion-dev-20261014-r2`通过：parent/candidate
  objective5.0930/5.1548（+1.2%），hand3.5260/3.6158（+2.5%），
  action0.4629/0.4605。相同窗口门通过只表明未出现原始LR的剧烈扰动。
- 最终Task48项测试通过；正式配置256/GPU/global512、101636370参数，
  继承原1000步model/已学R，optimizer按新分组重新初始化，新phase19000步；
  max_seconds80000扣除已用时间和工程余量，checkpoint不留10k milestone，
  仅保护best/latest以维持4GB单run预算。原三域38k和机器人1000权重不覆盖。

### 大模型正式运行启动

`rl660-joint-large-20261014-r1`已由监督器PID3344258启动，code b22978c1。
从原robot step1000重新迁移，未使用engineering checkpoints；全WAM+已有R继续
联合学习，optimizer按新分组重置。新phase步数0–19000，原phase1000步另计。
source/frame/准入/split不变。大模型双卡检查约1.7秒/步，预计9小时上下；
实际吞吐与GPU平均负载将在前40步后更新，不能再沿用小模型3–4小时估计。

正式大模型step100观测：1.466秒/update、349.2samples/s、data wait5.98ms，peak allocated10096MiB，physical11141/11197MiB。按当时进度剩余7.70小时；比工程检查更快，预留验证/保存按8–9小时更新。所有关键初始梯度合同通过、loss/梯度有限。

18秒、每卡10次稳态采样：GPU5平均94.7%（74–100%），GPU7平均93.4%（65–100%）；物理占用11141/11197MiB。该负载测量仅用于工程资源判断。


## 2026-10-14 中期结果（15:44 CST固定快照）

运行仍为 `rl660-joint-large-20261014-r1`，实际代码b22978c1、seed124、GPU5/7、
global512，数据准入/split/目标与启动协议一致。本节所有step均为扩容后的phase步数，
其前另有小模型1000步；不把两种架构的训练loss直接连成同一曲线。
快照为phase9460/19000，即累计10460/20000，最近1.450秒/update、353样本/s，
日志ETA约3.84小时，尚受后续dev/保存开销影响。rank0峰值分配10107MiB，
该条日志物理显存15393/11199MiB；单条GPU瞬时利用率不能替代上文稳态均值。

证据文件：`outputs/ptv3-shared-wam/rl660-joint-large-20261014-r1/interim-audit-20261014.json`。
其中保留观测时间、完整指标、代码/初始化来源、训练分段均值和checkpoint别名。
记录时 `best.pt → step4500.pt`、`latest.pt → step9000.pt`；它们随训练继续更新，
本节数值以该快照为准。

### 训练拟合趋势

下表是指定500步区间内已写入日志点的算术均值（通常每20步记录，共25–26点），
不是每个update的完整平均，也不是dev指标。

| phase区间 | 日志点数 | total | hand | scene | action | GT-anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–500 | 26 | 4.7296 | 3.2602 | 0.4088 | 0.5348 | 0.5258 |
| 2501–3000 | 25 | 2.9384 | 1.6402 | 0.3392 | 0.4844 | 0.4746 |
| 5001–5500 | 25 | 2.2655 | 1.1048 | 0.2708 | 0.4511 | 0.4388 |
| 8501–9000 | 25 | 1.8032 | 0.7435 | 0.2178 | 0.4259 | 0.4161 |

分段训练均值持续下降，说明训练集拟合继续改善；不代表每个batch单调下降，
也不能据此判定泛化、收敛或实际抓取能力。

### 固定dev结果

沿用固定512窗面板、相同序列划分与度量。hand是有效真实手点的未来位移EPE，
first8/all24分别覆盖未来1–8/1–24步；scene是同ID物体表面点的未来轨迹EPE，
并非物体质心误差。表内PD指标是预测τ经Chunk-R所得第一步命令相对记录命令的
误差：规范化action经clamp后乘native PD scale；xyz为三轴L2（mm），rotation为
三旋转坐标最大绝对差（rad），finger为有效手指坐标最大绝对差（rad），再对窗口
取均值。它们衡量命令预测，不是仿真实际执行后的末端位置/姿态误差。

| 指标（越低越好） | step1000 | step4500（dev best） | step9000（最近dev） |
| --- | ---: | ---: | ---: |
| dev objective | 5.0118 | 4.7704 | 4.9944 |
| hand first8 / mm | 51.74 | 51.52 | 51.88 |
| hand all24 / mm | 111.48 | 108.47 | 111.43 |
| scene all24 / mm | 40.72 | 40.47 | 41.95 |
| predicted-τ PD xyz / mm | 71.48 | 63.88 | 57.27 |
| predicted-τ PD rotation / rad | 0.1775 | 0.1482 | 0.1371 |
| predicted-τ PD finger / rad | 0.1355 | 0.1114 | 0.1063 |
| GT-τ PD xyz / mm | 70.08 | 61.75 | 54.39 |

step9000 predicted-τ PD xyz的p95=199.67mm，GT-τ为189.64mm，clipping均值为0。
对比step1000，命令xyz均值下降约19.9%，rotation/finger下降约22.8%/21.5%；
手轨迹几乎持平，scene略差。dev objective在4500达到当前最低，此后未持续下降，
不能用训练loss下降描述成整体dev稳定改善。

### 当前判断与下一步

动作命令映射出现局部PROMISING信号；手/物体轨迹的持续泛化收益仍UNCLEAR，
整条机器人执行路线尚无新闭环结果。保留当前best与latest，继续已授权的固定19000步
预算并观察固定dev；不根据本次快照加预算、改网络或宣布收敛。完成后优先对dev best
做独立native执行，再决定是否值得追加。GT-τ与predicted-τ误差接近只作诊断读数，
尚不能把剩余误差归因于R、τ或数据某一环节；抓取收益与最终Cm增益仍属延后证据。
