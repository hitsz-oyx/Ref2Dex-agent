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
预算并观察固定dev；不根据本次快照加预算、改网络或宣布收敛。当时拟完成后优先对
dev best做独立native执行；该默认步骤已被下节ref2审计修正。GT-τ与predicted-τ误差接近只作诊断读数，
尚不能把剩余误差归因于R、τ或数据某一环节；抓取收益与最终Cm增益仍属延后证据。


## 2026-10-14 ref2微调缺陷审计（16:10 CST）

依据用户提供的[ref2.md](../../user/ref/ref2.md)，对运行代码b22978c1及真实corpus做只读审计。
快照phase10540；原训练保持RUNNING，本节不改变loss/config/已有checkpoint的含义。
审计工具：[audit_ref2_finetune.py](../../../tools/audit/audit_ref2_finetune.py)。
证据：`outputs/ptv3-shared-wam/rl660-ref2-audit-20261014-r1/report.json` 与 `repeat.json`。
CPU只用于微小解析梯度和文件/标签算术，无模型拟合或批量神经推理。

### 已确认的问题与边界

1. **控制监督缺少执行前缀侧重（高优先级）**：`robot.py:135` 的predicted-action与
   GT-anchor均为all8×12 standardized L1，没有历史control-aware配方的first/PD0项。
   对真实 `robot_loss()` 做单位action_scale的最小梯度复现，两分支first/tail单标量
   梯度比均为1；第一步全部DOF占12.5%、第一步XYZ占3.125%。这些是损失在标准化
   输出上的系数占比，不是实际参数梯度或最终训练贡献占比；真实action_scale会改变
   native单位上的梯度。缺少第一步物理误差目标已确认，解释52–54mm偏差的因果占比
   尚未验证。
2. **选点/保存偏离控制目标（高优先级）**：`training.py:461` 只按包含hand/scene/
   action/anchor的总dev objective选 `best.pt`，不是独立forecast-best或control-best。
   固定快照总best=4500，GT xyz mean/p95=61.75/233.06mm；控制p95最优=10500，
   mean/p95=52.95/177.93mm，predicted-τ为55.79/188.55mm。后者总objective4.9415，
   因而没有成为best；10500又非1000步恢复点，未保存权重，无法从该次dev读数恢复。
   即便未来某个控制最优点恰逢恢复点，清理也只保护总best/latest/里程碑，不能保证
   控制最优长期存在。不能把现有 `best.pt` 自动当成执行候选的最优权重。
3. **RL660原生执行接口尚未适配（确定的工程blocker）**：训练manifest将reference/
   canonical写成逐序列来源描述，实际文件在corpus各record目录；`policy_server.py:47`
   仍把描述当文件路径，`execute_native.py:78` 仍读corpus根的 `trajectory.npz` 并固定
   env24及airplane环境，`audit_native.py` 也沿用旧reference路径。对本run实际metadata
   执行对应 `np.load`，canonical/reference/source-trajectory三项均FileNotFoundError。
   不能直接沿用旧入口对660训练做多物体native结论；这不影响当前离线更新本身。

### ref2中需要修正或暂缓的归因

- **执行频率前提不同**：当前launcher默认K_exec=4，ActionBuffer最小复现消费索引
  [0,1,2,3]后才重新查询；可显式选择1/4/8。ref2及历史control-aware实验的“每tick
  只执行a0”对应K=1，不能直接描述当前默认K4。后续first-action诊断应明确固定K1；
  如要保留K4，就应同时检查前4步监督与实际执行误差，不能只看第一步宣布K4可用。
- **历史结果存在，但范围有限**：原fit/result与独立audit确认report first-xyz p95
  19.51mm、GT执行teacher/oldR/newR连续held73/75/74，且逐tick执行第一步。该Probe
  支持优先测试control-aware目标，但它是不同数据/局部任务，且loss和selection一起
  改变，不能证明RL660的all8目标就是当前偏差的唯一或首要原因，更不证明其普遍无效。
  历史PD0为三个物理分组的逐坐标绝对误差均值/.01m、.1rad、.1rad，再等权平均；
  不能与报告xyz L2或ref2的示意公式混成同一实现。
- **state87与R容量是候选限制**：R仍128D、2层decoder，输入为q/dq/局部手点/object
  pose/velocity及24×33 τ，没有直接geometry/contact/previous-action/reference输入。
  信息不足、跨物体覆盖和容量均需区分；没有matched诊断前不据此加输入或扩模型。
  即使小预算first-only模型TRAIN仍高，也需先排除优化未充分/容量问题，不能直接证明
  inverse映射不确定。
- **分阶段读数缺失**：现有固定dev512窗中startup(tick<16)=39、later=473，覆盖69/71
  条dev序列；训练抽样则约一半startup。当前汇总PD不能说明起步与接触阶段各自是否
  改善。后续同时报告startup/later和手物距离分组；距离只作几何代理，不等同真实接触。

### 标签、坐标与PD复算

抽取TRAIN/dev各4条完整序列，每条各一个startup/later窗，共16窗；对真实loader输出
独立复算同tick applied action、future1–24手位移、state87、WAM→native点顺序，及
记录PD targets与扰动action的独立PD差。最大绝对差：action=0，τ/state=1.19e-7，
点序/坐标=2.05e-7，PD=1.19e-7、PD差=2.20e-7；均小于1e-6。
这些检查未见所抽窗口的时序/坐标/finger半scale错误，不是全660条的穷尽证明。

复现命令（使用全新报告名，禁止覆盖证据）：

```bash
PYTHONWARNINGS=ignore /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam/tools/audit/audit_ref2_finetune.py \
  --run outputs/ptv3-shared-wam/rl660-joint-large-20261014-r1 \
  --output outputs/ptv3-shared-wam/<fresh-ref2-audit>/report.json \
  --require-control-alignment
```

两次执行均按预期退出1：`DATA_REPLAY True`，随后
`FAIL: first-control emphasis, selection, or native inputs are not aligned`。
这是成功复现工程问题，不是训练进程失败；没有改动活跃trainer源码以使其“通过”。

### Decision Note：后续修复与最小判别

- 问题：控制目标/选点错位是否足以解释当前GT-τ误差，以及如何选择真正可执行候选。
  当前证据确认两处合同缺口和一处native接口blocker，尚无RL660上的因果比较。
- 行动：保留这轮为all8联合基线，继续已授权预算；撤回上一节“完成后直接取总dev best
  做native”的默认步骤。修复时必须独立保存forecast/control选择与对应权重，并在执行
  前补齐按指定record选择geometry/reference/场景的接口和审计，不能仅替换一个路径。
- 最便宜下一Probe：固定原TRAIN/dev、GT-τ、state87、128D×2、初始化与采样流，
  R-only比较A(all8)、B(first/tail/PD0)、C(first-only)。先测TRAIN可拟合性再看dev，
  同时报first XYZ mean/p95、rotation/finger/clipping、startup/later和尾部误差。
  不动WAM，不把历史单物体权重或新conditioning混入对照。
- 当前保守计数本地5/7、远端0/2及既有1/3合计六卡，故本次只做CPU算术审计，未新增
  GPU训练；待既有卡释放后，先做有界单卡Probe（最多20分钟、产物100MB，仍计入共享
  六卡/400GB），单预算到期或非有限/外部冲突立即停，不做扫参或自动扩模型。
- 正向时再按K1做GT执行门；不改善则分开排除优化/容量与输入覆盖，不能仅凭负Probe
  关闭inverse路线。此次未修改核心问题/claim、没有新增外部权限需求。
