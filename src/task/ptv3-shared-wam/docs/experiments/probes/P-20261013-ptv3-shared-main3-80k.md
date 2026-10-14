---
schema: ref2dex.probe.v2
probe_id: P-20261013-ptv3-shared-main3-80k
experiment_id: P-20261013-ptv3-shared-main3-80k
date: 2026-10-13
task: ptv3-shared-wam
branch: main
git_commit: 8510ec3a
claim_id: C1
hypothesis_family: HF-ptv3-shared-task-forecaster
probe_index_in_family: 1
seed_pool: probe
seeds: [100]
decision_changed_if_positive: transfer the trained shared forecaster into airplane full-pair adaptation
decision_changed_if_negative: inspect the forecast contract and data before spending on robot adaptation
status: PROMISING
run_id: pretrain-main3-ddp4-80k-20261013-r2
---

# 四卡三域共享预测器预训练：80,000 updates

Result: PROMISING（局部离线预测；用户已提前停止预训练）。37k固定dev显示局部学习信号：GRAB/ARCTIC目标物体中段误差低于goal-only插值；手轨迹相对保持当前手不动改善约35–39%，但GRAB后16步平均误差仍122mm。原训练最后日志39100，完整恢复点38000已保留；尚无抓取结论。
Decision: 用户要求停止三域80k训练，保留完整best available/latest，转向660条机器人执行补录与联合适配。

Class: Decision. 服务于自主灵巧操作的预测器基础，而不直接验证Cm增益。具体问题是：
该共享架构能否在三域实际手物运动上学到手轨迹和超过endpoint插值的中间场景预测。
正向信号支持下一阶段airplane完整forecaster+R适配；无手轨迹学习或中间预测收益时，
先查本Task数据/合同，不把human预测误差当作机器人抓取成功率。

## Decision Note

- 2026-10-14用户要求停止预训练并转向660条DExplore RL重定向数据微调。当前约39k，
  最近完整checkpoint为38k。选择验证完整38k后停止已确认归属的四卡进程，保留日志、
  20k及38k；best/latest别名只指向已有完整文件。历史最低dev objective在22k，
  其权重被此前恢复点保留逻辑清理，不能恢复或称38k为历史真正best。这是保存实现缺陷，
  后续训练须独立保留best。预计停止与核验为分钟级；660数据的执行标签合同需先核对，
  缺实际PD标签时不得静默替换为关节位置或把不同来源的手/物体轨迹配成完整机器人执行。
  停止操作限本卡owned进程，外部数据只读，不需要追加权限。
- 用户已授权四卡、80,000步及较大batch，本次按三域预训练解释。
- 已有20项架构合同测试及真实PTv3/robot/native短smoke；长期运行调整后22项测试通过。
  这些仅证明接线，不证明学习效果。
- 选择每卡128、global batch512；单卡128负载检查通过，峰值分配显存约12.8GiB，
  稳定约2.15秒/步，比每卡64的样本吞吐约高20%；四卡DDP四步也已通过，约230–245样本/秒、显存约14–16GiB/卡。
- 不扩展架构或数据，不改变核心claim。单seed长期训练仍是探索性训练，不是matched Validation。
- 初步预算约2–3天，硬上限7天/四卡；4GB单run、400GB共享产物预算保持不变。
  非有限loss/梯度、OOM、外部GPU进程、源码变化、30分钟无进展或资源上限触发停止。
- 成功后保留checkpoint进入airplane适配设计；失败保留日志和最近恢复点，先定位实际问题。
  没有需要追加的资源/权限授权。

## 固定协议

- 配置：[pretrain_80k.json](../../../configs/pretrain_80k.json)。80,000是optimizer updates，
  不含额外梯度累积；每步全局512样本，合计40.96M次有放回窗口抽样。
- 物理GPU4/5/6/7，NCCL DDP；启动前确认空闲。模型全量更新，bf16 autocast。
- OakInk2/GRAB/ARCTIC原TRAIN/dev划分，来源权重5:2:2和原category sampler。
  保持TRAIN-only统计、H4、30Hz、horizon24、每物体32点。
- 仅从已有PW step50000迁移场景编码器，不加载四步engineering预测器作为科学初始化。
  原PTv3结构不变；256 scene、22 hand、32 goal共同进入4层128D共享Transformer。
- 目标：分别归一化的hand/scene MSE，既有motion weighting不变。
  AdamW；PTv3/scene lr1e-5，其余1e-4，weight decay0.01，clip1。
  增大batch时不自动线性放大学习率，也不新增scheduler。
- 主种子100，rank初始化100–103；每步按主种子/step/rank派生独立抽样子流。
  固定dev panel seed104，共384窗、每source128；rank不是独立科学重复。
- 每1,000步验证：各source hand first8/all24；scene h1/first8/middle(9–16)/h24/all24、
  target/non-target、moving/static，以及goal-only线性插值基线。
- 主要读hand first8/all24跨source的变化。scene重点读first8/middle及non-target，
  不把目标h24近零当作独立动力学收益；无有效样本的类别报告null和count0。
- 每2,000步保存恢复点；保留step1、20k/40k/60k/80k及最近一个恢复点。
  删除范围仅限该run的登记表中被替代的非里程碑恢复点，完整验证曲线与其他run不删除。
  最多同时约6份617MB文件（含写入暂存），仍低于4GB。
- 每20步打印loss、GPU、step time、样本吞吐和ETA；监督进程每30秒记录GPU快照。
  首次梯度审查用实际小batch，避免保留FP32图限制bf16训练batch。

## 运行与追溯

实际运行commit、输入路径、worker PID、config和初始化记录以run的manifest为准。
本次实际运行commit为 `63bde79a`，监督PID1960948，torchrun PID1961023，
worker PID1961038–1961041；GPU4/5/6/7。产物：`outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r1/`；
launcher日志、GPU采样和scratch位于同名项目`tmp/`目录，训练目录提供对应链接。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/launch_pretrain.py \
  --detach --gpus 4,5,6,7 \
  --config src/task/ptv3-shared-wam/configs/pretrain_80k.json \
  --experiment-card src/task/ptv3-shared-wam/docs/experiments/probes/P-20261013-ptv3-shared-main3-80k.md \
  --output outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r1
```

工程负载记录：`engineering-batch64-20261013-r1`、`engineering-batch128-20261013-r1`；
四卡检查（GPU0/1/2/3）：`engineering-ddp4-batch128-20261013-r1`。均为四步，不能据此形成训练效果结论。

## 启动观察

- 首个step1完整checkpoint已保存，`engineering_only=false`、world size4、target80000。
- 启动记录观察至step80，loss/梯度有限；最近20步约1.82秒/update，约280.6样本/秒，
  当时ETA约40.5小时。随真实数据/I/O及验证成本更新，不把初始化首步ETA作为最终估计。
- rank0峰值分配约14.3GiB。四卡物理显存含allocator缓存约18–23GiB，明显高于四步
  负载检查的缓存占用；目前无OOM，监督进程继续记录变化。GPU利用率有批次/同步波动。
- 验证尚未到达1,000步，不能用这些训练loss判断预训练质量。
- 实时状态、完整输入与初始化追溯见run的 `manifest.json`、`progress.json`、
  `process.json`、`launcher.log` 和 `gpu.jsonl`。

## 早期利用率诊断（运行保持继续）

用户询问GPU利用率偏低。持续30秒dmon采样，GPU4/5/6/7的mean分别为
40.3/40.4/40.6/42.9%，范围0–100%；是持续均值偏低，不只是一次采样。
四个rank的主线程各约102% CPU，训练约1.8秒/update、280多样本/秒。

纯CPU输入路径控制：同主种子100、rank0、step101–104、batch128，不执行神经网络。
无cProfile干扰时，首次1.23秒，后三次总准备时间0.877/0.848/0.889秒，
包含约0.08–0.09秒collate。这是独立进程中的同路径测量，并非精确划分原训练的所有时间。
有cProfile的四batch归因显示序列读取、np.load/open_memmap/header解析及几何特征构造占主导；
512样本中调用SourceWindows.sequence471次，缓存32项对当前随机大batch命中率很低。

当前训练没有DataLoader workers/prefetch：逐样本同步draw→collate→阻塞to(device)→GPU。
数据准备本身约相当于当前step时间的一半，足以解释GPU的大量空闲。
优先改进是多worker预取、pinned memory/异步传输及有界序列缓存；还可检查逐样本
scene token采样同步。DDP等待和token采样的精确时间占比尚未做CUDA profiler，不能
把它们写成已经确认的主因。未修改运行中的模型/训练源码，也未中断80k任务。

复现工具：[profile_batch_preparation.py](../../../tools/audit/profile_batch_preparation.py)。
证据保存在同名run scratch下的 `utilization-window.txt`、`input-preparation-profile/`、
`input-preparation-control/`；只包含工程吞吐与CPU路径，不构成学习效果证据。

## 数据管线优化与checkpoint边界续训

用户进一步授权优化。保持模型、batch128/global512、学习率、80k总步数、源权重、
标签和step/rank抽样规则；运行参数单独记录在provenance.execution，不改模型config。
采用每rank4个spawn workers、prefetch factor2、persistent workers、pinned memory与
non_blocking传输，worker内每source有界缓存512个序列；去掉已生成标签后不使用的
legacy传输张量。worker IPC使用项目短路径，避免长run/rank名超过AF_UNIX108字节限制。

- 25个测试通过，含原抽样与resume suffix、worker顺序、主torch RNG不受影响、短socket。
- 真实四batch共88个tensor与同步版本逐位一致，CPU主RNG一致，缓存未改变标签。
- 单卡128预取smoke完成：稳定约1.22–1.32秒/update，取batch等待0.5–1.8ms；
  同规模同步约2.1秒。多worker初始启动约40秒，之后复用。
- step2→step4恢复检查：CPU/CUDA RNG一致，最终验证objective差3.58e-7、
  各EPE mean最大差9.85e-5mm。GPU参数未逐位一致（最大1.42e-4，集中于attention K-bias），
  不声称逐位确定的优化轨迹；观测的预测误差差异远小于1e-3mm。
- scene token采样单独测为17.2ms，批量原型0.43ms、所选tokens相同；当前总耗时贡献较小，
  不在本次生产切换中改scene sampler。

### 切换边界

原r1仍运行至step2000。已验证的training补丁暂存为Task-local runtime patch，
只在完整step2000 checkpoint原子保存后停止本任务的已核对torchrun组，才改动被监测的源码。
补丁检查要求源码与原运行base完全一致；不覆盖用户或其他Task修改。
随后提交代码，从模型/优化器/每rank RNG恢复到新run
`pretrain-main3-ddp4-80k-20261013-r2`，目标仍为总step80000。
原checkpoint和r1全部证据保留，dev curve拷贝延续，监督时间扣除r1已用时间以保持总7天上限。
切换后需观察至少40次更新和四卡持续利用率，才能报告实际长跑收益。

工具：`tools/run/switch_prefetch_at_checkpoint.py`；切换状态在r1 scratch的
`prefetch_migration.json`。工程记录：`engineering-loader4-batch128-20261013-r2`、
`engineering-loader-resume-continuous-20261013-r1`、`engineering-loader-resume-restored-20261013-r1`。
旧loader prototype r1没有完成首步，发现项目长socket路径问题后停止；不作学习负证据。
实际切换已完成，代码commit `8510ec3a`。从r1 `step2000.pt`恢复模型、优化器和每rank RNG，
新run r2的manifest记录resume_parent及PW初始化来源，验证曲线延续。新监督PID2062864，
worker PID2063036–2063039，GPU4/5/6/7、global512、目标仍step80000。
原r1的进程退出由此次切换引起，不是方法负证据；原checkpoint/日志完整保留。
观察至2040：稳定0.965–1.006秒/update、509–530样本/秒，data_wait约1.24ms；
原r1稳定约1.83秒/update、280样本/秒。观察至2140仍约0.97秒/update、527样本/秒、
取batch等待约1.2ms。新30秒dmon窗口四卡均值90.7/90.6/89.8/89.8%，
旧窗口40.3/40.4/40.6/42.9%。早期吞吐约1.8–1.9倍，剩余ETA约21小时。
这是工程性能结果，不是抓取或学习效果结论。

## 2026-10-14 固定dev预测诊断（validation step 37000）

本次为既有Probe的工程审计与中期读数，不新建训练或改变seed。问题是：手轨迹跨域
误差差距是否来自指标/单位错误，以及物体预测是否优于直接使用已知goal。
结果决定是否先修数据或指标；最便宜的检查是复算实际metric函数，并读取原384个
固定dev窗口。标签统计使用CPU，因为只涉及数组读取和几何计算，不执行神经网络；
模型误差来自原训练的已保存dev curve，没有启动额外GPU评估。

### 指标语义与手轨迹

`hand_all24`是未来第1–24步、有效窗口/手/语义点三维欧氏距离的点数加权平均，
单位mm；不是仅第24步误差。30Hz下终点为0.8秒。每个域128窗口，当前面板全部
2×11手点有效，first8/all24的count分别为22528/67584。
实际`forecast_metrics`复算：每步人为增加1、2、…、24mm误差，得到
first8=4.5mm、all24=12.5mm。聚合与该定义一致。

| 数据域 | 模型第1–8步 | 模型第9–24步 | 模型第1–24步 | 保持当前手不动，第1–24步 | 当前速度外推，第1–24步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OakInk2 | 5.76 | 24.42 | 18.20 | 29.80 | 24.70 |
| GRAB | 22.76 | 121.94 | 88.88 | 143.46 | 148.84 |
| ARCTIC | 16.58 | 69.74 | 52.02 | 80.29 | 99.63 |

表中单位均为mm。后16步由原first8/all24的sum/count相减得到；原日志没有单独的
hand h24，不能从后16步平均推断最后一帧误差。保持当前手不动的基线误差即真值
相对当前手的位置变化；其h24分别为54.76/267.05/142.22mm，均不是模型h24误差。
当前速度外推使用观测到的手点速度与实际1/30秒步长，不读取未来手点作为输入。

三域模型相对保持当前手不动改善约39%/38%/35%。GRAB、ARCTIC确有更大的
标签运动幅度；当前手点到腕部距离中位数约95.23/96.02/92.13mm，未见明显
米/毫米倍率差异。逐窗口检查future action位置=当前手+trajectory target，
有效标签有限；原loader统一在当前anchor坐标中处理历史和未来，并检查30Hz时钟。
这些检查没有证明所有原始标签都正确，也不能解释全部剩余误差。
GRAB后段约122mm、ARCTIC约70mm仍然很大，不能以跨域运动幅度不同为由视为足够精确。

### 物体表面点轨迹

物体误差是同身份表面query点的轨迹EPE，不是物体质心或姿态角误差。
goal-only基线让目标点从当前位置线性到已提供的t+24 endpoint，非目标物体保持不动。

| 数据域 | 目标全24步：模型 / 插值 | 目标第9–16步：模型 / 插值 | 运动物体分组：模型 / 插值 | 静止物体分组：模型 / 插值 |
| --- | ---: | ---: | ---: | ---: |
| OakInk2 | 4.38 / 3.29 | 5.32 / 4.79 | 10.71 / 15.06 | 1.66 / 0.14 |
| GRAB | 23.72 / 33.16 | 31.53 / 46.34 | 29.61 / 43.71 | 5.27 / 0.15 |
| ARCTIC | 17.47 / 19.65 | 23.59 / 28.06 | 23.29 / 38.83 | 5.80 / 0.53 |

单位均为mm。运动分组按每物体整个窗口的最大点位移>=2mm划分，组内所有有效点均
进入平均，不能把它说成只统计位移>=2mm的单独点。GRAB/ARCTIC目标中段比插值
低约32%/16%；OakInk2中段仍差约11%，静止物体漂移三域均差于插值。
目标h24已经给模型作为goal，插值终点误差近零是构造结果，不能拿终点分数证明
独立物理预测。物体头从5k到37k仍有改善，例如目标全24步
OakInk2为6.31→4.38、GRAB为30.28→23.72、ARCTIC为20.55→17.47mm。
整体dev objective约5k的1.58→37k的1.69、中途22k最低1.55；整体平台不等于
物体头停止进步，也不据此将尚未保存的best dev状态当成可用checkpoint。

局部判断：指标聚合检查通过；预测学习信号PROMISING，长期手精度与静止物体
漂移仍是限制。维持已授权80k训练，不改正在运行的源码、标签或统计，不升级成
接触动力学、机器人适配有效或抓取成功的正式结论。

复现工具：[audit_hand_metrics.py](../../../tools/audit/audit_hand_metrics.py)。它固定读取
指定dev step并要求原面板全部有效，遇到不同mask时退出，避免无效点拉低基线。

```bash
TMPDIR="$PWD/tmp" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam/tools/audit/audit_hand_metrics.py \
  --run outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r2 \
  --validation-step 37000 \
  --output outputs/ptv3-shared-wam/hand-metric-audit-20261014-r1/summary.json
```

原始临时复算保留在项目`tmp/ptv3-hand-metric-audit-20261014.{py,json}`；正式复算
摘要位于上述outputs目录。模型分段结果、物体对照与历史趋势来自r2 `curve.json`；
本节记录固定37k快照，不随仍在更新的progress/curve改变。

## Limitations / future evidence

单seed、三域dev曲线不替代匹配对照、多seed或机器人闭环。未来需评估完整airplane适配、
固定pair下K-step执行与最终Cm增益。视频/RGB输入、EgoDex、旧架构matched比较和
任意goal反事实正确性不属于本轮，当前不占用四卡预算。
手误差后续需要按逐帧、腕部平移/相对手形和交互阶段拆分，并结合可视化核对原始
标签；当前尚未做这些检查，不能归因于某个具体网络模块或宣布数据无误。
