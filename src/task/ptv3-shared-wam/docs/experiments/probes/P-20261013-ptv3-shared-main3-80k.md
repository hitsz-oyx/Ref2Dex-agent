---
schema: ref2dex.probe.v2
probe_id: P-20261013-ptv3-shared-main3-80k
experiment_id: P-20261013-ptv3-shared-main3-80k
date: 2026-10-13
task: ptv3-shared-wam
branch: main
git_commit: 63bde79a
claim_id: C1
hypothesis_family: HF-ptv3-shared-task-forecaster
probe_index_in_family: 1
seed_pool: probe
seeds: [100]
decision_changed_if_positive: transfer the trained shared forecaster into airplane full-pair adaptation
decision_changed_if_negative: inspect the forecast contract and data before spending on robot adaptation
status: RUNNING
run_id: pretrain-main3-ddp4-80k-20261013-r1
---

# 四卡三域共享预测器预训练：80,000 updates

Result: RUNNING. 四卡每卡128/global512正式预训练已启动；首个完整checkpoint已保存，早期loss/梯度有限，尚无学习效果或抓取结论。
Decision: 固定ref1_2的共享goal架构，完成负载检查后启动80,000次优化器更新。

Class: Decision. 服务于自主灵巧操作的预测器基础，而不直接验证Cm增益。具体问题是：
该共享架构能否在三域实际手物运动上学到手轨迹和超过endpoint插值的中间场景预测。
正向信号支持下一阶段airplane完整forecaster+R适配；无手轨迹学习或中间预测收益时，
先查本Task数据/合同，不把human预测误差当作机器人抓取成功率。

## Decision Note

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
实际切换commit/PID/性能回填本卡。

## Limitations / future evidence

单seed、三域dev曲线不替代匹配对照、多seed或机器人闭环。未来需评估完整airplane适配、
固定pair下K-step执行与最终Cm增益。视频/RGB输入、EgoDex、旧架构matched比较和
任意goal反事实正确性不属于本轮，当前不占用四卡预算。
