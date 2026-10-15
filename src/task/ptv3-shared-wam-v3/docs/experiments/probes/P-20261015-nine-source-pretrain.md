---
schema: ref2dex.probe.v2
probe_id: P-20261015-nine-source-pretrain
experiment_id: P-20261015-nine-source-pretrain
date: 2026-10-15
task: ptv3-shared-wam-v3
branch: main
git_commit: 67f7af78
claim_id: C1
hypothesis_family: HF-nine-source-pretrain
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: continue nine-source pretraining when sustained full-batch updates and recovery pass and fixed physical DEV errors improve
decision_changed_if_negative: inspect source semantics and actual input/loss chain before spending the remaining training budget
status: RUNNING
run_id: pretrain-nine-80k-20261015-r1
---

# 九源统一预训练启动与持续运行

Class: Decision. 服务于 MISSION 的短期手物转移表示。先区分九源统一输入/共同统计是否
能够持续训练与恢复，再观察同一固定 DEV panel 的物理手/物体预测误差是否改善。
最便宜的启动判别是全模型梯度检查、满 batch、双卡、checkpoint 恢复；通过才继续长训。
离线误差与训练 loss 不能证明 Cm 或最终机器人收益。

## Decision Note

用户明确要求第一轮合并九源并尝试大规模训练。采用现有 v2 架构从头初始化，九源共享
TRAIN 统计；原 main3 内分层不改，新六源只使用冻结的 verified encoder-compatible 窗口。
这一 recipe 显式授予本次训练用途，保持清洗产物的 engineering-only 标记不变，
不包含 HOcap、H2O TEST 和已隔离的越界窗，不覆盖旧 checkpoint。

source 概率 sqrt(N)，DexYCB 上限1%；全局 point-mask loss，hand/scene系数1:1，
无额外 source loss 权重。main3 各4096次 TRAIN 抽样估计统计，native 全量精确统计。
采用500步线性warmup到1e-4，AdamW decay.01/clip1，bf16/TF32。
统计seed19只用于数组采样，不是训练重复。DEV seed171，每个实际DEV源32窗；
DexYCB没有 DEV，不虚构；八源宏平均 objective用于恢复点选择，物理EPE按源报告。

执行固定代码 detached 工作树。2GPU，每卡128/global256，最多80000步/48小时/8GB；
保持全项目最多6卡，不使用其他任务临时7卡例外。先分段检查保存与恢复，然后持续训练。
停止条件：非有限损失/梯度，GPU预算或磁盘边界，30分钟无进度，STOP，超时。
成功不升级为正式科研结论；正式 Validation 和机器人收益作为 future evidence。

## Results

统计 `nine-source-recipe-20261015-r1` 完成，146.71秒CPU，TRAIN总数4,806,144。
九源概率顺序OakInk2/GRAB/ARCTIC/UniDex/TACO/HOT/DexYCB/H2O/HOI4D：
46.0886/10.6770/13.0795/12.0626/4.1012/6.2535/0.4395/1.7696/5.5285%。
main3抽样标签在九源collate中逐tensor与旧collate完全一致（九项）；九源混合真实batch通过。
5项测试通过，包括不等有效点DDP归一化的loss及梯度等价、stateless抽样恢复、完整optimizer/RNG保存。

启动检查 `startup-nine-ddp-20261015-r1` 已完成4步，每卡128/global256，固定代码f62ca8f8，
GPU1/3，耗时52.26秒（含梯度检查、DEV和worker启动）；保存step1与step4完整checkpoint。
首步实际optimizer batch覆盖全部九源；梯度与loss有限。rank0最终峰值分配12.96GiB，reserved17.62GiB。
下一步从step4原样恢复至80000步，继续观察稳态吞吐与物理DEV误差。

多卡损失按各rank有效点/scene运动权重分母全局归一化，补偿DDP的梯度平均；
不引入来源loss乘数。checkpoint milestone40k，保留step1、best/latest和恢复点，
避免80k阶段的checkpoint数量超过8GB预算。

## 持续运行与恢复

`pretrain-nine-80k-20261015-r1` 已从step4完整恢复，固定代码67f7af78，
GPU1/3，supervisor1584518；recipe、统计、world size和配置完全保持。
step40/60稳态0.933/0.918步每秒（全局239/235窗每秒），预估80k约24小时。
两卡利用率接近100%，rank0峰值分配13.64GiB、reserved21.15GiB，当前无OOM/非有限量。
启动与恢复通过属于工程证据；仍需1000步固定八源DEV变化判断学习进展。
用户追加九源远端落盘需求，已并行处理；不改变本地训练配方。
