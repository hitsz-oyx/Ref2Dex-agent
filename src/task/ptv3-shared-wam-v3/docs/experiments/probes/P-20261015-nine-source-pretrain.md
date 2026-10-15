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
status: STOPPED
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

## 启动阶段标签尾部审计

step140全局loss22.596（scene21.707），step160回落4.332；step220为3.479，
未发生非有限量或持续发散。为区分数值故障与标签尾部，CPU重放该步两rank共256个
确定性TRAIN窗口（不额外调用模型/GPU）。source4/rank1/position37的物体未来24帧
最大位移7.387m、RMS4.200m，单窗零标准化预测scene loss6569.4，足以造成batch尖峰。
其余top样本包括GRAB大动作。记录位于recipe目录`step140-target-audit.json`，
复现入口`tools/audit/audit_sampled_targets.py`。此审计不能证明模拟运动本身错误；
不按未来位移或成败自动删除样本，不静默改变冻结配方。梯度clip1继续生效；
若持续由尾部主导或固定DEV恶化，再做能区分数据语义与优化问题的最小Probe。

共同归一化实际手std：h1约5.3–5.6mm，h24约101–117mm；scene h1约2.3–2.5mm，
h24约39–40mm。hand/scene各24×3，九源共享，只有TRAIN参与拟合；物理EPE仍反归一化。

## 用户停止本地训练 / 后续远端四卡

用户明确要求停止本地WAM v3，后续在远端四卡训练。已写用户STOP回执，GPU1/3确认释放。
最后日志step440，loss2.197、稳态0.898步/秒；没有到达1000步DEV，因此有效性为UNCLEAR。
本地长训无新恢复点（checkpoint interval1000）；保留启动step4完整checkpoint与全部日志。
进程最终资源guard记为FAILED：停止同期GPU0/2有其他任务出现，合计超过6卡，
supervisor仅终止本run自己的进程组。保留实际失败回执与独立user-stop-receipt，不重写原始证据。
这不是方法负结果；后续不恢复本地训练，远端等待数据传完和质量审查。

独立raw复核（脚本提交0dbcb1dc）确认7.387m来自原始UniDexGrasp++仿真，
compact与raw[::2]完全一致，query重建误差2.38e-7m；60Hz某帧16.7ms位移0.465m、
差分速度27.90m/s、加速度约1677m/s²。CAD约3.45×11.35×3.45cm。
这是原始物理质量疑点，不能由格式有效性排除；viser可直接查看完整序列与速度曲线。


## Motion质量清洗后的输入（2026-10-15）

用户要求先清洗，未启动远端训练。全量扫描33,056条source4；新view隔离25条轨迹中的114 TRAIN/22 DEV窗口，其余八源窗口集合不变。新recipe `nine-source-motion-recipe-20261015-r1` 为4,806,030 TRAIN/821,613 DEV；保留原三源4096/source抽样，精确扣除被删除TRAIN贡献，重新计算配比与24-horizon归一化。独立114窗reader统计回加误差≤7.28e-12；本地96窗/2638CAD复核通过。

远端独立`data-motion-r1/recipe`已CPU验收通过（37,591payload/2,638CAD及36 TRAIN/32 DEV mixed样本）；与本地新的数值统计完全一致，已知row14909的59/67/75不准入但raw仍可读。原数据和旧配方保留。新配方改变了目标归一化，下一轮从头初始化，不直接恢复旧统计下的step4优化器状态。这里只记录数据工程进展，不形成模型效果结论。

规则是保守疑点隔离，不保证物理异常全覆盖；边界9674/17266仍列可视化复核清单。完整协议和限制见[清洗记录](../../../../3d-data-preparation/docs/research/2026-10-15-unidex-motion-cleaning.md)。


## 远端四卡motion全量启动 Decision Note（2026-10-15）

用户已明确要求“你直接开始吧”。本次run_id为`pretrain-nine-motion-remote4-80k-20261015-r1`，执行固定代码`6e535bc1077e1593d28a83d01b485b5a09cb8075`，短代码目录`/home/wbcd/workspace/oyx_ws/wam-v3-20261015/repo`；该版本的模型、采样、TRAIN归一化/训练循环与当前实现相同，PointWorld只读借用已部署vendor。输入为远端`data-motion-r1/recipe`，4,806,030 TRAIN/821,613 DEV。所有模型和优化器从头初始化，不加载旧归一化下checkpoint。

继续原实验问题：九源统一模型能否稳定优化，固定八源DEV物理误差是否改善，结果决定是否继续80k预算或先诊断；不改变最终claim。清洗前本地只验证了双卡工程链路；现在由启动阶段的九源梯度审计、初始DEV、第一步完整保存及随后的连续四卡更新，最低成本检验新视图/远端DDP路径。

资源核查：本地8卡当前均无计算占用，远端4×4090各49,140MiB空闲，使用远端0/1/2/3，符合全项目6卡上限。采用既定world4配置，每卡128/global512、80,000更新、warmup500、lr1e-4、bf16/TF32、每1000步DEV/保存；最长48h、产物8GB/run、磁盘余量20GiB。rank0初始DEV/保存期间其他rank等待，首步后再估计稳态吞吐与ETA。只允许停止own进程组；非有限loss/grad、资源超限、无进度30min、STOP或期限停止。

当前执行：准备启动；实际状态以远端process/progress.json为准。启动成功后记录PID、GPU实测、速率和ETA。
