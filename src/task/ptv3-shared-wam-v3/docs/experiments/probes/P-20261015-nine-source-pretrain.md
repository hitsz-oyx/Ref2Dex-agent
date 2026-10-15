---
schema: ref2dex.probe.v2
probe_id: P-20261015-nine-source-pretrain
experiment_id: P-20261015-nine-source-pretrain
date: 2026-10-15
task: ptv3-shared-wam-v3
branch: main
git_commit: 6e535bc1077e1593d28a83d01b485b5a09cb8075
claim_id: C1
hypothesis_family: HF-nine-source-pretrain
probe_index_in_family: 1
seed_pool: probe
seeds: [170]
decision_changed_if_positive: continue nine-source pretraining when sustained full-batch updates and recovery pass and fixed physical DEV errors improve
decision_changed_if_negative: inspect source semantics and actual input/loss chain before spending the remaining training budget
status: RUNNING
run_id: pretrain-nine-motion-remote4-80k-20261015-r1
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

已于14:07提交执行，supervisor1877848、torchrun1877923；四卡进程已启动，rank0九源gradient-audit与初始固定DEV curve已写出。正等待首步checkpoint及稳态速率；实际状态以远端process/progress.json为准。


### 远端四卡启动实测

`REMOTE_FOUR_GPU_STARTUP_PASS`：已连续更新至step120。四卡利用率98–100%，物理显存约21.9/26.5/28.9/31.6GiB（按卡实际0/1/2/3分别28.9/26.5/21.9/31.6），无OOM/非有限loss或gradient。rank0 peak allocated14.30GiB/reserved28.37GiB；九源hand/scene梯度进入PTv3/shared/hand_encoder/goal路径，初始固定八源DEV有限。

step1完整checkpoint已写入1,183,806,235bytes并越过多卡保存barrier；各rank RNG随checkpoint保存。step20–120区间约1.22–1.47更新/秒，近期1.40–1.47（约715–753窗/秒）；纯更新ETA约15–16h，含DEV/保存与吞吐波动暂估16–18h。当前仅为启动与稳定运行工程证据；首个训练后固定DEV在step1000，尚不形成模型效果结论。

输出：`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/outputs/ptv3-shared-wam-v3/pretrain-nine-motion-remote4-80k-20261015-r1/`。本地只读快照`tmp/nine-remote-20261015-r1/remote-motion-startup-audit.json`与启动监控JSONL可追溯；远端自己的supervisor持续记录GPU/进度并执行48h、磁盘、产物、无进度和STOP边界。


## 人工判定后九源严格准入与连续恢复（2026-10-15）

用户确认UniDex9674/17266与HOI4D37130异常，授权模棱两可者直接隔离、阈值设紧；详细Decision Note见[运动复核](../../motion-followup-20261015.md)。全39,528条运动审计完成；固定物体/有效手点速度>5m/s或SO(3)角速度>40rad/s，使用真实观测间隔，H4+K24覆盖异常转换的窗排除，UniDex异常后尾段排除，HOI4D37130另排初始0..30。manual-r2外再隔离2,943 TRAIN/204 DEV窗，有效4,803,076/821,409窗；可能丢弃正常快动作，不证明剩余物理质量全通过。

远端严格recipe为 `data-strict-r1/recipe`，实际固定代码3709c966。保留父数值normalizer和固定source概率作为此前TRAIN拟合参考，optimizer/每rank RNG/学习率全局步数恢复。新DEV面板已改变，不把前后曲线视为匹配比较。独立6合同测试、实际subset guard及本地/远端真实mixed-reader通过，原始数据不覆盖。

原motion-r1在完整step6000后切manual-r2；manual-r2在完整step8000后切严格版本。新run_id `pretrain-nine-strict-remote4-80k-20261015-r1`，supervisor1892360/torchrun1892489，远端0/1/2/3，每卡128/global512，仍总80k/48h。完整checkpoint中的4个normalizer buffers与strict stats逐个相等、4rank RNG及optimizer保留。已实际继续至8080，loss0.6652/grad4.5147有限，初始八源DEV和九源gradient audit通过；约1.465步/秒，显存17.7–36.1GiB、利用率82–99%，剩余暂估14–16h。工程证据在远端scope的strict-quality-startup-audit.json及本地同名快照；不形成预训练效果/机器人收益结论。

Future evidence：速度阈值无法覆盖缓慢错误、手物空间错配、接触质量及原始标注系统性误差；不为当前继续训练追加全量接触重演。将来形成模型收益结论须固定共同评估集合，避免DEV panel变化混淆。

## 2026-10-15 19:41 CST plateau check (same strict DEV panel)

Read-only snapshot: remote strict run28040/80000 (35.1%), best step27000, latest28000; four GPUs82–100% utilization,29355/42775/41581/38051MiB, finite loss/gradient. Steady1.44–1.47updates/s, remaining about10–11h including periodic validation/checkpoints. No process, LR, input, checkpoint or schedule was changed.

Compare means of the four fixed DEV evaluations at21–24k versus25–28k: TRAIN logged-batch mean0.42597→0.38488 (-9.65%); eight-source macro DEV objective1.11788→1.10543 (-1.11%); macro hand-all24 physical EPE34.554→34.418mm (-0.39%), hand-first8 11.445→11.311mm (-1.17%), scene-all24 14.228→13.943mm (-2.00%). Best DEV at22k1.076592 versus27k1.067248, only0.87% lower over5k. This is slow validation improvement/near-plateau behavior, not completed optimization: TRAIN is still descending. HOT hand/scene essentially flat, H2O hand worsens28.03→29.28mm across these blocks; TACO hand26.37→25.29mm (-4.08%), GRAB scene24.53→23.41mm (-4.58%), ARCTIC first8 13.687→13.088mm (-4.37%). Source trajectories differ, so neither a blanket convergence nor blanket healthy-generalization claim is justified.

Panel is only32windows/source (256 total), correlated fixed samples; these are descriptive Probe trends, not formal convergence or overfitting validation. Constant LR1e-4 after500warmup has no later decay. Decision: keep authorized run to40k milestone (~2.4h from snapshot), recheck same-panel and per-source physical metrics; if improvements stay this small, a bounded lower-LR continuation comparison is more informative than blindly extending at the same LR. Do not autonomously alter the live run in response to this status request. No architecture or FM conclusion follows from the plateau check.

Evidence: `outputs/ptv3-shared-wam-v3/plateau-check-20261015-1941/{snapshot.json,summary.json,learning_curves.png}`. Analysis script is an artifact beside the immutable snapshot; CPU array reduction/plotting only, no new model evaluation.

## 2026-10-15 21:19 CST: validation plateau persists at36k

Read-only remote snapshot36420/80000 (45.5%), best31000 objective1.03455, latest36000 objective1.11735; four GPUs100% utilization,29.4/42.8/41.6/38.1GB, finite loss/grad,1.466updates/s. Compared29–32k versus33–36k evaluation blocks: TRAIN0.34739→0.32162 (-7.42%); DEV1.09586→1.11683 (+1.91%); hand-all24 macro EPE34.082→34.389mm and scene-all24 13.648→13.832mm both slightly worse. Best31k still improves on old27k best1.06725 by3.06%, but no new best over32–36k. Validation plateau/oscillation evidence is stronger while TRAIN continues improving; small fixed256-window panel prevents a formal overfitting conclusion. Keep live run unchanged;40k milestone is about40–45min away and is the planned point to review a lower-LR continuation. Remaining80k ETA about8.5–9h including validation/save overhead. Artifacts `outputs/ptv3-shared-wam-v3/plateau-check-20261015-2119/` contain snapshot,summary and updatedlearning_curves.png.

## Decision Note: user-authorized lower-LR continuation (2026-10-15)

User explicitly requests lowering LR now. Latest36k evidence: TRAIN continues down while fixed DEV worsens slightly; best31k and five evaluations without refresh. Choose latest completed checkpoint after request (target37k) and reduce both geometry/shared LR1e-4→3e-5, preserving exact strict recipe, numeric normalizer, optimizer moments/step, four rank RNG, panel, global512, seed170 and cumulative80k/48h cap. The only config difference is the two LRs; no expanded DexYCB or boundary-filter data switch. Record the gate as an explicit LR-only resume, never allow arbitrary config relaxation. Preserve old best/latest/run; bounded4GPU20-update engineering smoke checks actual LR, model/optimizer/RNG restoration, finite updates and same-panel DEV, then exact full-state continuation in a fresh directory. This is an adaptive single-trajectory Probe, not a matched LR experiment or proof that lower LR improves generalization. Failure stops only the new owned group and preserves the37k recovery point. Recheck after2–3k lower-LR updates; do not claim improvement from a20-update smoke. Estimated transition/smoke several minutes, remaining long train8–9h, within existing resources. No new external authorization needed beyond the current explicit user request.
