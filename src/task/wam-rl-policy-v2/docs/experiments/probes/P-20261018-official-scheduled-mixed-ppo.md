---
schema: ref2dex.probe.v2
probe_id: P-20261018-official-scheduled-mixed-ppo
experiment_id: P-20261018-official-scheduled-mixed-ppo
date: 2026-10-18
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 4
seed_pool: debug
seeds: [0]
decision_changed_if_positive: retain upstream PPO as the common trainer and expand mixed-object training
decision_changed_if_negative: diagnose the BC/WAM interface under upstream PPO before changing the algorithm
status: RUNNING
run_id: official-scheduled-mixed-20261018-r1
---

# 六卡混合物体：复用官方 PPO + 自训练 BC + WAM-v4

Class: Decision。用户要求六卡混合训练，并明确要求优先复用已有官方PPO实验，
避免重新实现训练算法。此协议取代尚未启动的自定义六卡PPO计划。
假设：既有全量scheduled BC可通过官方PPO底座在混合TRAIN物体中继续学习。
最低成本路线：已有官方run/update/storage + 薄观测适配；先六卡32iteration工程检查，
随后直接启动1000iteration训练，不用小样本表现作为大规模训练前置门槛。

依据旧Task的 P-20261016-official-unidex-ppo 与 P-20261017-wam-runtime-correction：
固定上游6c9e710b；复用 `third_party/UniDexGrasp2-official-ppo` 和 process_ppo，
不是重写GAE/collector/update。旧实验曾有重复residual和全批history清空缺陷；
本次没有residual adapter，每环境独立history/reset，三条动作均值路径测试一致。

保留官方cfg：nsteps8，5epoch×4minibatch，共同Adam3e-4，gamma.96，lambda.95，
adaptive KL .016，原reward，native episode/reset/random_time=True。
actor/critic隐藏层1024/1024/512/512 ELU，critic官方初始化。
官方scale_tril=exp(log_std)^2和官方KL尺度约定均保留；init参数.8，实际std.64。
不移植上一轮自定义PPO的硬KL回滚、小critic或std.05。

必要差异：
- actor均值替换为全量scheduled BC (best46000)，包括其冻结归一化，全部权重可训练；
  不加载官方actor checkpoint。
- 565D current300+world tau264+valid1，冻结WAM-v4 step12000。
- 六卡平均梯度、全局KL、全局advantage均值/无偏std；单卡公式与上游一致。
- 多卡总192环境，六个互斥TRAIN分片各16物体×2环境，共96物体，非全量在线物体池。
- 每run记录上游源码的精确最小替换列表；PPO.run不改，GAE递推不改。

资源：本地GPU0–5，GPU6/7不分配本任务，已获共享monica轻负载授权。
先32iteration/96物体/192环境工程检查（每卡640optimizer steps），<=900秒。
随后1000iteration，共1536000训练环境步，<=7200秒；每200iteration及首/末端点保存，
仅rank0保存模型和optimizer，原BC不覆盖；小型各rank曲线与资源日志保留。
运行代码复制冻结，旧数据与模型只读复用；磁盘20GiB余量停止线；任一rank失败则
supervisor终止本任务其余进程组，不触及其他任务。STEP与checkpoint版本分别记录。

入口：`tools/run/launch_mixed_ppo.py`，默认使用官方engine。
旧自定义PPO入口仅留作历史工程诊断，不作为本轮六卡训练器。
23项测试通过，包括上游PPO.run AST不变、分布/动作路径一致、TRAIN分片与实测物体绑定。
性能、成功率与稳定保持均待实测；官方探索噪声可能损害BC控制，须记录而非事后改配方。

## 六卡工程检查通过

r1资产路径预检未进入训练；实际worker r1因原生scale_buf为dict而停止，
修正身份读取后r2完成，代码ce93103b。六卡32iteration共49152环境步，
每rank严格640个官方Adam step，全部learning-audit PASS；动作路径误差0，
首/末端点六卡参数误差0，各rank跨过43次环境reset。
全程136.6秒，纯训练115.5秒，约3.61秒/iteration；单卡约4.1GiB总显存。
预计1000iteration约1小时。注意smoke过程manifest的legacy ticks=180未参与官方循环，
实际nsteps由保存的官方training-config.json确定为8；正式训练metadata已明确记录8。

正式运行 `outputs/wam-rl-policy-v2/official-scheduled-mixed-20261018-r1/` 从同一BC重新初始化，
不从32iteration工程端点继续选模。加入被动episode计数/native success/held60+lift10cm，
不改reward或更新。tmux持久化supervisor，固定代码副本与最多2小时预算。

## 用户修订：官方10000迭代与吞吐测试

用户明确要求官方超参、计时并提高利用率。旧192环境/1000迭代run通过STOP停止，
原BC、首迭代checkpoint和全部日志保留；不将提前停止误称完成。
保持官方更新数学、8步rollout、5epoch/4minibatch/Adam/adaptive KL；正式目标10000迭代。
先在相同96 TRAIN物体与BC上测六卡每卡160/192环境，各12iteration，900秒上限，
记录上游采集/学习wall time、总交互吞吐、GPU利用率/显存；前2轮作预热。
总960/1152环境接近官方配置1000，六卡均匀分配避免修改梯度权重。
根据实测选择规模并估算10000迭代耗时，之后设有余量且不超过48小时的有界预算。
无需新增缓存或下载；停止线仍为磁盘20GiB和任一rank失败。

## 输入确认后继续正式预算（2026-10-18）

用户确认保留565维：native300（含最终位置误差）+tau264+history-valid1。
160环境/卡计时完成，192环境/卡计时因用户暂停而中止，不用于完整比较。
选择GPU0–5各160环境，共960环境、96 TRAIN物体，10000迭代（76800000环境步），
每卡20 Adam steps/iteration；48小时硬预算，预计约26小时，每200轮及末端保存。
run_id: official-scheduled-mixed-20261018-r2；从同一完整scheduled BC重新初始化，
不续接短计时试跑权重。输出outputs/wam-rl-policy-v2/official-scheduled-mixed-20261018-r2。
环境graspenv Python3.8/Torch2.4.1+cu121，固定运行源码副本和commit。
hand-only优化真实GPU等价性检查失败，max差0.00050056m，原因待查；
正式运行已恢复完整runtime.predict，不使用该优化，不放宽检查阈值。

## 预留卡性能诊断（用户授权）

用户明确授权使用剩余GPU测试提高利用率。主run固定源码不变。
纯工程计时：GPU6单rank160环境/16物体/12iteration，900秒上限，
REF2DEX_PROFILE_PPO=1输出cProfile分解；后续最多GPU6/7对照，总诊断预算30分钟。
目标区分CPU组装/scheduler、WAM计算、同步等待，优化决策以端到端吞吐及语义一致性为准。
所有输出独立v2目录，不覆盖主训练或BC，测试完释放预留卡。

### 性能诊断完成：Scheduler重复SE(3)计算

profile基线bff64750，GPU6，12轮；优化9b877365，GPU7，同配置。
总剖析124秒中scheduler.update累计63.54秒、其中_bounded37.57秒；
完整WAM33.50秒。优化将28次二分中不变的rotvec、距离、实际表面点移到循环外，
保留28次二分与相同限幅，先检查线性平移/角度预算，再计算表面点误差。
180组随机纯平移/纯旋转/混合情况对原实现误差<=1e-8，25项测试通过。
500固定输入原1.001秒→0.360秒；profile中_bounded37.57→13.40秒。

关闭profiler的最终native A/B：代码d29d9d6d，GPU6 baseline绑定冻结旧_bounded，
GPU7优化；相同16 TRAIN物体、160环境、seed0、16轮，各320 Adam steps，
每run600秒上限。两组均COMPLETED，动作路径误差0，跨96个episode。
去掉前2轮，平均每轮8.809→7.195秒（-18.3%），145.3→177.9环境步/秒（+22.4%）。
其中采集8.631→7.020秒，更新0.178→0.176秒。共同运行窗口1Hz利用率采样
GPU6均值39.6%、GPU7均值42.7%；包含monica的共享任务，仅描述性参考。
输出：outputs/wam-rl-policy-v2/scheduler-ab-{baseline,optimized}-20261018-r1，
汇总位于optimized的comparison.json；可复现入口tools/audit/benchmark_scheduler_worker.py。
首个wrapper尝试因native_helpers搜索路径未设置而在仿真前退出，已修正并重跑。

限制：单次短测、不同共享GPU，尚无六卡整体提速测量；非成功率实验。
六卡主训练r2仍用其固定40b35d51源码，未热替换、未重启、未损失进度；
此优化已提交供后续有checkpoint的续训使用。GPU6/7诊断进程全部退出并释放显存。

## 优化版续训r3与越界故障修复

用户要求从当前checkpoint切换优化版。检查发现r2已于第101轮后rank4异常退出：
valid point outside fixed PTv3 workspace，grid_min=[-6,171,169]。
完整checkpoint仅iteration1；第2–101轮未保存，无法恢复，不宣称无损切换。
模型/optimizer/step_size加载r2/rank-0/official-iteration-1.pt，重新初始化native环境，
新run official-scheduled-mixed-20261018-r3，追加9999轮，总目标10000。
继续GPU0–5各160环境/96对象/相同seed和资产；GPU6/7不分配。48小时预算。
采用已测试scheduler优化；每20轮保存，约500份×56MiB，约28GiB，有20GiB停止线。

越界保护：用相同float32原点/voxel/grid[0,65535]对WAM当前有效点预检；
只将越界环境标记done并置native reset_buf，终态tau=0/valid=0不用于下一次动作，
其余环境正常推理。未放宽WAM边界或裁剪输入，workspace_failures逐rank记录。
新增显式失败终止条件；reward公式/PPO更新不变。这是相对原环境的终止规则差异，
后续成功率比较必须保持该规则一致，并报告越界次数。

## 用户授权GPU6/7批量接口实验

主r3固定源码继续运行，不切换。工程实验比较REF2DEX_BATCH_PIPELINE五种模式：
legacy、tau（GPU批量插值/旋转并保留565观测在设备上）、inputs（加GPU向量化WAM输入及越界检查）、
scheduler（独立状态机+批量28次connector求解）、all。默认legacy。
每种单卡160环境/同16对象/seed0/12轮，900秒上限；最多两卡同时，整个诊断30分钟。
所有run保存源码副本，模式写入curve。WAM完整forward不变，不启用hand-only。
GPU6已通过输入字段、tau与CPU基线比较；部分reset、越界mask和完整scheduler状态对照通过。
WAM输入浮点容差2e-6，tau1e-7，scheduler1e-8；尚待端到端结果。

GPU真实归档3组×32样本检查：输入字段最大差1.82e-12，valid/mask相同。
record0原始完整forward两次最大tau差0.00062823m，candidate相对reference差相同；
record100/1000重复与candidate均0。不能宣称WAM推理逐位确定；优化未扩大本次测得波动。
见batch-pipeline-all-20261018-r1/archived-input-audit.json。
全部模式单卡检查后，再在GPU6/7跑all模式双rank160环境×12轮，900秒上限，
检查实际梯度/参数同步与设备观测路径，不切换六卡主训练。

### 批量接口计时结果（GPU6/7，完成）

五组batch-pipeline-{legacy,tau,inputs,scheduler,all}-20261018-r1均完成12轮，
每组240 Adam steps，跨77 episodes；去掉前2轮后的平均采集+学习：

| 模式 | 采集秒 | 更新秒 | 每轮秒 | 环境步/秒 |
|---|---:|---:|---:|---:|
| legacy | 7.174 | 0.179 | 7.353 | 174.1 |
| tau | 7.096 | 0.206 | 7.302 | 175.3 |
| inputs（含tau） | 5.952 | 0.183 | 6.136 | 208.6 |
| scheduler | 6.047 | 0.174 | 6.221 | 205.7 |
| all | 5.231 | 0.182 | 5.413 | 236.5 |

合并版相对当前legacy耗时-26.4%、吞吐+35.8%；单独tau收益很小，不能声称显著提速。
主要收益来自输入构造和connector批量求解，Scheduler其他状态机仍逐环境独立执行。
未改变WAM完整forward、预测内容合同、PPO更新和reward；默认pipeline仍legacy，
主六卡r3保持冻结84b59aa2，未切换。31项CPU测试通过，另GPU6字段/时间/坐标检查通过。

双卡all检查batch-pipeline-pair-20261018-r2（GPU6/7，每卡160环境，共32物体）完成，
每rank240 Adam steps，首末同步误差0、动作路径误差0。平均每轮5.399秒；
44个1Hz稳定阶段样本GPU6平均65.4%、GPU7平均70.3%，包含已有共享任务。
首次双卡r1因默认端口与主run冲突在初始化阶段失败，未开始更新；改独立29736后成功。
launcher现默认OS选择空闲端口，并在任何worker启动前拒绝占用端口；已实测拒绝且未创建输出。
主run第121轮同步检查通过，之后继续运行，未出现此端口错误。

限制：短时共享机器计时，不是成功率比较；五个单卡模式分阶段运行，存在负载波动，
双卡结果不能直接当作六卡收益。真正归档WAM存在上述重复推理数值波动，
不声称预测逐位确定。输出comparison.json位于all单卡run；双卡summary.json位于pair-r2。
GPU6/7所有本轮worker结束，均回到约347MiB已有其他用户占用。

## 进一步CPU/同步诊断（用户授权GPU6/7）

每卡160环境、官方PPO配置不变，主r3不切换。先all模式12轮CPU profile（GPU6），
及6轮CUDA timeline（GPU7，仅第4轮采样，限制trace范围）。每run900秒上限，
本轮诊断30分钟预算；之后对照去除重复CPU几何及合并同步检查，保证有效性检查仍执行。

进一步profile：all版77.9秒中update_schedulers31.54秒，重复_pose11.18秒，
完整WAM33.48秒；CUDA单轮1875次cudaStreamSynchronize、CPU等待2.11秒。
这不是全部可消除的空转，包含真实GPU执行。
REF2DEX_DEEP_OPT=cpu：批量验证SE3后复用已验证位姿、批量交互几何；
sync：process-local clean hooks保留所有masked finite谓词，在完整predict返回前
合并检查一次，context外仍原同步检查；把native pose/hand/goal打包成单次D2H，
不传重复未使用的current300。all同时开启；none默认。共享WAM源文件不修改。
各模式16轮/160环境，GPU6/7，900秒上限；进一步GPU预测/异常mask检查后计时。

### CPU/同步进一步优化结果（完成）

4个deep-opt-{none,cpu,sync,all}-20261018-r1均完成16轮、每rank320次Adam step；
同160环境/16 TRAIN物体，剔除前2轮，平均每轮耗时分别5.204、4.872、5.084、4.633秒。
合并相比上一版batch_pipeline=all耗时-10.97%、吞吐+12.32%；同步单项仅约2.3%，
短时共享机器条件下不宣称其显著性。CPU部分是本轮主要收益来源。
35项测试通过，含COUPLED→LOST、部分reset、NaN有效性/掩码和context退出恢复。
GPU6真实归档3组×32样本检查通过，输入字段最大1.82e-12；tau差未超过该样本
原WAM重复推理测得差异（record0为0.628mm，另两组0），每组有效NaN目标均被拒绝。

CUDA第4轮trace对照：cudaStreamSynchronize 1875→1770次，CPU同步等待2.114→2.008秒，
cudaMemcpyAsync 2306→2201次；并非全部同步已消除，也不能把等待时间全当GPU空转。
trace位于batch-deeper-cuda-20261018-r{1,2}/rank-0/cuda-trace.json.gz，约9MiB/份。

最终双卡deep-opt-pair-20261018-r1完成12轮、每rank240 Adam step，首末参数同步0、
动作均值路径误差0；平均每轮4.730秒（上轮双卡5.399秒，非同时对照）。
38个1Hz稳定窗口样本GPU6均值69.3%、GPU7均值76.0%，仍包含monica共享任务。
GPU6/7本轮worker全部退出，回到已有347MiB占用。
主r3保持84b59aa2冻结代码、仍运行（检查至第335轮），无自动切换。
结果：建议后续有checkpoint时评估启用两个all选项；剩余同步及CPU状态机仍存在，
不为追求利用率继续改rollout批量、WAM频率或检查语义。六卡收益仍需部署后测量。
