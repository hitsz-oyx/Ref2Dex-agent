---
schema: ref2dex.probe.v2
probe_id: P-20261019-inspire-task-contract
experiment_id: P-20261019-inspire-task-contract
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 12
seed_pool: probe
seeds: [246]
decision_changed_if_positive: train the full-data task-conditioned BC and official PPO under the repaired contract
decision_changed_if_negative: repair the first observation or native replay mismatch before further PPO scaling
status: COMPLETED
run_id: inspire-task7-fit-20261019-r1
---

用户ref1审计修订。Blocker：相同实测状态/终点而局部路径不同，在H4启动或τ被移除时，
旧577输入逐位相同；实际BatchedInspireInputs回归在旧代码失败。另复现静止接近物体而
无路径进度可每拍获得0.775奖励。优先补任务信息，保留全量数据和Shadow网络/PPO配方。

输入v2：physical312 + world tau264 + H4有效位1 + local task7 =584。
task7为当前物体坐标的局部goal平移3/rotvec3、confirmed_progress1，索引577:584；
tau仍312:576、valid576，已有物理与τ列身份不移动。task与WAM预测独立且从启动可见；
无WAM预测路径同样读取task7。current BC为physical312+task7，共319维；
scheduled初始化复制物理和task列、仅新增tau/valid列置零，保持当前BC函数。
task固定尺度平移.05m、旋转.35rad、progress1；物理归一化仍TRAIN-only/std下限.05。

完整660/50物体及589 TRAIN/71 DEV不变。只追加约4.64MiB task字段，
复用旧不可变物理/τ缓存与WAM80000身份，不重新推理神经缓存。
两阶段仍各50000更新、同1024/1024/512/512 ELU、同采样及DEV选择规则。
旧PPO已在972完整保存停止，不能把旧577/reward checkpoint直接当584普通resume。
新BC后PPO作为新合同运行，不能与旧972当成单变量连续学习曲线。

奖励v2：保留Shadow手掌/指尖接近项与高度项，将unfinished路径项改为局部目标点误差
-2*min(goal_point_m,.05)，去除持续的+0.9“贴路径”奖励；保留2*正confirmed进度增量。
完成bonus只在有序完成且当前实际SE3终态达标时支付，不依赖sticky complete单独判定。
为避免删除正偏置后主动越界提前结束，失败终止代价由非终态负项界限推导：
(.5*3+.5+2*.05)/(1-.96)=52.5，不改变PPO gamma=.96。
任务成功为有序完成后真实终态SE3持续1秒且不越界；高度10cm仅保留代理，
不作为放置终态要求，不宣称接触/释放证书。记录ever事件、当前终态、最终保持分别输出。

原canonical命令重放：当前纯物理host，不使用BC、WAM或PPO。先选cup/apple/cube三条TRAIN，
比较初态、第一拍PD、8拍手/物体偏差及全程首个分歧，不要求PhysX跨执行逐位相同。
初态/第一拍PD不对应则阻止长PPO并定位；短期对应但长时分歧需独立报告，
不能把全部闭环失败归因BC。首条cup初态q/hand/物体位置差0、第一拍PD差0，
8拍物体最大3.05µm、手平均最大3.72mm；全段物体差可达60mm，不称全程复现通过。

资源：模型graspenv/Torch2.4.1+cu121，物理dexplore_v120_train/Torch2.0.1+cu118。
准备与两阶段BC<=2小时、<=2GiB；CPU几何字段构造不是神经实验，不占GPU。
本地最多6卡、留出GPU6/7；重放只用GPU1–3，BC用GPU0。
PPO若工程检查通过，继续全局原48h截止与10000更新上限、滚动保存和20GiB磁盘保护。
不跑Shadow匹配KL续训，本次只补显式KL入口/身份；因果归因仍为未来Evidence。

产物：outputs/wam-rl-policy-v2/inspire-task7-bc-20261019-r1/cache/、
outputs/wam-rl-policy-v2/inspire-task7-fit-20261019-r1/、
outputs/wam-rl-policy-v2/inspire-task-audit-20261019-r1/。
实现提交45eb9837，BC冻结该提交；后续Shadow KL接线/重放审计提交53150deb。
85项Task测试通过。全量task字段准入检查通过：660序列、165677命令均有限，
50物体各前20拍的真实BatchedInspireInputs与离线task字段最大误差0；该检查不运行神经网络。

两阶段BC均已完成50000更新，单GPU0约309秒。current最佳DEV MAE0.03892193；
同一800行物体均衡DEV选择面板上，零扩展初始化与条件best均为0.03892196，
选择初始current扩展点，tau/valid权重为零，但新增task权重保留。
这没有优于旧312输入的离线MAE0.03850862，不能称局部任务输入已改善抓取或WAM已有收益。

原命令重放三条TRAIN的2环境结果：第一拍PD/初态q/hand/物体位置误差均0；
cup/apple/cube前8拍物体最大误差分别3.05/28/1.27µm，手点约3.72/2.72/4.51mm；
全程物体误差分别60.3mm/2.877m/83.4mm。因此只通过短程接口检查，长程动力学分歧仍记录。
进一步恢复cup原采集的16环境、批成员、seed120和fixed motion配置，重放所有16环境
共同可用的158拍：目标cup的q、手点、物体位置和PD目标全前缀误差均为0。
这是原canonical控制链可复现的证据，不证明不同采样/批配置的长期轨迹必然一致；
不能仅凭2环境漂移判定动作映射错误，也不能把16环境结果称为BC抓取成功。

原生工程检查inspire-task7-native-check-20261019-r1已完成：冻结53150deb，
64环境、50物体、4次官方PPO更新、80次optimizer step，2048交互；错开reset通过，
动作episode错配、PD目标误差、动作均值误差、logprob回放误差均0。
在线共享TF32对完整FP32最大tau差0.564mm，低于1mm门槛。
该检查只有3个终止（含2个强制reset），0任务成功，不能作为效果评估或长训起点。

新BC确定性首回合r2：50 TRAIN物体/64环境、300控制拍、0 optimizer step；
52个已结束回合中0任务成功、0最终高度保持、11个workspace失败，12个首回合尚未结束。
不能将其写为完整64回合成功率。r3将上限延长为1200控制拍，补齐完整回合；
使用GPU0/1、最多1200秒，不改变策略、输入或任务判定。这是是否继续扩大训练的Decision，
当前保持旧六卡停止，不凭工程通过重开万轮长训。
BC评估r1在启动时因未显式传--updates 0被拒绝，0物理步/0优化；正确重启为r2，
保留失败manifest，不将其计入评估。

r3完整评估已完成，484控制拍覆盖全部64首回合、50 TRAIN物体，0优化更新；
全部记录的初态字段与r2逐位一致。任务成功0/64、workspace失败17/64，
最终无越界高度保持1/64，不能把该高度代理写成完整路径任务成功。
结果与初态核对保存于`inspire-task7-bc-eval-20261019-r3/summary.json`。
决定：输入/在线/原生PPO工程检查通过，但新版BC仍非可靠控制底座，
保持旧六卡停止，暂不启动584合同的万轮训练；先定位BC闭环与任务条件的信息充分性。
task7解决明确的局部物体指令缺失，不保证相同物体任务下不同抓取/操作意图均可辨识。
未恢复手参考，未改变官方gamma/nsteps/noise数值，未进行Shadow KL匹配续训，
不将单seed TRAIN小面板、离线MAE或tau权重变化升级为WAM净收益结论。
