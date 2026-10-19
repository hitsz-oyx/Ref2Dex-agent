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
status: RUNNING
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
83项Task测试通过（准备阶段）。未有新版BC/PPO成功率结论。
