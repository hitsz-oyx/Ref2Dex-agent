---
schema: ref2dex.probe.v2
probe_id: P-20261014-rl660-geometry-lift-joint
experiment_id: P-20261014-rl660-geometry-lift-joint
date: 2026-10-14
task: ptv3-shared-wam
branch: main
git_commit: e65f9bd8ef5d538bd0af6999acfc9a33a0795a53
claim_id: C1
hypothesis_family: HF-rl660-geometry-lift-joint
probe_index_in_family: 1
seed_pool: probe
seeds: [124, 125, 126, 147]
decision_changed_if_positive: qualify branch-preserving target R with matched native execution after joint training
decision_changed_if_negative: distinguish optimization and predicted-branch errors before adding inputs or expanding R
status: UNCLEAR
run_id: rl660-joint-geometry-lift-20261014-r1
---

# RL660带连续分支的几何目标：原大规模配置联合微调

Result: UNCLEAR；全量分支回放、global512双卡和K4工程门通过，新19k联合run已运行到首段20步，loss2.38有限。约1.43秒/步，尚无收敛或抓取结论。
Decision: 按用户授权在GPU5/7沿用原大规模超参运行19k新更新，监视实际PD/分支及forecast指标；保留旧latest/best，不以首段loss形成效果结论。

## Decision Note

服务于MISSION自训练执行底座。2k步三个R仍在下降，不能以短预算关闭物理target路线。
用户授权修复几何分支并按原大规模配置训练；本轮不是matched native/PD长训比较，不能
把联合变化解释为label单因素因果收益。旧raw-native step14000/latest与step4500/best
保留，不恢复其旧R或optimizer，不删除原结果；远端v2持续运行。

核心缺陷：SO(3)终点不包含连续Euler分支/绕圈，已有nearest-live解码不能还原teacher。
最小修复保留原12维几何目标，加3维相对同一query q的未折叠腕部角差作为分支参考。
查询冻结绝对姿态与连续角参考，消费时将姿态提升到参考附近的Euler分支再用live q
逆控制器；精确标签保持原servo方向，含±π饱和和gimbal lock。新增输出共387个参数，
R其余128D/2层/4头、8queries不变。模型预测的姿态/参考不一致时先按姿态选参考附近
的分支，不能把输入标签无歧义等同预测正确；须看实际PD/clipping和后续GT执行。

## 固定训练配置与初始化

`configs/robot_660_geometry_lift.json`逐项继承`robot_660_large.json`全部原字段：
共享trunk512D×16、PTv3/goal/hand/scene头不变，GPU5/7、每卡batch256/global512，
19,000个全新更新（不是从14k精确resume）、80000秒上限，AMP/TF32/重计算/fused AdamW，
weight_decay.01、clip1，scene_lr1e-5、new_lr5e-5、interaction_lr2.5e-5、new_layer_lr1e-5，
原expanded_source_layers4参数分组不变。all8 standardized L1等权，加原hand/scene与GT
anchor，四个原权重均1；只新增必要15维target表示，不使用first-control加权。
TRAIN/dev仍589/71整序列、110777/11908准入窗，原sequence-uniform/half-startup采样。
seed124/125与原旧run相同；dev seed126固定512索引、每500更新评价，非独立holdout。
保留sample_step_offset1000。沿用loader每rank4 workers、prefetch2、sequence cache128、pin-memory。

从旧`rl660-joint-large-20261014-r1/step14000.pt`只继承完整forecaster；新15维R和optimizer
显式重置。TRAIN-only重新计算state/τ/15维action的per-horizon/coordinate统计；state/τ
方法不变，物理输出std floor1e-6，未读取dev统计。source/schema/单位和head形状显式绑定。

新run独立8GB产物预算（4个选择别名/最近恢复点与写入临时完整权重），保留best/latest/
forecast_best/control_best；原保存周期1000、评价500、milestone20000不变，任一选择改善
即保存并保护。按更新后的CAMPAIGN使用剩余20GiB保护，既有训练自己的输出/时间预算
不扩大；未改用户对其他数据任务的授权。保守远端最多4卡+本地5/7=6，不操作他人GPU进程。

## 工程门与停止条件

先真实full-batch双卡2步smoke（global512，最多600秒/2GB），再新15维K4 native4步
smoke（seed147、dev s10_airplane_pass_1，最多300秒/200MB）；工程失败先修实现，不将
其作为方法负证据。确认选择/保存、梯度、native buffer15维和query/live时钟独立核对。
通过后启动19k；监视GPU利用率、显存、速度和ETA。非有限、源码漂移、foreign PID、
30分钟无进度、8GB产物或20GiB剩余空间门、80000秒超时则停止本run并保留证据。
不自动换配置或覆盖旧权重；不因单seed loss下降声称R已解决、能抓取或Cm有效。

## 分支修复证据

`tools/audit/audit_geometry_lift.py`解析全部TRAIN/dev实际pd_targets，encode→query冻结→
live转换后用独立NumPy controller重建PD；纯标签代数无神经网络，CPU2线程。
报告`outputs/ptv3-shared-wam/rl660-geometry-lift-audit-20261014-r1/result.json`：
TRAIN旧转换1571个target误差>1e-4、最大6.283185rad；dev旧转换110个、最大4.021811rad。
新表示全部122685窗/8步无该类错误、无clipping；TRAIN最大3.34e-6、dev1.43e-6，均为
float32恢复残差，低于5e-6工程门。完整标签通过不证明模型预测的分支/接触/抓取正确。
回归测试覆盖真实s6_airplane_fly_1 query201/chunk7的±π方向反转、Euler奇异姿态和多圈。

## Limitations / future evidence

无matched长训control、无多seed或独立holdout，不能将旧与新run差异单独归因标签。
15维辅助参考改变了R输出与监督，不能称完整PointWAM复现。完整GT执行/teacher校准、
预测分支准确率与闭环保持门待训练后再决定；工程4步不提供抓取效果结论。


## full-batch工程门

实现83c06492；原75项Task测试通过，补充三项配置/恢复/空间保护测试也通过（18项
相关测试命令包含原15项，不重复计数）。`engineering-rl660-geometry-lift-ddp-20261014-r1`
在GPU5/7、global512完成2步、评价和完整checkpoint保存；约60.36秒（包含梯度审查），
训练更新约1.85秒/步，峰值allocated9703MiB。hand/scene均到PTv3/interaction/goal；
action到forecaster/R非零，GT anchor到forecaster=0且到R非零；scene不经hand head。
这只验证真实大batch接线、内存和数值，不形成学习效果结论。

## K4工程门通过，启动原规模联合训练

`engineering-rl660-geometry-lift-native-20261014-r1`使用15维full checkpoint、GPU7、
固定seed147/s10_airplane_pass_1完成4控制/K4，37.39秒；独立审计pass：native PD和clock
差0，几何buffer姿态最大差2.20e-7、query冻结目标到live command差1.52e-7。3角色的
4控制都不构成held45证据。本轮仅接线，没有宣称模型已学会正确分支或抓取。

用户授权的19k联合训练现在启动。实施code83c06492，具体启动/worker身份见新run的
manifest.json与tmp同run_id的launch.json；原双卡超参逐项相同，仅新增表示与运行保护字段。
CAMPAIGN已取消全球400GB上限，新run仍限制自己8GB并保留20GiB可用空间；远端监测器
已由对应任务更新为无全球总量停止线，本轮没有改变远端权重、进程或训练超参。


## 启动回执与首段观测

实际运行commit5a97b27b，run_id=`rl660-joint-geometry-lift-20261014-r1`，
supervisor82326、torchrun82434、workers82548/82549。参数101,636,757（比旧pair仅多387）。
manifest确认world_size2、parent step14000、R_reset/optimizer_reset=true；旧权重未覆盖。
首段step20：loss2.37985（hand.56586、scene.16587、action.82406、anchor.82406），
global512，1.430秒/步、358.0 samples/s，data_wait.00555秒；峰值allocated9974MiB，
physical GPU5/7显存11141/11197MiB，后续即时利用率均100%。初估余下约7.5–8小时，
不是完成承诺；含验证/保存和共享机器负载变化。监控记录startup_observation.json、
process.json、progress.json和gpu.jsonl。首个固定dev在500步，当前尚无dev趋势或新best；
总loss与旧12维native目标不可直接作优劣比较。阶段结论仍UNCLEAR。

## 完成回执（2026-10-15 03:00 CST）

`process.json` 最终为 `COMPLETED`、`exit_code=0`，19,000/19,000 updates 完成；输出目录
`outputs/ptv3-shared-wam/rl660-joint-geometry-lift-20261014-r1/` 的 `step7000.pt`、
`step16000.pt`、`step19000.pt` 和四个别名均可读，未覆盖旧 run。`engineering_audit.json` 的
梯度/有限性门通过，训练 JSON 与曲线没有非有限数值；GPU5/7 在结束后已释放。`best.pt` 指向
step16000，按预定义指标 `forecast_best.pt` 指向 step7000、`control_best.pt` 指向 step19000；
这三个选择反映不同工程指标，不能合并成单一效果结论。

最终固定 dev（step19000）工程读数：`hand_all24=109.88mm`、`scene_all24=42.18mm`，
`pred/PD_xyz=56.14mm`（p95 187.80mm），clipping 为 0；这些是单 seed/同一 dev 面板的
Probe 指标，不是抓取成功、闭环稳定或跨源泛化证据。run 使用既有 RL660 corpus 和旧 v2
stats，与当前 native3D 多源 admission 无关；后续不自动据此启动混合训练。实验状态保持
`UNCLEAR`，完整 GT 执行、matched control、多 seed 和独立 holdout 仍是 future evidence。
