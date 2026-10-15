---
schema: ref2dex.probe.v2
probe_id: P-20261015-v2-stage-grasp
experiment_id: P-20261015-v2-stage-grasp
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: b8542ef6
claim_id: C1
hypothesis_family: HF-v2-stage-grasp
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: qualify successful stages on a broader physical panel and compare action versus forecast stage selection
decision_changed_if_negative: separate R execution failure from predicted trajectory failure before further fine-tuning
status: UNPROMISING
run_id: v2-stage-grasp-20261015-r1
---

# WAM-v2 微调阶段的实际抓取 Probe

Result: COMPLETED；27组物理及最终独立审计全部完成。9阶段预测τ→R均held0；GTτ→R最好8k飞机27帧/杯子19帧，未达held45，局部UNPROMISING。
Decision: 用户要求直接拿保留的微调权重沿用旧Task物理评估；不再延长训练。

## 动机与协议

Class: Decision。服务于MISSION自主执行底座，区分离线动作loss降低能否改善抓取，以及
GTτ→R与预测τ→R在哪一层失效。最便宜方法是复用已有三路native执行，逐个完整权重
严格加载同阶段WAM与R（含normalizer），不跨阶段混配，不替换原80k parent。

固定完整阶段1000/3000/3500/4000/8000/9000/10000/11000/12000；权重只读位于
`outputs/ptv3-shared-wam-v2/robot-finetune-stage-archive-20261015-r1/`。
固定原dev面板s10_airplane_pass_1、s10_mug_pass_1、s4_spheresmall_pass_1，沿用旧
字典序选择，不按新结果筛样；控制长度分别149/172/141，留足GTτ的未来24帧。
每组3环境frame0重置，角色teacher、GT_tau_R、shared_WAM_R。K4刷新、8步chunk、
30Hz、原PD增益/coupling/clamp；query冻结目标并以live q逆转换。GTτ取原RL660实际
未来手轨迹（oracle诊断）；自主行只取自己的H4状态、原有参考物体t+24目标和预测τ。
本轮没有添加新的输入，也没有把参考未来手轨迹送给自主行。

v2 checkpoint保存原source而没有重复保存controller；从其精确source manifest读取
controller并在每组与真实native controller逐项比较。仅内存补齐执行元数据，不覆盖权重。
显式v2 schema/robot stage/完整WAM+R/goal/表示检查；v1默认行为保留。

每组记录原始手物轨迹、命令、实际PD、query/chunk时钟、WAM历史/目标和R输入输出。
复用独立NumPy/SciPy审计核对控制转换、连续角分支、输入时序和held统计。
主要描述性指标为连续held帧与held45：抬升>=3cm、手物表面距离<=1cm且脱离桌面支撑。
它是持物代理，不能当正式任务成功率；另报最大抬升、相对原记录手点误差、相对参考物体
位置误差和clipping。teacher未达到held45的序列标为校准不足，失败不可独立归罪模型。

若某阶段自主行在teacher合格序列达到held45且clipping<1%，记局部PROMISING并择优
做后续扩面；全部自主行未达门且teacher合格时仅记本面板UNPROMISING；teacher不合格
或关键工程问题使归因不清时UNCLEAR。不得从三个已用dev序列宣称泛化或Cm有效。

## 资源与停止条件

本机GPU0/1/2三卡并行，各卡一个序列逐阶段；远端80k和本地四卡微调已完成。
先4控制工程smoke<=180秒，再27组每组<=180秒、面板<=1800秒（最多90GPU-min），
输出<=2GB、文件系统余量>=20GiB；不复制checkpoint，不操作其他任务进程。
每个launcher拥有其policy/native两个进程组；超时/foreign GPU PID/非有限/控制不符
停止该组保留日志。独立失败不得悄悄换阈值/目标/数据；执行问题先定位后只补受影响组。
具体运行提交/进程/预算见process.json与各组group_status.json；初始入口code b8542ef6。

## Limitations / future evidence

单物理seed148沿用旧面板，非独立复制；dev曾用于离线模型选择。完整episode末尾24帧
被截去；held45不含最终放置/任务成功。原记录未来τ和当前执行状态可能分布偏离。
完整多序列、多seed、不同K及人工视觉确认待本Probe改变下一步决策后执行。

## 工程门与启动回执

27项checkpoint/source/表示/控制合同测试通过。4控制smoke为
`outputs/ptv3-shared-wam-v2/engineering-stage-native-20261015-r1`，27.76秒正常完成；
独立审计实际PD、query/chunk时钟、自身H4/state/goal误差均0，姿态与目标转换误差
分别1.46e-7/3.01e-7。另一次只读CPU复算GT-R/预测τ行R的首chunk误差
1.97e-6/2.03e-6。它们只验证接线，不证明WAM预测准确或抓取。
启动前旧存储检查触发历史400GB门，按已授权CAMPAIGN改为20GiB可用空间门；
未进入物理、未产生方法结果，权重与训练配置未改写。

正式panel代码5756c23b，supervisor1768030，GPU0/1/2，run_id如上；
`tmp/v2-native-20261015/launch.json`记录启动命令，正式process.json记录运行身份。
每组policy/native为launcher所属进程，组内跟踪GPU占用及foreign PID。
独立只读审查未见关键训练/在线语义错配；审计没有重新推理WAM，也不证明预测正确。

## 首批工程定位与有界恢复

r1在1k/3k飞机、杯子物理完成后暂停并结束派发，保留全部原轨迹。小球在physics前因
旧PhysicalGeometry读取的裁剪资产目录缺mesh失败，不能计作模型失败。改为读取本次
native实际加载的完整资产目录；飞机/杯子/手/table相关文件两处逐字节相同。

1k飞机原独立审计两个角命令差最大2.871e-6，定位为审计漏建模绝对rotvec的float32
缓冲量化。旋转量化仅1.20e-7rad，经近奇异Euler放大；分支选择无差异。补齐缓冲
与controller dtype后全149×2×12命令复算max0，原1e-6阈值不变。生产控制未修改，
原物理无需重跑，诊断脚本与JSON保存在r1/engineering-diagnosis。

修复代码226bb36c。r2恢复支持只复用checkpoint/sequence/seed/K/controls完全相同、
物理COMPLETED且独立审计PASS的旧组；失败和未运行组用新路径补跑。总科学问题、
面板、资源上限、阈值不变；r1运行不到3分钟，r2及r1合计仍限定90GPU-min预算。

小球4控制smoke29.80秒完成、独立审计PASS后启动r2；supervisor1777429、代码2ff0dfc6，
三卡与原面板不变，剩余wall上限1500秒，以使r1+r2不突破原合计预算。
1k/3k飞机、杯子的初始q/dq/hand/object/velocity在角色间及阶段间最大差均0；
同seed teacher后续保持帧存在小幅物理运行差异，故逐run保存teacher校准，不声称逐位
相同后续接触solver轨迹。r2从原4组已完成物理复用，其余执行于新路径。

## 最终完成与阶段比较

27个有效物理组均正常退出，三卡已释放。r2 supervisor原状态保留
COMPLETED_WITH_FAILURES（25个原审计pass、2个orientation数值审计失败），没有覆盖
历史记录；最终`r2/final-audit.json`和`r2/summary/manifest.json`确认全部27组PASS。
独立audit现在分别检查double Euler→量化buffer的数学姿态、float32 controller执行
姿态与rawcommand；三个1e-6门均保留，clamp/实际PD/时钟/held门不变。旧orientation
检查混合不同数值阶段：mug11k浮点往返1.0514e-6，但完整独立命令重建仍逐位0。
该数值差单列diagnostic。全27命令重建max0，姿态/实际PD/clock通过；人为扰动命令
的负对照仍拒绝。独立review确认没有以自比替代真实命令检查，没有放宽抓取门。

所有27组的角色间/跨阶段初始q、dq、hand、object、velocity最大差0。
teacher全部达到held45：飞机61–72帧、杯子90–97帧、小球58–77帧。
GTτ→R通过数0/27，预测τ→R通过数0/27；后者连瞬时held样本也均无。
以下顺序为飞机/杯子/小球，loss来自原固定dev256，保持帧来自本次物理。

| step | action loss | forecast loss | GT tau R held frames (air/mug/sphere) | WAM R held frames (air/mug/sphere) |
|---:|---:|---:|---|---|
| 1000 | 0.4897 | 4.2289 | 0/0/0 | 0/0/0 |
| 3000 | 0.4150 | 4.4449 | 0/0/0 | 0/0/0 |
| 3500 | 0.4070 | 4.4783 | 0/0/0 | 0/0/0 |
| 4000 | 0.4021 | 4.6118 | 0/0/0 | 0/0/0 |
| 8000 | 0.3763 | 4.7845 | 27/19/0 | 0/0/0 |
| 9000 | 0.3727 | 4.8456 | 8/14/0 | 0/0/0 |
| 10000 | 0.3703 | 4.8503 | 0/0/0 | 0/0/0 |
| 11000 | 0.3689 | 4.8483 | 0/0/0 | 0/0/0 |
| 12000 | 0.3677 | 4.9001 | 0/0/0 | 0/0/0 |

8k在GTτ条件下出现短暂持物；9k退化，10k及以后回到0。动作dev从1k的0.4897降至
12k的0.3677，没有对应持物改善；forecast合计则从4.2289升至4.9001。这个小面板
不能判断两种loss对最终抓取哪个更有帮助，也不能仅归因于WAM，因为GTτ→R同样未过门。
GTτ不是GT q_PD重放，本结果不推翻旧PD replay能持物的证据。

12k的自主行hand mean（相对原recorded hand）为飞机789.51mm/杯子504.78mm/
小球313.88mm；GTτ行234.92/968.14/1734.03mm，出现明显闭环偏离。它是整个执行
期间的均值，不等同离线第一步PD误差；不据此单独认定接触信息、表示或模型容量是原因。

Decision: 本次K4阶段筛选记UNPROMISING，保留全部权重/轨迹，不继续延长微调。
下一项最小判别可考虑8k/12k的K1对照或首接触前误差累积定位，区分chunk消耗与
R在偏离状态下的执行；不因本次Probe直接关闭q_PD或WAM路线，不启动新长训。

正式产物：`outputs/ptv3-shared-wam-v2/v2-stage-grasp-20261015-r2/summary/`中的
`outcomes.json/csv`（81行全部指标）、`stage-summary.json`、`stage-comparison.png`、
`table.md`、`initial-state-audit.json`和`manifest.json`。原轨迹在r1/r2对应子目录，
每条summary记录保留实际path。汇总工具`tools/audit/summarize_stage_grasp.py`可重导出。
