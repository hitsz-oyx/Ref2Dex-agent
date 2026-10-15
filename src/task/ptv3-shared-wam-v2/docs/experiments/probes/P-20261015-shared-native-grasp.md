---
schema: ref2dex.probe.v2
probe_id: P-20261015-shared-native-grasp
experiment_id: P-20261015-shared-native-grasp
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: f6f1b080
claim_id: C1
hypothesis_family: HF-shared-native-grasp
probe_index_in_family: 1
seed_pool: probe
seeds: [148]
decision_changed_if_positive: prioritize shared actor native qualification with broader heldout sequences
decision_changed_if_negative: inspect fitting and early closed-loop action errors before additional scaling
status: UNPROMISING
run_id: shared-native-grasp-20261015-r1
---

# Inspire shared-only 原生抓取初筛

Class: Decision。用户要求检查当前shared版本是否能抓取，服务独立执行底座资格验证。
检验离线forecast/动作误差与实际闭环之间是否已出现正向信号；不把Shadow与Inspire
不同数据、状态和控制接口当作数据量因果对照。最低成本是既有三条teacher可抓起序列。

固定当前control_best(step7000)和forecast_best(step1000)，在运行前以硬链接保留到本轮
inputs目录，不追随训练过程中移动的alias。每个checkpoint分别测试
s10_airplane_pass_1(149 controls)、s4_spheresmall_pass_1(141)、s10_mug_pass_1(172)，
seed148，完全沿用旧阶段测试序列/时长。均为val序列时只称小面板探测，不代表总体成功率。
先4-step工程smoke，然后完成三条序列。主要报告每条teacher和主student的最大抬升、
连续held帧数（surface gap<=1cm、lift>=3cm、排除桌面支撑），沿用旧DExplore测量。
短序列按partial metric记录，不称完整hold/place成功或跨手匹配成功率。

执行K=1，每个30Hz控制步重新从本行实测H4/87D状态和已知参考物体goal[t+24]预测，
沿用训练已有goal合同；不额外输入未来实测hand/teacher动作/GT tau。无warmup或teacher
混合。另一个student row独立用自己的实测状态运行同一模型，作为工程重复，不把两行
当作独立seed。row0为teacher正控制。shared接口显式准入，legacy默认仍拒绝shared权重。

新桥接只改变动作头调用：一次forecaster(return_shared=True)的tokens+mask进入原训练
SharedQueryActor；q_PD geometry-lift冻结/当前状态解码和实际native PD转换沿用已审计链路。
每次完整输出用独立audit_native核对实际命令、PD、冻结目标、H4、状态、goal和held计数。
任何执行合同失败先修复，只重跑受影响部分，不把错误实现当方法负证据。

## Decision Note / 资源

用户2026-10-15新授权：停止Inspire shared_tau/GPU4,5，改为Shadow扩大训练；空卡用于评估。
本轮GPU6执行（GPU7可用于另一checkpoint以缩短等待），不影响GPU0..3剩余Inspire和
GPU4,5新Shadow训练。每个group<=600s/1GB，总两checkpoint<=1h/8GB，磁盘20GiB底线。
固定提交和checkpoint输入，外来GPU进程、超时、非有限、PD/接口不匹配只停止自己的组。
teacher不能复现该序列的正控制则不把该序列的student失败解释为控制方法负证据。

## Limitations / future evidence

两种checkpoint按不同离线标准选择，完整报告两者，禁止只展示有利阶段。扩大序列面板、
独立reset、多seed和完整hold/place验证待正向信号后再做。当前仅Probe，结论PROMISING/
UNPROMISING/UNCLEAR，不与Shadow的60Hz/5cm目标保持指标直接比较。


## 已完成结果 / 2026-10-15

执行代码f6f1b080（固定工作树tmp/shared-native-code-20261015-r1），GPU6/7两个checkpoint
并行、各三条序列。driver2240975已COMPLETED；先4-step smoke和独立审计PASS，随后6个
完整短episode全部双peer exit0、独立审计PASS。检查同时覆盖两student本行H4/87D、
原goal、geometry-lift冻结目标、实际PD、命令时钟和独立held计数。

| sequence | teacher held(control / forecast组) | shared7k held | shared1k held |
| --- | ---: | ---: | ---: |
| s10_airplane_pass_1 | 62 / 62 | 0 | 0 |
| s4_spheresmall_pass_1 | 76 / 74 | 0 | 0 |
| s10_mug_pass_1 | 92 / 95 | 0 | 0 |

两种checkpoint、两student行在三条序列全部held0。主student最大抬升仅2.663/0/3.289mm，
未达到3cm阈值。7k主student动作裁剪帧比例38.26%/37.59%/1.16%；1k为75.17%/57.45%/61.63%。
杯子7k低裁剪也失败，不能将全部失败简单归因控制饱和。

选择身份：7k是当时位置P95最优（DEV XYZ mean62.70/p95171.07mm，action0.41059）；
1k是总loss和forecast物理误差和最优（mean79.96/p95215.14mm，action0.49407）。
Inspire best按total objective、control_best按XYZ p95再mean、forecast_best按hand+scene物理
ADE和；Shadow best按common DEV normalized action L1。不得说两边使用同一选点规则。
动作都是normalized L1，forecast都是同实现horizon-normalized MSE（scene motion加权），
但动作含义/normalizer、状态profile、优化曝光不同，不能比较绝对loss推断跨手能力。

本固定三条已知teacher正控制面板上未出现shared抓取信号，局部UNPROMISING；
不等于整个shared方法或数据量假设被REFUTED。Inspire A/B原12k继续，不追加本轮native
checkpoint sweep或修改控制补救。下一步由Shadow扩大训练和已有拟合证据决定是否扩大面板。

结果：outputs/ptv3-shared-wam-v2/shared-native-grasp-20261015-r1/panel-status.json；
紧凑指标compact-results.json；每组native/trajectory.npz、result.json、independent_audit.json
保留。输入是硬链接快照，后续训练移动alias或清理恢复点不影响本轮权重。
