---
schema: ref2dex.probe.v2
probe_id: P-20261022-teacher-action-reset-validation
experiment_id: P-20261022-teacher-action-reset-validation
date: 2026-10-22
task: ptv3-shared-wam-v6
branch: main
git_commit: 60efadce2444709bbbf9802107de7bc5e3f9eef6
claim_id: C3
hypothesis_family: HF-native-teacher-action-reset-acceptance
probe_index_in_family: 1
seed_pool: probe
seeds: [195]
classification: Blocker
status: COMPLETED
run_id: teacher-action-reset-validation-20261022-r4
decision_changed_if_positive: accept measured native action and indexed reset before interpreting teacher and source-PD task ceilings
decision_changed_if_negative: stop task attribution and locate the concrete failed action or reset invariant
---

# 原生动作与indexed reset验收；teacher/source-PD上限

机主新增授权GPU4/5/6的小规模teacher/action/reset验证；本run仅分配GPU4，不启动RL或大训练。
每次启动重新nvidia-smi。任务语义由另一实现分支修复，动作/reset先独立执行；不擅改其
measured_official_env、task_scheduler或task_admission。

## 执行前固定预期与预算

工程阶段2环境，使用P0既有panel前两条作为固定native初始化（不按结果挑物体）。
第2步只reset env0、第4步只reset env1；reset调用不计物理step、不偷走一步仿真。
要求untouched环境q/dq/object/hand/physical/tick/table/force-valid/episode逐数组差0；
目标env episode加1、tick0、force invalid，下一控制tick1且force valid。

对12个active原生轴同时测试env0正向/env1负向，共24个有符号case。每轴重新reset，
20步预置指关节到合法中点，12步跟踪固定目标；腕平移±3mm、腕旋转±.03rad、手指±.08rad。
其他独立轴保持测得基准目标，PD被动轴只由固定coupling映射一次。
要求24/24实际返回PD和独立NumPy映射最大误差≤1e−6；实测q/dq必须finite，目标轴
12步后的有符号响应≥预置幅度25%、目标误差≤幅度75%。PD正确不能代替运动响应通过。
若轴目标因原生边界不可实现、动作发生clipping或响应不合格，停止并保存具体轴/实测q/dq，
不降低门槛追求24/24。

工程物理预算：4reset-fixture controls +12×(20+12)=388个batched controls，2env共776
envcontrols；wall≤600s。不加载teacher网络或WAM，不优化任何权重。
保存response trace、reset前后完整快照、实际PD以及有判别力的错误映射mutants；
只在相同实测输入上构造mutant，不把合成算式当新的物理控制实验。

后续teacher/source-PD上限总计（含工程）≤10000envcontrols，不增加训练：使用既有panel
固定前6条，每臂最多600controls（最多7200envcontrols），完整task语义按修复后的同一
TaskController合同评估，不将截断prefix重新标为HOLD。实际source-PD先执行并保留失败。
目标为注册完整任务成功≥95%，同时分别报持握/抬升代理和删失；6条需6/6才达到95%。
未完整观察到终点/settle的回合标右删失，不能当成功或静默降成功阈值。若参考本身缺真实
计划意图/终点定义，完整任务正控制资格为待定，仍需报source-PD实际行为。

停止条件：任何动作/PD、episode/tick、reset隔离、数值非有限、预算、GPU争用或磁盘20GiB
边界失败立即保存停止；仅管理本run自己启动的child，不碰其他进程。结果不进入v6预测
质量、sample-efficiency或主路线结论，除非对应执行/任务正控制确实合格。

## 2026-10-22 工程阶段结果（teacher/source-PD待冻结合同）

r3完成388batched/776envcontrols，24/24正负轴响应通过；实际PD对独立NumPy合同maxabs0，
响应幅值98.1494%–99.2358%，最大终点误差1.8506%。第2/4步indexed reset对untouched环境
13字段逐数组差0；目标episode/tick/forcevalid通过。r1/r2工具中止保留，累计含失败786
envcontrols；未修改native reset，未降低门槛。warmup明确为每tick合法zero wrist delta+
finger midpoint；初态q不能普遍反解成合法自保持目标，详见审计。

证据：`outputs/ptv3-shared-wam-v6/teacher-action-reset-validation-20261022-r3/`；
报告`src/task/ptv3-shared-wam-v6/docs/audit/teacher-action-reset-validation-20261022.md`。
工程结果不能证明teacher≥95%或完整任务可达，后两项仍未测。

## 追加有界continuous腕角代表对照（执行前注册）

代码身份为上述base+dirty工作树，按运行保存source snapshot；不把base当运行源码无修改。
首批fresh teacher/source-PD已完成各3600envcontrols；两个冻结评分器均fulltask0/6，
lift+forcehold30均6/6。原source-PD cubemedium在真实参考期10controls clipping，
首tick实测palm真旋转2.484rad。URDF wrist joint4/5/6为continuous，独立FK验证
reset腕角modulo2π前后所有link变换maxabs3.3e-16，允许单例追加600controls：
仅审计host normalize reset连续腕角，PD选择最近等价角，保存原rawtarget及修正target。
预期首tick真旋转<.1rad，source参考期clipping10→0，raw/等价target的独立FK差<1e-6；
完整任务仍用已冻结v2门槛，绝不以角代表修复替代任务通过。先核native实际hasLimits，
不改生产共享host/native源码，累计最多8586envcontrols。若不符合，报告具体失败。

最后授权4envcontrols的隔离：normalized-reset cubemedium单env，4次canonical零（腕无增量、
手指同r3合法中点）。预期首tick实测palm真旋转<.1rad；与r3 raw reset下完全同canonical0
输入对照，明确实际基线从r3 trace离线FK核算，不能把source-PD强动作下2.484rad当neutral
基线。单env相对原2env fixture有batch差异，初始物理位姿等价要单列核对。后续不增物理。

最终生产回归前注册：root冻结生产host reset continuous腕正规化；只修改初态，不重写
rawsource PD target。回归沿原r3前2固定序列、同seed195、388batched/776envcontrols，
24方向门槛不变。新增实测palm_pose，使untouched隔离为14字段精确0；首tick cubemedium
实测body quaternion旋转<.1rad。harness将生产入口/normalize/reset/FK源码快照保存到run，
最多累计9366envcontrols，之后不再运行新物理。

## 最终结果

生产host r4完成388batched/776envcontrols：24/24轴响应、PDmaxabs0，两个indexed reset
14字段（含palm_pose）untouched全0；cubemedium首tick实测rigidbody quaternion旋转
.000435070rad<.1通过，airplane .002433885rad。17项CPU生产helper回归通过。所有自有
进程已退出、GPU4核查6MiB/0%，总9366envcontrols，本轮结束。

实验执行COMPLETED，完整任务资格FAIL/未资格分开记录：旧初始化或审计branch的
原source-PD0/6、branch0/1；fresh teacher0/6但5条缺终点settle记录删失；两臂真实
elapsed≥1s持握代理均6/6。生产reset修复后只跑工程回归，未重跑完整正控制，不能将
旧0/6套作修复后完整链，也不能声明teacher≥95%或WAM解码是主瓶颈。
