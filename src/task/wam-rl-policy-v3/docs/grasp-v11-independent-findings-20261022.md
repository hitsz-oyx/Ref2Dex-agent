# v11：既有死因、目标合同与待裁决项

## 结果解读表

| 已核结果 / 后续结果 | 结论 | 下一步决定 |
|---|---|---|
| v9 seen：15成功、16持握未到位；25/40目标误差>15mm | 旧物体终点判据已有成功路径；剩余大误差不能用腕/掌角色问题解释 | 收完v10固定15mm验收，再讨论版本化的新几何契约 |
| v9 unseen：40限位，36/40末次全部18个PD目标合法 | 名义目标预检不足以保证动态受压状态安全 | 保留实测安全中止；候选筛选同时处理接触几何/受压卸载 |
| 新SE3草稿：实测物体滑移20mm而腕X修正0 | 首次抓持变换的固定腕目标不能自动补偿物体滑移 | 修live物体误差反馈，并用此反例作回归；未验收草稿不进入v10 |
| 覆盖草稿每臂12，独立子集8/4 | 不符合仍有效的每臂每子集n≥40 | 正式覆盖扩大规模；工程smoke不能承担泛化/显著性结论 |

本报告只记录本执行流独立读取代码/已有产物的发现。v10现已完成160/160回合，seen35/40对0/40、原未见3/40对0/40，
详见[完成交付](grasp-handoff-results-20261022.md)。新尺寸化/SE3控制没有混入v10。
后续[每链n40接口未见覆盖](experiments/probes/P-20261022-grasp-coverage40.md)另行冻结、排队；
不对未满40的中间数据宣称显著性，不把12例工程覆盖与其合并。

## 互斥死因与重叠指标

`outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/v9-death-geometry.json`包含每例实测对象/目标SE3、掌相对物体SE3、真实掌面与腕原点间距、5指尖间距、力、q/dq、PD、末态桌面间距和轨迹路径。

| v9 A1互斥类别 | seen n40 | unseen n40 |
|---|---:|---:|
| 成功 | 15 | 0 |
| 持握但未到位 | 16 | 0 |
| 关节限位中止 | 5 | 40 |
| 接触但未抬升 | 2 | 0 |
| 抬升但连续持握不足 | 1 | 0 |
| 曾持握后丢失 | 1 | 0 |
| 合计 | 40 | 40 |

“目标误差>15mm”是重叠指标：seen25/40、unseen40/40，不能再作为第三组死因相加。seen全体末次误差中位79.5796mm、p90=579.0795mm；共同20s生存者35例中位16.8498mm、p90=565.8278mm。unseen全部提前中止，没有共同20s误差分布。

117.566mm指原先把`hand_base_link`腕原点当掌面的错误。`stage_evaluation.py:35`已用实际URDF掌面网格适配S2c；`run_large_grasp_worker.py`的`goal_error`是**物体中心到任务终点**，二者不是同一几何量。不能把修掌面角色再宣称为修掉这25次目标未达。

限位方向：小指近节6次（5下界、1上界），拇指近节俯仰11次上界、拇指中节25次下界；42个事件/40回合。31回合approach、9回合close，39/40不是腕pitch近奇异构型。`v9-pd-target-admission.json`独立核到36/40全部18个末次PD目标在名义范围内；40/40实际越界的指关节自身目标都合法。这支持“预检必要但不能包办动态跟踪”，不支持“更多数据/学习无法解决”。

复现（CPU，无新仿真）：
```sh
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/audit/classify_measured_grasp_deaths.py --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g4/manifest.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g5/manifest.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g6/manifest.json --output outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/v9-death-geometry-repro.json
```
期望两子集各40、互斥合计40、held-off-goal16、unseen joint40；所有几何证据逐例可查。

## 主张来源与本线程判断

| 主张 | 来源 / 当前分歧 | 本线程独立判断与证据 |
|---|---|---|
| v6在线物体坐标有问题 | Hermes已撤回，GPT反驳 | 反驳成立。`online_runtime.py:35`以当前物体pose的逆作anchor，`:42`将goal转相对坐标；不是本轮待修坐标bug |
| goal_valid可能训练/在线不同 | Hermes已撤回，GPT反驳 | 反驳成立。在线`:49`全True；V6Windows继承V5，`ptv3_shared_wam_v5/data.py:61`同样全True，v6/data.py只新增事件dropout；`model.py:134`按mask处理 |
| goal仍需语义/时间契约 | Hermes/GPT一致 | 成立：训练`future_flow[-1]`为实际t+24终点，在线传`decision.local_goal_pose`是调度器意图。应记录goal_kind、deadline/horizon=24/30s、frame=current_object；不能用未来真值在线补条件 |
| 所有阈值尺寸化、补姿态项是统一解法 | Hermes初始建议；GPT反对一刀切 | 不把它当统一答案。新物体尺寸归一化应版本化，并与旧15mm主指标并报；位置任务/对称物体不能擅自加任意yaw成功条件。已有成功证明旧窄任务并非全不可达 |
| 先扩大初态覆盖、尽早有预算RL | GPT主张，Hermes接受覆盖及条件启动 | 支持覆盖与有条件启动；覆盖不能n12冒充n40，且须标明是否同时换了控制/判据。不能因seen<80单独否决PPO |
| PPO准入要seen≥80 | Hermes v11初稿，后续已降低 | 按最新三条件：干净消融、阶段reward就位、预注册步数/时间/对照。三臂原始样本效率主张与工程残差probe须分清；本轮“方向说明不要求实现”不被解释成大训练授权 |
| BC改为漂移状态意图纠偏 | Hermes，方向保留 | 支持；监督必须是实际纠偏控制器的意图，不能把参考帧开环命令重命名为DAgger；尚未物理资格的几何草稿不能因测试PASS被当作专家上界 |

## 草稿SE3待裁决：物体滑动没有反馈

只读检查当前并发`grasp_geometry.py::RelativePoseServo`：先令object=goal=I、palm平移X=0.1m；第一次调用捕获抓持相对变换。第二次仅让实测object向X滑动0.02m，palm不动。实测物体目标误差X=-20mm，输出腕X修正=0。原因是`desired=effective @ self.grasp_relative`跟踪首次抓持变换构造的固定腕目标。

后续复核：并发提交`e8f6ce65`已加入live物体相对误差；同一20mm反例输出X=-14.2mm，接口反例已消失。此项从“待修”更新为“代码反例修复，物理验收待定”，不继续指称当前代码仍输出0。

修复期望：位置任务下此反例应给负X修正；全姿态任务还须包含当前物体误差引起的腕部旋转/杠杆平移，并守住范数限速与最终饱和后的抗积分回授。一个可核的几何关系是`T_WPalm_des = T_WGoal @ inv(T_WObj_live) @ T_WPalm_live`，position-only则只用物体平移误差、不擅自增加姿态成功门槛。实现选择由几何/SE3维护会话确认；此处不直接改并发文件。

## 并发与版本边界

检测到独立执行流维护`grasp_geometry.py`、几何tests、`continue_grasp_batch.py`与v6 `staged_progress.py`。本执行流的目录级add误将其中最初版本纳入`c93164b5`；该提交不是这些草稿全部物理验收的声明。后续仅逐文件暂存，保留并发工作。分工已向机主异步确认。

并发`g4/5/6-resume2`正在续v10，已核consumer/pose/stage与原冻结快照字节相同；不重复启动。其技术预算按3×600=1800保守预留，native退出收据实际记录161+510+210=881额外控制；最终同时报告保守预算和可核实实际数，不混进160个完成回合的分母。

分类器防错：把真实`held_but_off_goal`的`final_success`故意翻为True，完整CLI以`published outcome disagrees with saved physical endpoint`拒绝；正常80条A1记录通过。原正式数据未改。故障副本与日志在`tmp/v11-classification-mutant/`，收据`.../grasp-v10-stat40-20261022-r1/classification-mutant.json`。分类器从消费日志读取真实postgrasp目标，不重新假设初态+8cm；仅适用于v9/v10冻结15mm协议。

后续打包阻塞已修：geometry worker直接导入`source_pd`，但它不在legacy基线快照，也未在新复制清单。实际隔离旧包导入报ModuleNotFoundError。提取`freeze_grasp_geometry.py::copy_geometry_runtime`供真实worker调用，补该依赖；隔离包回归先1 failed（真实缺项），修后通过，再故意删该模块仍被拒绝。只补打包，不改几何/伺服常数。运行中v10不依赖这次改动。

## v11 三处实现的可复核位置与数值期望

| 修复 | 当前文件:行号（wam-rl-policy-v3内） | 可核的数值期望；物理效果另验 |
|---|---|---|
| 尺寸几何/到位合同 | `src/wam_rl_policy_v3/grasp_geometry.py:20`、`pose_consumer.py:221` | 同物体尺度×2，容差×2；圆柱容差3.98077mm；真实源参考仍hold2.000s，不能因换测量角色改称伺服成功 |
| 实测物体相对SE3反馈/限速/抗饱和 | `src/wam_rl_policy_v3/grasp_geometry.py:137`、`:222` | 固定腕、物体X滑移20mm应给负X纠偏（反例实测−14.2mm）；每步平移范数≤25mm、旋转≤.08rad，最终饱和回授积分 |
| 几何候选+原生18目标准入 | `src/wam_rl_policy_v3/grasp_geometry.py:54`、`:96`、`:170` | 非法目标候选0被接纳；实测限位目标≤4/40，不能把静态合法等同动态安全，实际计数等物理结果 |
| 快照依赖修复 | `tools/audit/freeze_grasp_geometry.py:8`、`tests/test_geometry_runtime_snapshot.py:1` | 干净隔离包可导入source_pd；故意删除依赖必须拒绝，不fallback到工作区包 |

复现用上方[三处修复报告](grasp-geometry-coverage-20261022.md)的geometry测试/真实参考命令；
独立快照回归：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/wam-rl-policy-v3/tests/test_geometry_runtime_snapshot.py`，
期望1passed/0failed。软件检查通过不证明物理修复幅度，正式数值由120回合冻结覆盖给出。
原生目标准入、相位和真实关节中止分别统计，不用final_clipped_controls=0替代“无饱和”。
旧v10 seen A1实际R上游裁剪2/23562、最终归一化裁剪0、PI限幅235次；
原未见A1上游2/15597、最终0、PI限幅145次；两者不能报成统一的“动作饱和0”。

无法自证：canonical稀疏点几何与真实接触面的完全一致、两指接近与力是否来自相同接触体、
动态受压安全、三方独立物理复跑的一致性。A0 source-PD是不同执行链，成功也不能当GTτ→R上界，
更不能凭A0>A1直接决定重训WAM。
