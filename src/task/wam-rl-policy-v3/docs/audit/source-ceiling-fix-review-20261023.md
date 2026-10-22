# source控制器接近/闭合修复（v14）

## 结果解读表

| 结果 | 结论 | 下一步决定 |
|---|---|---|
| 修后source≥32/40，物理安全通过 | 本开发面板控制器资格达标 | 才与WAM同初态/预算/后段比较 |
| 提升但<32/40 | 修复不够，不能拿WAM解释数据量 | 按余下死因继续修source |
| 不提升或实际限位仍多 | 当前方案未解决物理瓶颈 | 保留全部失败和gate，继续修；不训练v6 |

当前状态：**新物理批次尚未运行，无修后成功率**。GPU4/5/6被并行v13占用；软件检查不是物理达标。旧source=14/40（Wilson95 [0.2213453,0.5049412]），16745控制，0更新。

## 修复、可证伪预测与复现

可核对定位（均以`src/task/wam-rl-policy-v3/`为前缀）：
`src/wam_rl_policy_v3/source_ceiling.py:16`物体SE3映射、`:33`余量准入、`:64`实测卸载/限速/交接；
`tools/run/run_source_ceiling_worker.py:111`不实例化WAM、`:214`source映射实际调用、`:233`游标暂停；
`tools/run/run_source_ceiling40.py:12`原40初态/源码冻结；
`tools/run/summarize_source_ceiling.py:63`初态/逐物体核对入口。
首版diff为`git show 9f578b7d -- src/task/wam-rl-policy-v3`，交接/日志修订见同文件后续提交；
用`git diff 9d005b4e -- <上述单个文件>`可查看本轮最终diff，不含并行v13的文件。

- `src/wam_rl_policy_v3/source_ceiling.py`：接近/闭合实测指尖力>12N卸载；真实关节位置+速度向限位滑行触发独立退让（指尖力不是全手接触检测）。卸载暂停参考游标，不让参考时间继续跑；无效力数据不触发力卸载。
- 同模块`MarginAdmission`：候选排序前按原生18关节含被动耦合检查，目标内余量40mrad/根位移10mm，范围小则限制为10%；没有扩大实际关节界限。手指目标1.2rad/s，接近0.30m/s/1.2rad/s，几何替换后仍限速。实际状态不被裁剪。
- `tools/run/run_source_ceiling_worker.py`：仅A0，WAM不实例化；旧R无τ后备保留，source表达失败照报；抓后使用原RelativePoseServo/force规则，不改成功、支持、碰撞、持续时间或终点目标。
- `tools/run/run_source_ceiling40.py`：冻结v11源码和同一40初态，独立空闲卡队列，24000控制上限；每回合保存q/dq/force/PD/游标/卸载/候选/阶段/限位。`summarize_source_ceiling.py`输出逐物体死因、哪个关节、阶段、q/上下界/超界量、PD目标、物体抬升、palm/指尖距离与目标误差；重新对齐40个旧初态。

预期数值：source成功14/40→≥32/40；接近/闭合限位12/40→≤2/40；初态对齐40/40；0训练更新。若未达到，仍不跑WAM。候选几何仍是稀疏表面/点位近似，尚无完整扫掠碰撞证明，不冒充安全保证。

```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY -m pytest -q src/task/wam-rl-policy-v3/tests/test_source_ceiling.py --basetemp=tmp/pytest-source-v14-review
$PY src/task/wam-rl-policy-v3/tools/audit/run_source_ceiling_mutants.py
$PY src/task/wam-rl-policy-v3/tools/run/summarize_source_ceiling.py --baseline outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1/summary.json --output outputs/wam-rl-policy-v3/source-ceiling-20261023-r3/baseline-details.json
# 第一次执行；禁止对已有registration重复prepare，复跑请换fresh run路径。
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r3 --prepare
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r3 --detached
```

实测软件：专项10 passed/0 failed/0 skipped；完整v3 279 passed/0 failed/0 skipped（详见tests.log，3条万向节锁警告）；故意错误7/7 killed（忽略压力、允许目标跳变、遗漏余量、篡改抓后伺服、漏交接限速、日志目标失真、继续使用录制世界坐标）。旧真实step复现为2 failed，见原r1产物red.log（另有handover-red.log/telemetry-red.log）；没有把纯import失败冒充复现。
原生解释器`/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`/Torch2.0.1+cu118；策略解释器graspenv Python3.8.20/Torch2.4.1+cu121；native先import IsaacGym。

## 逐物体旧source失败及几何证据

不是把WAM的13/14套给source：source实际12限位（approach1/close11），13个关节事件。
下表超界量单位rad，原始全量q/dq/PD/距离/支持面间隙见`outputs/wam-rl-policy-v3/source-ceiling-20261023-r3/baseline-details.json`。
`stage_contact_not_certified`指阶段认证未过，**不等于没有真实接触**；例如knife/phone末态伺服接近目标，仍可能因点位/阈值未获得阶段认证，不能把它们强行算成功。

| 物体 | 成功 | 死因计数 | 关节及超出名义界限rad |
|---|---|---|---|
|bowl|0/2|held_then_lost:1, physical_table_penetration:1|—|
|camera|1/2|success:1, physical_joint_limit:1|thumb_intermediate_joint:0.131|
|cylinderlarge|0/2|physical_joint_limit:1, contact_without_lift:1|ring_proximal_joint:0.143|
|eyeglasses|0/2|physical_joint_limit:1, lift_without_continuous_hold:1|thumb_proximal_pitch_joint:0.128; thumb_intermediate_joint:0.120|
|fryingpan|1/2|success:1, physical_joint_limit:1|thumb_intermediate_joint:0.118|
|hand|2/2|success:2|—|
|headphones|1/2|success:1, physical_table_penetration:1|—|
|knife|0/2|approach_without_contact:2|—|
|lightbulb|1/2|success:1, held_off_goal:1|—|
|mouse|1/2|success:1, physical_joint_limit:1|thumb_intermediate_joint:0.104|
|phone|0/2|approach_without_contact:1, physical_joint_limit:1|thumb_intermediate_joint:0.100|
|piggybank|0/2|physical_joint_limit:2|pinky_proximal_joint:0.145; thumb_intermediate_joint:0.106|
|pyramidlarge|2/2|success:2|—|
|pyramidmedium|1/2|success:1, contact_without_lift:1|—|
|pyramidsmall|0/2|physical_joint_limit:1, contact_without_lift:1|thumb_intermediate_joint:0.142|
|scissors|0/2|physical_joint_limit:2|thumb_intermediate_joint:0.353; thumb_proximal_pitch_joint:0.101|
|spherelarge|0/2|physical_joint_limit:1, contact_without_lift:1|thumb_intermediate_joint:0.185|
|spheremedium|2/2|success:2|—|
|spheresmall|0/2|approach_without_contact:1, lift_without_continuous_hold:1|—|
|stamp|2/2|success:2|—|

## 无法自证 / 待独立裁决

1. 12N与余量/速度常数是所有物体共用的首版控制参数，尚无新物理回合证明能改善成功率；测试只证明指定输入会退让与限速。
2. 候选使用当前手点几何，不是目标手形的全URDF碰撞/力闭合解；静态目标合法不等于实际接触动态安全。
3. 尺寸化landmark阶段是否对细薄物体过严仍不确定；本轮不偷偷改判据或重标旧成绩。真实接触与阶段认证分开记录，必要的合同更改须显式新版本并同时复核基线。
4. 同一开发面板不是新holdout；即便32/40达标，source与WAM的信息和参考执行链不同，差距不能单独证明只因数据量。
5. Hermes/GPT尚未复跑此实现；三方分歧逐项保留。本轮不动并行v13/原训练，不重训v6，不做RL或完整RELEASE。

预启动修订：原r1已冻结但未获GPU；查出close→lift手指目标跳变与最终目标日志失真，分别由红测试复现。r1 supervisor明确记CANCELLED_PRESTART、0子进程/0控制，原快照和日志保留；物理批次改为fresh r3，不混算。修后对12个真实首限位前状态单步重算：12/12目标准入、6/12触发卸载；这不是反事实仿真，不声称已防止任何旧失败。

第二次预启动修订：r2仍为QUEUED/无子进程/0控制时，生产source-PD→intent边界的红测试确认：物体平移50mm，原source抓姿仍移动0mm。已加入通用参考物体→live物体SE3映射（含姿态），手指目标不变，修后目标同移50mm；不是单初态常数，也不是GTτ/R上界。原r2 receipt明确CANCELLED_PRESTART和原因，fresh r3才是计划物理批次。新10项专项与7个mutant均须通过，40/40参考含实测palm_pose。
