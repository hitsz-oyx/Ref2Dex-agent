---
schema: ref2dex.probe.v2
probe_id: P-20261022-grasp-interface-stat40
experiment_id: P-20261022-grasp-interface-stat40
date: 2026-10-22
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-grasp-interface-stat40
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259]
decision_changed_if_positive: retain WAM reference contribution on the fixed panel and continue executor qualification before RL
decision_changed_if_negative: do not claim tau contribution or stable engineering qualification; retain RL HOLD
status: RUNNING
run_id: grasp-v9-stat40-20261022-r1
---

# v9：四臂，每臂seen40/unseen40，三卡并行（预注册）

Decision：区分新接口的成功是否依赖WAM参考，还是通用闭合/物体servo即可；几何参考是否
替代WAM；同协议GT候选是否达到物理上界。结果决定保留参考、修执行合同，或建议改造WAM。
最便宜的可判别方式是固定权重与全部低层控制，运行80组配对初态/种子×4臂，不训练。
机主v9明确要求放大，旧n5结果不并入新分母，不因中途显著/全零提前结束。

## 冻结任务与主终点

沿用已授权抓取—抬升**HOLD**任务：初始物体位置上方8cm，position-only、不要求支撑。
不是501条完整RELEASE任务。每回合**600 controls/20秒**；成功或编码失败均不提前停，
只在原真实物理安全gate触发时停止，物理中止计失败。固定时域后独立报告：

- **主终点**：完整观察20s，S2c `current.hold`（连续1秒、接触后再升≥5cm）为true，
  同时末态物体XYZ目标误差≤15mm、无物理中止。达到一次后再掉落不算主成功。
- 分别报告S2c approach/contact/lift/hold的ever和final；最长连续持握与20s生存数；
  release为该HOLD任务的**不适用**，不伪造release授权，也不把自然掉落算成功释放。
  `drop_after_hold`另记。若机主要求完整RELEASE，须另冻结可达支撑目标与释放合同。
- 单报所有回合终止时XYZ误差完整向量/分位数、≤15mm计数与Wilson；另列完整20s
  生存者的误差，避免把提前物理中止当作共同20s测量。主终点的中止全部计失败。

此协议加严旧“5cm+1s即停”的口径，不把二者直接混为同一成功率。

## S2c测量合同修复，门槛不变

原RewardFeatures把native `hand_base_link`腕原点当palm点；真实URDF掌部网格沿局部Z延伸
约161mm。真实S2b205状态有2秒持握，但原点距物体最小117.566mm，导致S2c100mm
approach gate永不触发。新增本Task `PalmSurface`，用**实测palm SE3 + 实际URDF
collision mesh**到原object32表面的最近顶点距离作为S2c palm_gap；不移动手点/WAM/R，
不改100/50mm门槛、力阈值、抬升量或持续时间，不修改v6的S2c实现。

同一保存源轨迹：approach/contact/lift/hold从全false→全true，时刻
[.9667,1.1333,1.6667,2.6667]s，最长持握仍2.0s；最小CAD palm gap .109mm。
来源是实体几何角色修正，非阈值调优。此距离是离散顶点表面近似，不是精确碰撞距离；
手物体穿透没有因此被认证。原物理安全gate原样保留。测试故意换回腕gap，S2c拒绝；
另拒绝“早期1秒持握/末端掉落/目标超15mm也算成功”的错误。

## 四臂与信息来源

| 臂 | 参考提供者 | 共享消费链 |
|---|---|---|
| A1 | 冻结step11000 WAM，live H4+goal/event，24帧native11 | 原R→同v7 GraspIntent→实测三阶段→PD |
| A2 | 无τ，live物体+冻结通用抓姿/预形/闭合模板；模板来自原cylindersmall单示教 | 同GraspIntent/实测阶段/物体servo/PD，不调用R/WAM |
| A3 | S2e `object_local_hand/geometry_future`；每个参考未来姿态转物体局部再映射live物体 | 同A1的R/24帧编码/前8帧/消费/PD |
| A4 | S2d整段未来统一 `T_obj_live @ inv(T_obj_ref[t])`；真实成功机器人手点参考 | 同A1的R/24帧编码/前8帧/消费/PD |

四臂奖励无更新，初始化、goal、阶段机、PD、观测、预算与安全完全一致。A3/4使用同一
条已录制、通过S2c抓取抬升持握的机器人参考，不执行source action/PD。A3不输入live手
来伪装参考；只在统一Δ编码时减当前实测手。所有未来首帧是t+1/30s，不把当前帧塞入τ。
A3/4参考末端不足24帧时明确invalid，继续预定无τ后备，不填充假未来；覆盖率另记。
源有真实primitive成功，不等于已认证本次20s+15mm主终点的上界，A4必须实测后才有资格。
A2模板示教先验、A3/4额外oracle示教信息明示；这是固定执行链比较，**不是训练数据量
对齐的RL样本效率试验**，也不能由结果自动推论普遍预训练价值。

## 初态、独立性与顺序

每臂seen40和unseen40，合计320回合/最多192000controls。seen是已调参的
`s1_cylindersmall_pass_1`一族40种子；unseen为以下8个未在此接口调参的物体/初态各5种子：

alarmclock(`s10_alarmclock_pass_1`)、banana(`s10_banana_pass_1`)、
cubesmall(`s10_cubesmall_inspect_1`)、cup(`s10_cup_drink_1`)、
cylindermedium(`s10_cylindermedium_inspect_1`)、flashlight(`s10_flashlight_on_1`)、
gamecontroller(`s10_gamecontroller_pass_1`)、hammer(`s10_hammer_use_1`)。

选择在新策略评估前冻结：按原序列名排序，engineering PASS、缓存几何可用，排除全部已
暴露接口物体{cylindersmall,airplane,apple,cubelarge,duck,mug}，只按**旧录制**S2c有≥1秒
持握选取首8种不同物体。筛14条得8条，完整screened记录保留，未用四臂成绩筛样本。
未修改标签/未将UNKNOWN改HOLD；这里是单独声明的合成抓取抬升任务。unseen相对接口
调参，不保证相对WAM/父R预训练未见；不宣称40物体或40独立训练种子。

80个配对trial固定native seeds100..179、model seeds180..259（全在Probe池，无Validation
holdout）；同trial四臂初态/seed/卡完全相同。拉丁方轮转，seen/unseen各臂各位置10次。
**每个episode重新启动native进程**，清除上一失败回合隐藏solver状态；不沿用跨臂接触历史。
model在worker中冻结保留，shape按物体缓存，R与未用校准R构造顺序保持原约定。三张卡
各自同卡运行model+native，显存预估<6GiB/24GiB；GPU4/5/6用前查空闲，与S2e冲突排队。

Wilson/Fisher针对此固定面板的episode trials；8物体重复不是40独立物体。seen一族不支持
跨初态泛化；补报同初态配对McNemar，不把不配对Fisher冒充随机物体总体推断。

## 预注册统计与停止条件

每个子集分别给每臂k/n、Wilson95、六个臂间Fisher双侧p，不混seen/unseen：

- 主判据A1>A2：Fisher p<.05 **且** Wilson(A1).lower>Wilson(A2).upper；只有n=40齐全
  才允许正式判定。全局稳健贡献要求seen与unseen都通过，单独通过只作该子集结论。
- 次要A1 vs A3同样报告；原始p值保留，次要多重比较只作描述性结果。
- A4≥32/40、A1≤8/40时标记WAM改造/重训建议；**不自动启动训练**，沿用机主不重训
  v6的纪律，交Hermes处理。A4不通过则不得把失败归因于WAM质量。
- 对照数值：0/40 Wilson上界≈.0876；32/40与8/40应能过主联合门槛；n40不是保证任何
  真效应有80%功效，特别是本联合判据比单一Fisher更严。
- 每worker6h、总192000controls、总产物≤约3GiB预估（停止新增写入阈值磁盘20GiB）；
  无显著性提前停、无失败后补种子。技术失败保存partial、预算照记，同registration恢复
  原assigned episode，不能标物理失败或静默删样本。控制器/评分语义变化必须新协议。
- 0训练更新，三臂RL HOLD，flowmatching不恢复。统计前逐回合从原始物理trace重算阶段
  与终点，核PD/时钟/初态；单元PASS不等于物理资格PASS。

## Diff与可复现命令

新增`src/wam_rl_policy_v3/stage_evaluation.py`：CAD palm测量与严格末态门；
`src/wam_rl_policy_v3/binary_statistics.py`：Wilson/Fisher；
`tools/run/prepare_large_grasp_panel.py`：旧源资格与冻结未见面板；
`tools/run/register_large_grasp.py`：80配对trial分配；
`tools/run/run_large_grasp_worker.py`：三卡单卡同驻、native每episode重建、四臂同链；
`tools/run/summarize_large_grasp.py`：分子集统计与未齐样本阻断。相对本Task根目录。

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
TOOLS=src/task/wam-rl-policy-v3/tools/run
$PY $TOOLS/prepare_large_grasp_panel.py --output outputs/wam-rl-policy-v3/grasp-v9-reference-panel-20261022-r1
$PY $TOOLS/register_large_grasp.py --panel outputs/wam-rl-policy-v3/grasp-v9-reference-panel-20261022-r1/panel.json --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/registration.json
# 三行在独立进程同时启动；入口发现被占即拒绝，不抢卡。
$PY -u $TOOLS/run_large_grasp_worker.py --registration outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/registration.json --worker 0 --gpu 4 --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g4
$PY -u $TOOLS/run_large_grasp_worker.py --registration outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/registration.json --worker 1 --gpu 5 --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g5
$PY -u $TOOLS/run_large_grasp_worker.py --registration outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/registration.json --worker 2 --gpu 6 --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g6
$PY $TOOLS/summarize_large_grasp.py --registration outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/registration.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g4/manifest.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g5/manifest.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g6/manifest.json --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/summary.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q -p no:cacheprovider --basetemp=tmp/v9-tests src/task/wam-rl-policy-v3/tests
```

所有产物目录需不存在，复跑改run_id；面板prepare不幂等覆盖。无需重建已保存面板才能复跑。
环境：model graspenv Python3.8.20/Torch2.4.1+cu121；native固定venv Python3.8.10/
Torch2.0.1+cu118。采样、mesh、参考、顺序与代码快照均保存在run目录。

## 结果

正式评估已从`07d4db08`启动，GPU4/5/6各约4.9GiB，model/native同卡驻留。
首批9/9完整轨迹独立重算一致，发布标签反转mutant被拒绝；不把中途结果当正式判定。
236 passed/0 failed/0 skipped（18.58s）预检通过。实际3worker身份g4/g5/g6与registration均保留。
完整40/臂/子集后补数字，旧n5不入新分母。


原始状态重算（CPU，无WAM推理）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $PY $TOOLS/rescore_large_grasp.py --worker outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g4 --verify-mutant --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g4-rescore.json
```

期望`PASS matched=n`且`publication_label_mutant_killed=true`；g5/g6同理。
统计入口额外回归：即使伪造5/5 vs0/5已显著，也必须因n<40阻断发布，不能偷跑停止规则。


掌部测量正负控制可独立复现：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 $PY $TOOLS/check_stage_palm_source.py --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/palm-source-check.json
```

期望PASS：原点版approach/contact/lift/hold全false；CAD版全true、最长2s。它只核CPU测量
合同，不是正式四臂样本、不把source-PD动作回放当A4控制。

Diff定位：`stage_evaluation.py:12`掌部几何、`:43`严格末态；`binary_statistics.py:6` Wilson、
`:13` Fisher联合门；`run_large_grasp_worker.py:10`每episode新native、`:113`固定jobs、
`:175`禁止成功即停；`register_large_grasp.py:7`冻结80个配对trial；
`summarize_large_grasp.py:10`分子集聚合并阻断n未齐；`rescore_large_grasp.py:10`从原状态
另算终点/阶段/PD；路径均相对本Task的`src/wam_rl_policy_v3`或`tools/run`。


九字段配对初态核验（`tools/run/verify_large_initials.py:9`；首批38/38配对一致，故意手点平移mutant被拒绝）：

```bash
$PY $TOOLS/verify_large_initials.py --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g4/manifest.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g5/manifest.json --manifest outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/g6/manifest.json --output outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/initial-parity.json
```

完整时预期80/80配对通过，九字段最大误差≤2e-6。它不认证隐藏solver真值，fresh-native程序用于消除前回合历史。
完整单元/合同测试已更新为237 passed/0 failed/0 skipped（19.54s），实际运行代码仍冻结`07d4db08`。
