# v7：三项抓取意图接口，最终窄任务5/5

主指标：历史原R+WAM **2/5 → 最终接口5/5**；本轮未改原R同期baseline **3/5**。
这是同一已见cylindersmall初态上的工程Probe，**不是泛化、统计显著性或RL样本效率证明**。
只改消费和执行相位，不为归因再加GT链，不训练R/v6；完整任务资格和三臂RL仍HOLD。

## 最终实现

保留原R作为**冻结的live WAM→机器人意图适配器**。R读原真实8步τ与实测状态，其腕PD目标
转换为物体坐标关键腕姿，6主动指关节目标完整保留；额外WAM24步几何给一个候选抓姿。
低层只收`GraspIntent`：

1. `approach_pose_object/grasp_pose_object`：短期可执行腕姿与候选抓姿，两关键位姿即可。
   附带同一SE3目标的Euler IK分支，避免连续腕关节切换等价解；不增加任务信息。
2. `preshape_q/closure_q`：原R对真实τ的手指预形，以及有界闭合目标；**close阶段继续保留
   live预形/抓姿，不能刚靠近物体就冻结手腕和手指**。
3. `postgrasp_object_pose`：同一task给定的物体目标（初态上方8cm）。这不是按经过时间
   给腕加10cm的动作脚本，也不是GT实际未来。

阶段由测量推进：

- **approach→close**：现有observer实测near_fingers≥2，或实测腕距候选姿≤25mm且指尖
  gap≤40mm，开始闭合；预计contact帧或经过时间不能触发该切换。
- **close→lift**：继续用真实live WAM+原R修正腕姿和手指预形，直到现有FeedbackObserver
  报有效force contact持续实测elapsed≥.12s且`grasp_established`。此时捕获**当时**腕姿与
  闭合目标，再交接给物体目标servo。没有用graded stage分数冒充接触传感器。
- **lift/hold**：每tick读实测物体位姿/速度及q/dq，用物体目标误差给腕PD修正，手指由
  测量force有界调整；失去实测接触回close。物体目标误差≤15mm才标行为hold。位置任务
  保持测得抓取朝向；SE3接口保留目标旋转。没有固定抬升时长或参考帧时钟。

S2c stage/dense reward实现未修改，其stage只读记录；控制相位直接复用原FeedbackObserver
测量contact/持握信号。成功仍由原**≥5cm、force耦合、离桌连续elapsed1s**窄witness判定，
不把新行为hold或S2c分数当成功。原物理安全gate、动作范围、voxel网格均未放宽。

## 冻结协议与最终全量结果

Decision目标：b≥4/5且优于同期a才保留此接口继续开发；否则不晋升。不再验证“是不是唯一
原因”。最便宜比较为a/b各5次；同已见轨迹、同R/WAM11000、native195/model196、30Hz/env1、
每回合最多300controls/10s，成功或原物理gate触发即停。原注册AB/BA/AB/BA/AB；工程中断后
携带未改baseline，故实际分布跨进程（见下），不宣称隐藏physics状态完全一致。

总上限**3000环境controls/1800执行wall秒，包含失败尝试**；0学习更新，不追加归因链。
沿用已登记`P-20261022-single-tau-calibration` probe seed195/196，run各有独立身份。
最终接口代码`5e187dc040d6e1cf1ecadb15a922e4e129eddd58`，run=`grasp-interface-v7-20261022-r3`。

| 链（n=5，block0→4） | 最长持握elapsed秒全分布 | ever / final | controls | 最终输出clip | 关节安全中止 |
|---|---|---|---:|---:|---:|
| 未改原R+WAM a | [.300,.900,1,1,1] |3/5；3/5|701|0|1/5|
| 最终三项接口 b | [1,1,1,1,1] |5/5；5/5|530|0|0/5|

**效果目标≥4/5已达到**；最终b的5次全部按实测持握交接到物体servo，交接时刻
[2.5333,2.3667,2.3667,2.6333,2.9000]s。累计行为approach/close/lift/hold controls为
240/144/145/1。1s成功即停，因此持握数是协议删失下界，不是持握极限。

9个可见初态字段对旧共同初态最大误差**0**；最终新执行actualPD合同最大误差
**2.0926e-7**，含所有工程尝试的最大误差**2.9460e-7**。最终输出clip不代表原R内部raw
未超界（adapter沿用原R有界输出，未另存上游raw统计），不据此宣称全链零饱和。

**post-grasp目标已入闭环，但精确到位未全部验收**：b停止时物体目标误差为
[121.145,46.130,12.794,80.332,80.246]mm，仅1/5≤15mm；实际行为hold也仅1例触发。
其余在lift servo过程中已满足原1s持握而停止。物理持握5/5不能替代精确8cm到位或完整
RELEASE；这轮不追加运行去补更长时域、终态、泛化或GT executor资格。

## 工程失败与全部预算（没有删掉旧版本负结果）

| run | 当前版本新增controls | 完整旧B结果 | 中断 |
|---|---:|---|---|
| r1 `bd671f3a` |412|0/1、最长0s、300controls|腕姿/IK检查拒绝，partial19controls；A0已完成93controls |
| r2 `e2b22351` |1316|0/4、最长全0、748controls|本线程SIGINT自有client，partial66controls；新增A1..A3完成502controls |
| r3 `5e187dc0` |636|最终B5/5、530controls|正常退出0；新增A4为106controls |

全部 **2364/3000 controls**，其中最终对比1231、旧版B和partial共1133；总执行wall
**330.888/1800s**（50.323+187.697+92.867，不含改代码的人时）。旧版本5个完整B结果
均为0、两个partial共85步，全部保留；不并入最终版本分母，也不隐藏其预算。

实现修复有可重现的红绿回归：reset mount与过早冻结腕姿两项先2 failed/6 passed→8 passed；
close覆盖live预形另先1 failed/8 passed→9 passed。最终5B均使用同一冻结修正版；基线A0..A3
复用未改分支的完整结果，A4在r3。所有相同权重/初态/评分身份保留，但跨进程隐藏接触warm-start
不能自证相同，故只报告局部工程改善，不能升级论文级优越性。GPU5/6均释放至2MiB。

## diff、复现命令、期望数值

所有路径相对`src/task/wam-rl-policy-v3/`：

- `src/wam_rl_policy_v3/pose_consumer.py:159`：三项`GraspIntent`；`:172`：冻结R的腕/指目标
  转换。期望approach PD往返误差≤1e-6，修改预形会改变执行的指关节目标。
- 同文件`:194`、`:204`：测量相位和物体目标servo。期望预测mode/纯时间不能推进phase，
  无有效force不能抬升，close保留live预形；物体从目标下方换到上方，z动作正→负。
- `tools/run/run_grasp_interface_improvement.py`：真实native两链各5，支持显式carry全部
  预算/未改baseline；期望COMPLETED、n各5，效果目标b≥4/5。本次实测5/5。
- `tests/test_grasp_intent.py`：9项语义回归，含故意错误输入及已实际跑红的失败案例。
  最终全v3 **220 passed /0 failed /0 skipped，17.69s**；不把测试PASS当物理效果证明。

```bash
git diff 2851060a 5e187dc0 -- src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py \
 src/task/wam-rl-policy-v3/tests/test_grasp_intent.py \
 src/task/wam-rl-policy-v3/tools/run/run_grasp_interface_improvement.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/grasp-interface-v7-repro-tests src/task/wam-rl-policy-v3/tests/test_grasp_intent.py
# 期望9 passed/0 failed，覆盖真实状态/目标语义，不能以导入成功代替。

# 只打印既有结果与完整预算，无新仿真；期望a=3/5,b=5/5，总2364，target_met=true。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/run/summarize_grasp_interface.py \
 --run outputs/wam-rl-policy-v3/grasp-interface-v7-20261022-r3

nvidia-smi
# 如需独立物理复现，仅GPU5/6仍空闲时运行；fresh输出、不带carry即完整新2×5。
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=$PWD/tmp \
/home2/wyy/miniconda3/envs/graspenv/bin/python -u \
 src/task/wam-rl-policy-v3/tools/run/run_grasp_interface_improvement.py \
 --output outputs/wam-rl-policy-v3/grasp-interface-v7-independent-review --native-gpu 6 --model-gpu 5
# 完成期望n各5/≤3000controls；≥4/5是效果目标，跨GPU physics不保证逐位复现5/5。
```

输出必须fresh；依赖已有R、WAM11000、stats/geometry和源轨迹/物体资产。client为graspenv
Python3.8.20/Torch2.4.1+cu121，native为dexplore_v120_train Torch2.0.1+cu118。
本轮只用物理GPU5/6；GPU0–3/7不动。carry命令增加`--carry-run <旧run>`，严格保留总账；
r3实际carry r2，r2 carry r1，manifest链可核对。最终权威输出为r3的`summary-final.json`，
逐回合`trajectory.npz/consumer.json/stages.json`和冻结runtime-code均保留；r1/r2原样保留。
测试、红绿日志、执行日志在`outputs/wam-rl-policy-v3/grasp-interface-v7-20261022-preflight/`。

仓库门禁`python3 tools/verify.py --changed`实测 **VERIFY PASS**，日志preflight/verify.log。
自有client/native均退出（r3 native_returncode0），不使用剩余636步追加实验。
