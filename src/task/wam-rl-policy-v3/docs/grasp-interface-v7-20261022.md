# v7：抓姿、预形/闭合与物体目标的测量闭环接口

本轮替代v6“两字段”的新控制原型，不把历史接口当已解决；只实现和验证，不做额外归因实验。
原R作为**冻结的τ→机器人意图适配器**保留：读真实live WAM及当前q/dq，输出的腕PD目标转
成物体坐标下的关键腕位姿，手指PD目标完整保留为预形/闭合目标。控制器收到的是
`GraspIntent`，不是按参考帧号重放的手点。接近阶段保留原R已具备的approach能力，而不是
把“未来0.8秒末端”误当成完整抓姿后缓慢逼近。WAM几何给另一候选抓姿，两关键腕姿即可。

接口保留三项：

1. `approach_pose_object/grasp_pose_object`：短期可执行腕姿与接触候选抓姿，均物体坐标。
2. `preshape_q/closure_q`：原R对真实WAM的6主动指关节预形，及有界闭合目标；不再丢手指
   信息或只调用无τ先验。低层仍用原18关节耦合，未更改机械结构。
3. `postgrasp_object_pose`：来自同一task的真实物体目标（初态上方8cm），不是按时间给腕
   高度加10cm的脚本。位置任务保持测得抓取朝向；SE3接口保留目标旋转。

三个行为阶段均由测量推进：接近由实测两指近表面（现有observer near_fingers）或实测
腕距候选姿≤25mm且指尖gap≤40mm触发闭合；闭合读现有FeedbackObserver的
`force_valid/contact_observed`，接触持续实测elapsed≥.12s才抬升；抬升/保持每tick根据
**实测物体位姿误差/速度**给腕闭环目标，误差≤15mm才标行为hold。失去实测接触回close。
时间仅作实测contact dwell，不以预计接触时刻/参考tick/固定抬升时长推进phase。

S2c graded stage信号仍只读记录，不作为contact传感器；复用原FeedbackObserver的测量
contact，不重写S2c dense reward或正式任务评分。物理成功仍是同5cm/force耦合/离桌连续
elapsed1s窄指标，**不把新行为hold或S2c分数当成功**。原物理安全gate不变。

## 冻结的效果验证预算（运行前）

Decision：只判断这次接口能否改善原R+WAM；若≥4/5且优于同期a则保留继续开发，否则不晋升。
最便宜比较为原a与新b各5次，共10次，不额外增加GT归因链。沿用登记的单轨迹probe
`P-20261022-single-tau-calibration`，seed195/196、同已见cylindersmall初态、同R/WAM11000、
同native1env/30Hz、每回合最多300控制步（10s），**最多3000环境controls/1800s**。
成功/原安全中止即停；顺序AB/BA/AB/BA/AB。运行前冻结实现与参数，单批完成，不追加重复，
不训练v6或R。两链9可见初态对旧run要求≤1e-6。历史2/5只作主指标起点，另报同期a，
不把不同run相减当统计效应；五次同轨迹重置不是五独立任务/训练种子。

每链必须报告n、全5次最长持握、ever/final、clip、限位、phase触发、物体目标误差。
目标b≥4/5是**效果期望而非预先声称成功**。不加新审计条目，不补历史卡，正式三臂仍HOLD。
run_id=`grasp-interface-v7-20261022-r1`；GPU5模型/GPU6物理，仅两卡实测空闲时用。

## diff、验收数字和复现

- `src/wam_rl_policy_v3/pose_consumer.py`新增`GraspIntent/grasp_intent_from_retargeter`：
  τ解释的腕与指目标都保留；单测正反输入保证改变预形会改变执行目标，approach PD往返误差≤1e-6。
- 同文件`MeasuredGraspConsumer`：实测位置→闭合，实测有效接触dwell→lift，物体目标误差
  驱动lift/hold；期望只有时间或预测mode变化不能推进phase，物体从目标下方换到上方时
  腕z动作由正变负，force invalid不能触发抬升。
- `tools/run/run_grasp_interface_improvement.py`：同初态两链各5，模型/物理源码冻结；期望
  COMPLETED/10回合/≤3000controls，b主指标历史2/5→至少4/5。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=tmp/grasp-interface-v7-repro-tests src/task/wam-rl-policy-v3/tests/test_grasp_intent.py
# 期望8 passed/0 failed；覆盖无实测位置/无有效force等相反输入，不以预测phase代替传感器。

nvidia-smi
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=$PWD/tmp \
/home2/wyy/miniconda3/envs/graspenv/bin/python -u \
 src/task/wam-rl-policy-v3/tools/run/run_grasp_interface_improvement.py \
 --output outputs/wam-rl-policy-v3/grasp-interface-v7-independent-review --native-gpu 6 --model-gpu 5
```

输出必须fresh；依赖原R/base checkpoint、WAM11000、stats/geometry以及原始录制/物体资产。
client graspenv Python3.8.20/Torch2.4.1+cu121，native dexplore_v120_train Torch2.0.1+cu118。
不覆盖任何checkpoint或原输出，GPU0–3/7不动，GPU4留空。

腕SE3附带原R求解的Euler IK分支（同一目标的关节代表），避免连续腕关节在等价姿态解之间
跳变；每tick前向校验分支确实到达该SE3。它不增加任务信息，也不是参考时钟轨迹。


## 工程中断后的修复与预算续接（r2启动前）

r1于412controls中断：完成原A0（93controls、持握.3s、限位中止）和第一版B0
（300controls、持握0、close252步未取得有效contact），再执行B1的19个control后，
腕姿IK一致性检查拒绝。所有原始轨迹保留，退出回执可查；不把未完成B1算成功/失败率。

修复两项，不放宽物理gate：腕姿适配器和执行器统一使用**同tick q/palm**来算mount；
接近位置触发闭合后，在有效force contact前继续用live抓姿修正腕目标，不能提早冻结。
首次有效contact才锁定close pose，实测force dwell完成后才交物体目标servo。
回归先**2 failed/6 passed**，分别重现错用reset mount、闭合过早冻结；修复后**8 passed/0 failed**。

修正版固定为r2；保留未改A0作为同期baseline的一项，补其余4A和修正版5B。第一版B0
不并入修正版分母，仍作为具名工程结果0/1列出，partial19也保留；这是版本变化而非按
成败挑样本。r2 ledger先扣**412controls和r1全wall**，原总上限仍3000controls/1800s，
不倒改旧预算。若剩余预算不足完成各5次，必须报告真实n/中断点，不补跑、不改成功率。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TMPDIR=$PWD/tmp /home2/wyy/miniconda3/envs/graspenv/bin/python -u  src/task/wam-rl-policy-v3/tools/run/run_grasp_interface_improvement.py  --carry-run outputs/wam-rl-policy-v3/grasp-interface-v7-20261022-r1  --output outputs/wam-rl-policy-v3/grasp-interface-v7-resume-review --native-gpu 6 --model-gpu 5
# fresh输出，原A0只读软链接；全部已花预算计入。修正版本身亦可用上节不带carry的命令复现。
```
