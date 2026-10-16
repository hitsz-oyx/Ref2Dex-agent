---
schema: ref2dex.probe.v2
probe_id: P-20261016-teacher-coverage-native
experiment_id: P-20261016-teacher-coverage-native
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 6
seed_pool: probe
seeds: [292]
decision_changed_if_positive: retain scratch coverage controller as native baseline before WAM comparison
decision_changed_if_negative: localize first closed-loop deviation before spending on more offline fitting
status: RUNNING
run_id: teacher-coverage-native-20261016-r1
---

# 教师状态覆盖收益的闭环检验

Result: RUNNING。
Decision: 将固定三个scratch终点接入原生物理，不再追加离线配方。
Class: Decision。服务MISSION自训控制底座；检验held/unseen离线PD改善能否进入真实
腕部/物体跟踪。最便宜方法为复用旧六条VAL杯类轨迹，单reset、四并行环境。

## 决策与预定协议

旧覆盖实验balanced离线通过门，expanded均值更好但原杯拟合较差。各取final24000，
不选DEV checkpoint，不加载teacher actor到学生。教师仅执行row0；原始/expanded/
balanced分别row1/2/3；每步各自raw1442，固定原RMS/目标归一化，relative wrist6+
absolute active fingers6经原PD接口转换。教师在其他行仅查询并保存，用于后续诊断，
不干预学生。固定batch64推理规避已知native singleton数值问题。无WAM输入。

面板：s10_mug_pass_1、s2_cup_pass_1、s3_mug_drink_1、s4_cup_pass_1、
s5_cup_drink_2、s5_mug_pass_1。frame0开始，长度frames-25（171/182/214/131/228/177
控制步），保留旧H4/future24支持边界；不将截断序列称任务成功。共同seed292，
禁早停/自适应kappa，无学生重置/救援。先4步工程smoke再正式六条。

主读出：各序列实际物体位置相对原生参考的逐帧mean/mm，再六序列等权；报告腕部
位置mean/mm、object p95、旋转、clipping、连续hold45及启动16步。保留教师校准表现，
不事后剔除失败序列。若candidate相对original物体macro改善>=10%、腕部不恶化>10%、
物体p95不恶化>10%，且至少一条hold45，则本局部PROMISING；两候选均没有改善则当前
离线覆盖转移UNPROMISING；方向/指标冲突或教师校准失败则UNCLEAR。教师全无hold45
或物体macro>100mm时不把学生结果用于否定coverage方法。所有结论仅单seed Probe。

逐步保存raw观测、12D预测、teacher own-state查询、实际动作/PD、clock、腕/物体姿态、
surface/support gap。核对自己的raw行→学生输出、相对PD解码、实际apply、同初态和
参考时钟，指标独立从数组复算。源码tools/run/execute_teacher_coverage.py；
outputs/ptv3-shared-wam-v2/teacher-coverage-native-20261016-r1/。

## Decision Note / 资源

用户2026-10-16明确“空闲的卡都可以用”，本轮临时允许使用0/5/6/7全部空闲卡，覆盖
CAMPAIGN通常保留两卡的默认限制，不改写长期限制，不影响1–4任务。不新分支。
每序列600s/256MiB，全轮含smoke<=30min/2GiB；磁盘至少20GiB，非有限、时钟/PD错位、
外来consumer或超预算停止自身run。固定提交后启动，monitor记录GPU/ETA。
先完成六条，依据首次偏离选择一个最便宜归因，不无界扩充评测。

## Limitations / future evidence

曾多次使用的DEV、完整特权手部参考、单训练seed、单物理reset、四行环境物理差异、
短前缀而非完整任务；不是纯物体条件、不证明少示范/WAM/跨手型。正式多seed和
无暴露面板留待正向Probe，不立即运行。hold45仅测离桌接近持物，不能证明稳健抓取。
