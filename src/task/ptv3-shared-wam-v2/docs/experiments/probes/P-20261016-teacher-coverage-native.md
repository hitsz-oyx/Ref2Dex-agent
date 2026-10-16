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
status: PROMISING
run_id: teacher-coverage-native-20261016-r1
---

# 教师状态覆盖收益的闭环检验

Result: 六序列COMPLETED，PROMISING（仅现有全参考控制器的覆盖收益）；部件归因亦完成，不能单独归给某一部件。
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

## 工程资格

1490ee7f的4步smoke完成；独立checkpoint/raw重推、实际PD、同初态、参考clock、
hold读出复算PASS。四行q/obj/raw初态差0。GPU0原生约18.2GiB（24GiB可容纳），
无外来consumer。工程结果不进入科学面板，正式六序列按原协议推进。

## 六序列结果与独立审计

代码9f2e0e78；每序列24.9–32.0秒，六条总176秒左右仿真进程耗时合计，显存18.2GiB。
六序列等权（mm；p95为各序列p95再等权）：

|角色|物体mean|物体p95|腕部mean|hold45序列数|
|---|---:|---:|---:|---:|
|teacher|34.742|78.708|92.495|6/6|
|original|434.427|1242.361|382.567|4/6|
|expanded|76.476|355.685|100.079|6/6|
|balanced|77.654|349.385|108.104|6/6|

两候选均通过预定门，物体mean改善82.40%/82.13%；这是闭环正向Probe，不是完整
任务成功。hold45可在后续掉物前通过，不能单独替代轨迹质量。s3_mug_drink_1
teacher36.17mm而expanded284.35/balanced293.71mm，其余5条expanded25.73–48.87mm。
该失败没有从汇总中剔除；不能把balanced离线拟合门推广为在线优于expanded。

独立审计先因glob误纳同名.log中止（无物理结果受影响，已限定目录）；训练环境Torch2.4
重推一行xyz最大0.105917mm略超0.1mm容差，未放宽。改用实际部署Torch2.0.1的全六条
新进程重推后全部输出差0，实际PD/同初态/参考clock/hold复算PASS。属于跨runtime
数值差异，不是已发现接线错误。审计输出audit/results.json，保留失败日志供追溯。

## 后续 Decision Note：剩余drink失败的部件归因

问题：明显失败是否主要来自腕部或手指控制？关键证据：扩展模型其余五条物体均值
25.7–48.9mm，但s3-drink284.3mm，teacher36.2mm。先做最便宜的两个部件反事实，
不追加训练；固定s3_mug_drink_1为失败诊断，s5_cup_drink_2为另一drink对照。
每条四行teacher/expanded/teacher-wrist+student-fingers/student-wrist+teacher-fingers，
所有teacher查询来自该行自己的状态；只借部件，不借row0状态。后两行明确教师辅助，
不能计入自主控制成功或主面板。长度/seed/物理设置保持原协议。

若某种替换使s3物体mean降低>=50%、hold45保持且另一种未达门，优先对应部件修复；
若都改善或都无改善则耦合/分布证据不足，记UNCLEAR，先看首次偏离，不扩大同类oracle
搜索。另报s5控制以防牺牲已有行为。每条600s，上轮总30min/2GiB预算内，0/5两空闲GPU；
产物component-attribution子目录。停止条件继承；无模型/claim/数据许可变化。

## 部件归因结果与放置后撤手定位

代码8dd9592a，两个run各约28.5秒，原生runtime独立重推max0、混合前预测/混合后
动作/PD及同初态/clock/hold均PASS（component-attribution/audit/results.json）。

|序列|teacher|expanded|teacher腕+学生指|学生腕+teacher指|
|---|---:|---:|---:|---:|
|s3_mug_drink_1 物体mean/mm|33.33|246.96|33.71|31.42|
|s5_cup_drink_2 物体mean/mm|39.67|29.51|38.16|22.35|
|s3最长held帧|97|77|97|93|
|s5最长held帧|106|106|105|97|

两种替换都救回s3，因此按预设规则部件归因UNCLEAR，不宣称单独腕部或手指是唯一
原因。s5 teacher腕替换比原学生29.51→38.16mm更差；不是普遍单调改进。s3原学生
从主面板284.35到诊断246.96mm、teacher36.17到33.33mm亦提示原生运行/row差异，
本轮不是独立重复Validation，不把微小差异归因到方法。

逐帧检查进一步定位到**放置后的松手—撤离阶段**：原始参考global contact于tick156
从1变0，物体目标已停在桌面，手部参考继续离开。诊断中tick155四行物体都在桌面
footprint内，支持面距离约1–2mm；tick160教师/两hybrid的手物surface gap约76/80/60mm，
学生约5mm。学生在tick163物体误差超过200mm，tick165已经离开桌面footprint，之后
仍保持手物近距离并随手离开。主面板expanded对应tick162超过200mm，结论方向一致。
所以不能把这例简单叫“没有抓住”或从hold45直接宣布任务成功；更具体的探索性解释是
没有正确完成松手/撤离的协调。gap为采样表面距离，不是精确接触力或抓取因果证明。
原数组与可复算事件保存在summary/release-transition.json；曲线summary/tracking.png/pdf。

## 收束后的决策

保留expanded为下一轮物理基线，balanced为覆盖/采样对照；不再扩大本轮oracle实验。
下一轮最便宜的判别应首先审计合法TRAIN中放置—松手—撤离的阶段覆盖/监督误差，再
决定重采样或在TRAIN序列采集学生实际访问状态的teacher PD查询；这六条VAL只能评测，
不得回灌。若做继续训练必须保留相同更新预算的纯离线续训对照，避免将更多步数当收益。
不立即扩大网络、WAM或全量数据，当前核心WAM增益仍未证明。

已追溯trajectory-policy的[P-20261011-reference-goal-dagger](../../../../trajectory-policy/docs/experiments/probes/P-20261011-reference-goal-dagger.md)：
旧单motion的own-state PD查询曾优于离线续训，且明确没有把混合rollout未来轨迹当expert
标签。它仅支持候选方法选择，不是本轮多序列迁移有效的证据；旧失败状态query误差仍大。
当前先做phase覆盖审核，比直接照搬旧DAgger更便宜。无需改MISSION或长期GPU默认规则。

本轮9次native运行全部完成，约4.2 GPU分钟（进程elapsed合计；另有加载/审计），产物
约27MiB，GPU0/5/6/7已释放；其他任务未动。代码和本轮文档提交main，未新建分支。
正式多seed、完整时长任务、更多未暴露物体以及WAM因果比较仍延后，当前标签PROMISING
仅指预设六序列物理覆盖收益，部件归因与松手机制保持探索性边界。

## 阶段覆盖最小审核协议（读数前）

Decision：区分合法TRAIN中松手阶段明显缺样本/欠拟合，还是已有教师分布阶段拟合尚可
但学生闭环访问状态不同。只读取已有TRAIN的native reference global contact1→0，
把转换前12拍至后23拍标记release-window（是松手候选，不保证放桌）；保持106005
合法成员，分别报告all和cup/mug内样本/序列数、expanded/balanced的XYZ与手指PD误差。
参考contact只用于分层标签，不加入模型。不读VAL监督、不更新参数；startup<16单列，
其他状态作描述性对照。若杯类release误差明显更大且>10mm，先测试阶段采样；若已
<=10mm且有多序列覆盖，则优先TRAIN own-state查询，不把缺乏恢复覆盖当已证明结论。
上轮预算内GPU0，600s/128MiB输出，新目录release-coverage；读取/推理完成后停止。
