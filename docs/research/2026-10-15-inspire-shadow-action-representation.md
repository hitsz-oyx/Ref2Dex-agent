# Inspire / Shadow shared 动作表示独立审查（2026-10-15）

范围：只读当前源码及已有 rollout，以 CPU 做数组与控制器转换审计；没有模型推理、训练、GPU占用或新仿真。服务当前决策：是否值得在 Inspire 内做固定输入的 native-command 与 geometry-lift 配对 Probe。跨手观察不能识别动作表示的因果效应。

## 主要发现

1. **给定同一当前 q，Inspire 首拍 geometry-lift q_PD 与其12维独立原生命令信息等价。**现有数据没有发现索引、标签时钟、Euler分支或裁剪导致 GT 首拍无法还原的证据。但信息等价不意味着训练损失、可拟合难度及闭环误差相同。
2. **两条 shared 路线的 teacher 输入完整性不同。**Shadow 动作头直接得到与 teacher 相同的完整300D状态；Inspire 动作头仅87D实测状态，加 shared 表征，缺 teacher 的完整1442D和同刻参考分支 t+1、t+16。当前 shared 输入包含预先已知物体目标，不等于 teacher 的整套参考手/物状态、误差与接触信息。
3. **未来 action chunk 语义确有差异。**Inspire geometry-lift存 query 冻结绝对 PD 目标，消费时用 live q 求原生命令；native chunk存固定命令。两者离开 teacher 状态轨迹后不再等价。但已完成 Inspire shared 抓取面板使用 K=1，因此不能用未来 chunk 语义解释该面板失败。
4. **“q_PD 对比归一化 action”不是“未归一化对比归一化”。**两边动作损失都除各自 TRAIN 标准差；区别包括监督坐标、冗余旋转3维、每维尺度、输入完整性及 actor LR。

## 源码合同

### Inspire

原生映射见 [dexplore_inspire.py](../../third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py:38)：

- wrist 0:6：`q_PD = q_live + offset + scale * a`，前三scale=1，旋转三维scale=π。
- finger：`q_PD = offset + scale * (1+a)/2`。
- 耦合 distal/parent：7/6、9/8、11/10、13/12比值1.05；16/15比值0.6；17/15比值0.8。
- 因此只需 `ACTIVE=(0,1,2,3,4,5,6,8,10,12,14,15)` 的12维命令。其余6维 teacher输出虽可能非零，会被原生耦合覆盖，没有独立物理执行自由度。

标签与逆映射见 [action_targets.py](../../src/task/ptv3-shared-wam/src/ptv3_shared_wam/action_targets.py:69)、[pd_retargeter.py](../../src/task/trajectory-policy/src/trajectory_policy/pd_retargeter.py:40)：

- 所有8拍均相对**同一个 query q[t]**编码，既不是未来逐帧实测q，也不是 q_ref。
- 15D排列是：XYZ相对位移3、相对SO(3) rotvec3、6个手指相对角度、连续Euler差3。
- **首拍XYZ标签就是原生平移命令×scale；最后3维Euler lift就是原生旋转命令×π。**6维手指标签是相对当前手指角的PD位移，而原生命令是绝对关节目标区间的归一化坐标。
- 因而首拍已经保留与原生命令线性等价的全部腕部6维，额外又监督SO(3)旋转3维。真正改变的包括手指坐标与额外旋转监督/权重；不能只凭改用native标签预期XYZ会自然改善。
- `freeze_chunk`加回 query q，并将相对rotation组合到世界orientation；`consume_target`利用 lift选回正确Euler winding，再按live q逆控制器并裁剪[-1,1]。保留lift避免只用SO(3)丢失广义关节winding。

状态87D为 q18、dq18、物体局部手点33、物体姿态12、物体速度6；没有完整原生 body/contact/reference观测。见 [retargeter_data.py](../../src/task/trajectory-policy/src/trajectory_policy/retargeter_data.py:19)。shared-only头未读取显式预测τ；见 [shared_actor.py](../../src/task/ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/shared_actor.py:26)。

### Shadow

原生执行见 [shadow_hand_grasp.py](../../third_party/UniDexGrasp2/dexgrasp/tasks/shadow_hand_grasp.py:909)：

- canonical24D，前三根部力、后三根部力矩，经 pose_vec 到执行坐标并乘native比例。
- 后18维经各actuator上下限的affine map成为手指PD位置目标；当前配置 absolute control、actionsMovingAverage=1，无跨步目标平滑。
- 该模型没有与 Inspire腕部XYZ PD位置直接对应的根部目标。
- teacher状态300D含22维q、dq、DOF力、五指尖完整状态/传感力、手根位姿、上一拍24D action、物体位姿/速度/目标偏差、64D视觉特征；目标手姿29D在本UniDexGrasp++版本置零。准确slice见 [compute_full_state](../../third_party/UniDexGrasp2/dexgrasp/tasks/shadow_hand_grasp.py:735)。

学生直接使用该完整 `clip(observation,-5,5)`，并监督原clock下 `action[t:t+8]`；见 [data.py](../../src/task/shadow-finetune/src/shadow_finetune/data.py:43)。此前全TRAIN1,536,897首拍label已经用官方actor穷举重放，最大误差1.12057e-5，MAE6.06093e-8，1e-4阈值异常0；这是teacher身份/时钟证据，不证明闭环或未来动作已拟合。报告：[full-teacher-label-audit](../../outputs/shadow-finetune/full-teacher-label-audit-20261015-r1/report.json)。

## 本轮CPU数值审计

数据：[660补采manifest](../../outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1/manifest.json)。逐条读取其trajectory.npz内q、pd_targets、applied；使用manifest原生offset/scale，不调整坐标或标签。全部660条、165,677个控制：

| 检查 | 最大绝对误差 |
| --- | ---: |
| `q_PD[t], q[t]`逆原生控制器，与记录的12维ACTIVE applied比较 | 5.96046e-7 |
| 原生6组指关节耦合 | 0 |

未使用的6维teacher命令最大幅度为1，这不等于漏掉独立动作：原生转换确实会覆盖其PD目标。不能要求重建的18维命令逐值相同来判定物理转换错误。

geometry闭环转换抽样：每10条取一条（66条），对原manifest admitted_ticks每5个取一窗，共2,190窗，每窗8拍。

| 检查 | 与记录ACTIVE applied的最大绝对误差 |
| --- | ---: |
| encode→freeze→consume首拍，同一q[t] | 5.66244e-7 |
| 后续7拍，消费时使用对应teacher真实q[t+k] | 3.87430e-7 |

这验证所查数据上的ACTIVE顺序、Euler分支和GT逆转换；抽样geometry检查不声称覆盖整个660数据的所有rotation分支。纯PD逆映射是全量检查。

另把消费的live q人为固定为query q，仅作数组语义反例（没有新仿真）：后续k=1/3/7，恢复腕部命令相对原始teacher命令的平均L1分别0.0179020/0.0414607/0.0823828。这正是因原生增量控制依赖消费时live q所致，不能据此指控现行buffer错误。

已有独立全teacher输入重建还覆盖全部660序列：delta0观测最大误差2.38419e-6，delta1+16恢复官方teacher action最大误差2.89083e-6，PD XYZ最大误差0.000794562mm；见 [full-teacher-cache manifest](../../outputs/ptv3-shared-wam-v2/full-teacher-cache-20261015-r1/manifest.json)。因此当前没有把一般标签错拍或teacher随机性作为首要解释的证据。

## loss及优化差异

Inspire [robot_loss](../../src/task/ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/robot_model.py:62)、Shadow [loss](../../src/task/shadow-finetune/src/shadow_finetune/model.py:44)均为：同实现forecast loss + `mean(abs((pred_action-target)/TRAIN_action_std))`。当前Inspire anchor_weight=0。两边actor均先预测标准化量，再用buffer scale/mean解码输出，故不能把Shadow称为唯一做标准化的一组。

但下列项不同，不能把loss数值直接横比：

- Inspire15D中旋转同时占SO(3)3和Euler lift3，Shadow24D为root6+finger18；平均L1的组权重自然不同。
- Inspire物理label的标准差floor=1e-6；native label及Shadowfloor=.01，且各自由自己的TRAIN分布拟合。源码：[initialize_retargeter](../../src/task/ptv3-shared-wam/src/ptv3_shared_wam/robot.py:93)、[Shadow normalization](../../src/task/shadow-finetune/src/shadow_finetune/data.py:118)。标准化相同不等于相同loss函数对实际控制误差的加权。
- 当前Inspire actor LR5e-5，Shadow5e-4；几何/shared LR相同5e-6/1e-5。两者大规模阶段global batch512，但Shadow旧1k阶段batch64；训练步数/曝光不同。
- Inspire30Hz每拍控制、Shadow60Hz每拍控制；两者forecast均30Hz、future24=0.8s，action8持续时间不同。

## 对当前决策的含义

最便宜的可判别 Probe 是**在Inspire内部**固定数据/输入/父WAM/采样/优化预算/标准化规则/执行K=1，仅改变动作表示，并用相同held-out首拍native command误差、物理PD误差与同三条teacher-positive native面板作比较。可先训练短期actor并检查是否出现显著信号，再决定是否投入完整joint训练。若同时把输入87D换成完整teacher输入，不能将改善单独归因表示。

当前证据排除了“GT geometry-lift首拍本来就还原不了teacher动作”这一粗大工程解释，但没有排除表示的优化条件、指关节current-relative监督、额外旋转监督、标准差或闭环分布偏移的影响。完整teacher输入差异依然是独立重要变量。

已完成 Inspire抓取 Probe 的协议明确K=1，6组转换/时钟/输入独立审计均通过；其失败暂不能归因K=4/8冻结语义。见 [shared-native-grasp卡](../../src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-shared-native-grasp.md)。Shadow目前只有小面板位置抬升保持的正向信号，且与Inspire表面间隙/抬升判据不同；以上不是跨手因果结论或正式Validation。


## 后续数值核对

配对初始统计显示Inspire首拍XYZ target mean最大差2.33e-10、std差0；std均为
[.06822,.07376,.13319]m，见[实际归一化核对](../../outputs/ptv3-shared-wam-v2/inspire-action-representation-20261015-r1/first-XYZ-normalization-audit.json)。
因此若native包有收益，不能单独说是XYZ由q_PD换成归一化action。

还有split不匹配：现有Inspire三条均val，Shadow1k保持TRAIN3/8、DEV1/8；2k为TRAIN4/8、
DEV1/8。2k teacher保持也从7变9，不能把学生4→5解释为匹配训练增益。
独立raw核对与三条TRAIN候选见[split/面板审计](../../outputs/ptv3-shared-wam-v2/inspire-action-representation-20261015-r1/train-panel-audit.json)。
旧Inspire面板是sequence-heldout，Shadow DEV是object-heldout；不能把不同split混成跨手成功率。

同24拍/30Hz的运动基线也不支持直接用绝对forecast mm差推断表征好坏：Inspire DEV256
模型122.52mm、零位移279.39mm；Shadow common DEV64模型30.47mm、零位移63.38mm。
相对零位移为0.439与0.481，相对速度外推为0.374与0.357。目标幅度不同可解释绝对误差
的大部分差距；任务/面板仍未匹配。完整统计与复核范围见动作表示实验卡的CPU基线段落。
