# v4：τ条件/闭环首偏差与冻结重复（2026-10-22）

所属`wam-rl-policy-v3`；关联既有`P-20261022-single-tau-calibration`，本轮run为
`single-tau-repeat-20261022-r1`。S2b独立补完整终态参考，本轮不修改v6代码/脚手架。

## P0：保存trace合同（先执行）

区分参考时钟δ与live条件δ，及“高度保持失败”和“接触/耦合丢失”。使用旧7条真实trace，
不生成物理交互。统一Inspire30Hz，action[t]导致state[t+1]；tau[0]为第一未来。
首偏差阈值：canonical action/q/重算PD 1e-5（PD分平移m/转动rad）；dq1e-4；hand1e-5m；
force1e-3N；τ1e-6m。阈值只定位数值分歧，不是物理危险或因果判据。

初步实核：GTτ为ref[t+k]−ref[t]；WAM实际调用live H4+goal/event。R仍读live q/dq/force/
previous action，**不是整条执行器开环**。因此固定δ不能自动成为执行上界，但不能用此
构造差异直接解释排序。65/67帧首先失败的是升高≥5cm；64/66是最后满足帧，并非已证实
滑落。两帧仍有3个有效力接触、GT有效，故fallback不能单独解释更早的高度保持失败。
旧trace未保存实测PD/未裁剪actor输出，P0 PD仅独立重算；P1补齐这些缺项。

## P1预注册：Decision，先重复再解释

区分“旧四条执行链反常在固定条件下可重复”与“单次GPU物理/顺序差异”。结果决定是否
需要继续检查δ构造和闭环条件分布，不能产生WAM优于GT的泛化或统计结论。最便宜方法是
复用原权重、原81状态reference和原任务条件生成器，仅做5次重复，不再拟合。

- **n=5/链**：source-PD、原R+GT、校准R+GT、原R+WAM11000、校准R+WAM11000；25回合。
- 固定`s1_cylindersmall_pass_1`（已见/前轮微拟合序列），Inspire一手一物体，env数1、30Hz。
  R=Scheduled686 r4 step50000；校准R=旧run calibrated.pt（2000updates后），都只读。
- GT固定旧source-pd/trajectory.npz的81状态，3..72共70个有效control时刻；τworld native11，
  每步ref[t+k]−ref[t]，尾部无τ沿用原R零τ/validfalse。**不使用S2b新完整参考**混改重复条件。
- WAM固定step011000、旧stats/geometry、forecast_only/FP32；保留live H4+goal/event，
  固定flow_eval_seed、预测器eval模式；不更新权重，不重训v6。
- 隔离输出快照固定旧run基线1f41bcdb及当时source.diff；补其实际保存的native host、
  physical_safety、execution_contract。条件生成器是旧scheduler v2（为复现），**不回滚生产
  v3**。当前新增代码仅负责冻结/记录/相同δ计算；实际源码/补丁全部保存在run里。
- 共用native进程、生产continuous reset（旧运行同一机制）；每回合indexed reset，9个
  可见初态数组差≤1e-6，否则setup失败。记录episode与tick逐步断言；GPU隐藏warm-start
  仍不能自证相同。五条链按预先固定循环顺序轮换，每链在每个顺序位置出现一次。
- seed沿用native195/model196；这是同初态协议重复，不是5条独立任务或独立预训练种子。
- 窄任务沿用原定义：升高≥5cm、真实≥2指0.1N、运动耦合、未失握、离桌连续elapsed≥1s。
  单回合首次成功/原物理安全abort/300controls停止。TaskSpec精确8cm/完整RELEASE另列不认证。
- **硬预算25×300=7500envcontrols，wall1800s，产物≤2GiB，空闲磁盘≥20GiB**。GPU6仿真、
  GPU5推理，启动前实查；4留空。资源冲突/shape/时钟/PD合同错误停止run并保留断点，
  不把基础设施错误记方法失败。不占GPU0–3/7，不启动任何RL/学习更新。
- 补齐raw→clipped/canonical→expanded→actual PD target→q/dq，以及τ/valid/episode/
  force/真实时间。PD独立重算期望≤2e-6，>2e-5停止run。原先“0.001849”仅监督动作RMSE。
- 报每链成功n/5、最长持握的中位/范围、首个高度失效与接触/耦合失效、有效τ覆盖、
  clipping/安全退出；不以最长持握排序替代成功率。
- “排序描述可重复”预定义：原R WAM−GT持握中位>0且至少4/5重复差>0；校准交互分别
  看GT校准−原R、WAM校准−原R的中位/逐重复方向。即使满足也只说明该单轨迹协议现象，
  n=5不作WAM方法优越的统计证明。全零成功必须照报，不能靠0.9667s升级成功。

## P2：完整RELEASE0/6（诊断侧）

只读本线程`hermes-source-pd-20261022-r2`及完整原录制；两例安全中止与四例完整参考+
终态已观察必须分开。比较实际路径/末端误差、接触/释放/支撑/稳定秒数和缺失milestone，
给出必要条件清单，不凭0/6宣布判据物理不可达。不能把GT短参考问题移作source-PD缺观察。

## 复现命令与预期

```bash
# P0：81状态、四条GT构造合同0不一致；监督RMSE约.001848882；同保存obs动作回放误差<1e-5。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/audit_first_tau_loss.py \
  --run outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1 \
  --output outputs/wam-rl-policy-v3/tau-loop-independent-review/p0 --gpu 5
# P1/P2命令待脚本和实测完成后填充；此节的n/预算/条件在运行前冻结。
```

## P0实测（重复前完成）

存档`outputs/wam-rl-policy-v3/tau-loop-audit-20261022-r1/p0-final/`：7条trace时钟均对齐，
四条GT共280个实际有效controls、构造不一致**0**。NN用保存obs直接回放最大动作差
**6.56e-7**；校准同reference监督RMSE **0.0018488823**，基线 **0.1035400331**。

| 比较/事件 | 最早可观察帧 | 含义 |
|---|---|---|
| 校准GT两次 vs source：action/重算PD |0/0|τ尚无效时已有小动作差，不能把监督拟合当闭环无误差|
| 校准GT两次 vs source：q/dq/hand |1/1|action[0]后出现状态差|
| 校准GT两次 vs source：force |30/29|数值首差阈值1e-3N，不等于该刻滑落|
| 原R GT vs WAM：τ/action/重算PD |3|首次启用τ；q/hand首差4，force首差48，无一帧错配证据|
| 校准GT repeat0 vs repeat1：force/action |29|τ完全一致；有物理接触/反馈分歧，不把重复当独立样本|

校准GT最后满足窄条件为66/64；首个失败为**67/65（2.233/2.167s）**，唯一首个失败分项
均为`lift_below5cm`，仍3力接触、motion_coupled=true、grasp_lost=false；GT和上一control
GT都valid。相对reference手点均值漂移分别 **0.408842/0.410608m**；这让“GT一定是执行
上界”失去保证，但尚不识别排序原因。

首次接触/抓稳条件缺失100/104、首次grasp_lost标志104/110，均在GT末端fallback后。
因此外部“64/66是滑落/丢接触”的物理解读**不成立**；“更早高度保持失败在有效GT内”成立。
原WAM最长30状态=29/30=0.9667s仍失败；不是WAM优于GT统计证据。

代码证据：旧实际冻结`single-tau-calibration-20261022-r1/code/.../run_single_tau_calibration.py`
的`reference['hand'][t+arange(1,9)]-reference['hand'][t]`与`wam.predict(history,hands,goal,event)`；
`inspire_shadow.py::current_state`给R传live q/dq/physical/previous action，
`bc_mlp.py::forward`仍使用这些列。旧PD只有重算，无法自证当时每tick实测PD；重复会补齐。

```bash
# P1：fresh输出，n=5/链、25回合、最多7500controls，learning_updates必须0；先确认GPU5/6空闲。
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/repeat_frozen_tau_chains.py \
  --output outputs/wam-rl-policy-v3/single-tau-repeat-20261022-r1 --native-gpu 6 --model-gpu 5
```
