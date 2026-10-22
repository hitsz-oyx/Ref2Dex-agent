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

## P1结果：完成25/25，0学习更新

代码提交`7dc82741`，实际使用run/runtime-code的旧版本隔离快照；没有回滚生产v3。
物理GPU6、模型GPU5（均RTX3090）；native Torch2.0.1+cu118、graspenv Torch2.4.1+cu121。
实际 **4933envcontrols /374.490s**，其中WAM推理169.567s；未花满7500预算、不追加训练。
9个可见初态字段对旧参考、对其余回合逐数组最大差0；不能自证GPU隐藏接触warm-start相同。

| 链（每链同一已见序列n=5） | 窄任务ever/final成功 | 最长持握中位数s | 范围s | actual安全abort | 实际controls | τ有效controls |
|---|---:|---:|---:|---:|---:|---:|
| source-PD |5/5|1.000|1.000–1.000|0/5|400|0|
| 原R+clock-GT |0/5|0|0–0|0/5|1500|350|
| 校准R+clock-GT |0/5|0.400|0.333–0.567|4/5|1139|350|
| 原R+live-WAM11000 |2/5|0.9667|0–1.000|2/5|931|916|
| 校准R+live-WAM11000 |0/5|0|0–0|5/5|963|948|

成功即停止，因此1.000s是协议截断下界，不是最长持握能力上限。四条R链clip controls分别
5/50/24/11；source0。安全abort均是原DOF限位代理；未放宽阈值、未改voxel网格、未因
encoding不可用直接判任务失败。新见证不是完整RELEASE或精确8cm终点成功率。

预注册持握方向规则得到**描述性重复**：原R WAM−GT正4/5、平1/5，中位+0.9667s；
GT校准−原R正5/5，中位+0.400s；WAM校准−原R负4/5、平1/5，中位−0.9667s。
**2/5 vs0/5不构成WAM优于GT的统计证据**，也不测样本效率。固定δ不具备同状态oracle
上界条件，不能从这组排序推导轨迹质量高低。它支持先查R的条件分布/闭环稳健性，而非
追加WAM预训练；尚未通过“相同live状态下不同δ构造”的因果干预证明机制。

独立保存状态重算**25/25 online verdict/duration一致**；动作→实际PD独立重算最大差
**3.1342e-7**，raw→clipped→expanded日志误差0，episode/clock错误0，GT合同错误0。
校准GT5次首个高度保持失败tick **64/62/62/60/68**，全在GT仍有效时，且首失败分项都是
lift_below5cm；首次grasp_lost标志 **100/99/88/93/107**，都在末端fallback之后。
原两个存档的64/66“失持”物理解读须按P0更正：最后通过帧不等于接触滑落。

复核入口：`outputs/wam-rl-policy-v3/single-tau-repeat-20261022-r1/`含25条原始trajectory、
raw/canonical/expanded/actualPD/episode、native_returncode0、源码快照、manifest；
`scoring-final/summary.json`逐链/逐回合，physical-rows.json逐帧分项。纯重算可重跑，不再
启动仿真。对私有副本故意注入PD错、episode错、packedτ错、GT值错，**4/4被拒**；
原真实trace均未改。import/执行崩溃不算抓住错误。

## P2结果：不是“2个限位+缺终态观察”能解释全部

固定新生产v3的六条source-PD（不是上表旧条件单序列重复）：0/6保留。两例中止为
cubemedium/cylindermedium；四例完整执行参考且真实后缀充足，不能再把它们全归于缺观察。

必要条件证明使用**当前固定桌面**反馈和完整TaskSpec：8mm平移、0.12rad旋转下任意CAD点
最多移动 `D=0.008+2*r_max*sin(0.12/2)`。支撑要求最低点gap∈[−0.02,+0.02]且中心在
桌板范围；忽略额外8mm平均点容差只扩大可行集合，所以扩大集合仍不相交是保守否定。

| sequence | 终点gap m | 最大点移动D m | 支撑兼容性 / 实际source结论 |
|---|---:|---:|---|
| airplane |−0.804783|0.018714|终点在桌外，允许gap区间与支撑区间不相交；当前完整TaskSpec不可达|
| cubemedium |−0.601447|0.016040|同样几何矛盾，另在106帧限位中止；不能双计两个独立病例|
| cylindersmall |+0.000087|0.011409|未被几何反证；最近终点位置差13.19mm，未达8mm；缺pending frame43|
| cubesmall |+0.000174|0.012020|终点位姿/释放/支撑/稳定连续7.167s，但缺pending frame75|
| cup |+0.000794|0.015818|同一物理末端条件连续6.733s，但缺pending frame140|
| cylindermedium |+0.000175|0.014818|未被几何反证；36帧限位中止、参考未完整执行|

计数：安全中止2/6、几何不相容2/6、交集1/6，另3例都无这两项否定。所有6条记录和0/6
描述性原结果保留，**没有删除两例后冒充新成功率**。几何证明仅否定“既在该终点附近又
有该桌面支撑”的生产判据，不证明物体不能落地/由其他表面支撑，也不证明其余4条可执行。
原始planned intent仍无独立认证；原录制末帧不是自动成立的任务指令。

cubesmall/cup说明：物理放置条件实际成立，但原完整任务还要求严格按序通过录制路径；
未认证的早期milestone不能由后来到位替代，因此full仍0。若科学指标只是抓举/放置，
以此全路径0否决物理能力是口径错配；若目标确实包含路径，当前source-PD尚未完成它。
本轮不静默删path gate改成2/6，也不把尚未反证的条件称为可达证明。

### 需要什么才可能通过（验收目标，不是已做到）

1. **可信任务规格**：独立确认目标/事件/支撑对象；两条当前几何矛盾显式记annotation_invalid，
   或在有意图证据时重新声明正确support surface，保留原始记录/差异。验收“不可达却仍标
   有效”2/6→0/6；不能只取消支撑、不把UNKNOWN改HOLD，不倒改旧分母/旧成绩。
2. **安全source控制**：保留实际DOF gate，解决两例q[16]低于−0.1rad的跟踪/求解器问题；
   目标安全abort2/6→0/6，实际PD对合同误差仍≤2e-5；不能放宽边界凑通过。
3. **明确路径与物理任务口径**：若要全路径，必须真实达到43/75/140等未决milestone并维持
   规定姿态误差；若只要抓举/放置，用新注册的goal/event任务协议评估，完整path另报。
   两个已有末端物理阳性不能被当作当前full-task资格，也不能被全路径失败抹掉。
4. **真实末端与执行资格**：在新的可信合同下真实观察连续1s；分别验证source-PD、GTτ→R、
   WAMτ→R。正式RL要求source/GT两链各≥95%；当前source窄5/5不能替代full或GT0/5。
5. **同状态GT条件合同**：S2b补完整未来必要但不充分；必须核参考相位/live锚点。
   `h_live+δ_ref = h_ref_future + (h_live−h_ref_now)`，旧两次偏移约0.409m；锚定修复的
   构造目标是这项绝对端点偏移→0，并另验物理可执行性。完整参考不能单独解释/修复
   发生在有效GT内的首个高度下降。暂不把简单重锚定自动称为GT物理上界。

## diff / 独立复跑入口

本轮基线418e4c0a；`git diff 418e4c0a HEAD -- src/task/wam-rl-policy-v3`。
核心：`tau_replay_contract.py:6`（第一未来/参考时钟）、`:21`（action→PD）；
`audit_first_tau_loss.py::physical_rows`（逐分项区分高度与接触）；
`repeat_frozen_tau_chains.py::freeze/main`（隔离旧条件/预注册25回合/实际PD日志）；
`terminal_support_contract.py:8`（必要条件上界）、`audit_release_reachability.py::main`（六例）；
`summarize_frozen_tau_repeats.py::main`（独立物理重算+clock/episode/packing/PD验收）。
旧真实构造证据：冻结`run_single_tau_calibration.py:107/110/144`分别为GT、WAM与监督GT；
`inspire_shadow.py:97`给R传live state；`bc_mlp.py:28`读取这些观测，不是无反馈轨迹播放器。

```bash
# 纯状态重算，期望25/25一致；5/5、0/5、0/5、2/5、0/5（按summary的链名核对）。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/summarize_frozen_tau_repeats.py \
  --run outputs/wam-rl-policy-v3/single-tau-repeat-20261022-r1 \
  --output outputs/wam-rl-policy-v3/tau-loop-independent-review/repeat-scoring

# P2纯CPU，期望2几何不相容、2安全中止、交集1；原始6条均保留。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/audit_release_reachability.py \
  --run outputs/wam-rl-policy-v3/hermes-source-pd-20261022-r2 \
  --output outputs/wam-rl-policy-v3/tau-loop-independent-review/p2

# 四类日志故障，期望scorer退出1/FAIL，外层验收PASS caught4/4；不做物理交互。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/check_tau_repeat_log_faults.py \
  --run outputs/wam-rl-policy-v3/single-tau-repeat-20261022-r1 \
  --output outputs/wam-rl-policy-v3/tau-loop-independent-review/log-faults

# 最终v3全目录：本轮201 passed/0 failed/0 skipped；后续新增测试必须报新总数。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=tmp/tau-loop-independent-tests src/task/wam-rl-policy-v3/tests
# 9个故意源码错误（含错一帧、错时钟、PD coupling、错误支撑可达断言），期望9/9被断言拒绝。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/check_hermes_fix_mutants.py \
  --output outputs/wam-rl-policy-v3/tau-loop-independent-review/source-mutants
```

所有输出路径必须新建；输入依赖同服务器已有旧run、checkpoint、geometry/cache和native
运行环境，不声称空数据仓库可生成这些录制。原输出/权重不覆盖。最初P0独立单测曾有
1 import失败（trajectory_policy路径未显式加入），已修正确导入；不把该错误算mutant kill。
最终201 passed/0 failed/0 skipped，17.98s；9源码mutants和4日志故障均明确被拒。
外部三方尚未复跑本轮diff；待裁决是原始任务意图/支持对象、完整路径是否属于科学成功
指标、GPU隐藏接触状态的重复性。因果机制未由本轮单轨迹排序证明。P1三臂RL仍HOLD，
不重训v6，不追加未注册的R拟合或重复数；S2b完整参考应另起明确身份的下一run。

精确定位补充（均可用`nl -ba`复核）：

- `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/tau_replay_contract.py:6`、`:21`。
- `src/task/wam-rl-policy-v3/tools/audit/audit_first_tau_loss.py:14`。
- `src/task/wam-rl-policy-v3/tools/audit/repeat_frozen_tau_chains.py:12`、`:36`。
- `src/task/wam-rl-policy-v3/tools/audit/summarize_frozen_tau_repeats.py:13`。
- `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/terminal_support_contract.py:8`。
- `src/task/wam-rl-policy-v3/tools/audit/audit_release_reachability.py:22`。
- `outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1/code/src/task/ptv3-shared-wam-v6/tools/audit/run_single_tau_calibration.py:107`、`:110`、`:144`。

最终仓库门禁`python3 tools/verify.py --changed`为VERIFY PASS，日志
`outputs/wam-rl-policy-v3/tau-loop-audit-20261022-r1/verify-final.log`；只stage本线程v3范围。
GPU5/6已释放；原数据/权重保持原位，`docs/PROJECT_MEMORY.md`未提交。
