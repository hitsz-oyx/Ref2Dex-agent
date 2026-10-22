# Hermes复核差异与source-PD生产正控制（2026-10-22）

范围：只修改`wam-rl-policy-v3/**`；v6接线/校准脚手架交S2，根STATE/历史卡交S3。
输出在`outputs/wam-rl-policy-v3/`。共同fix-review只在结算时追加本报告链接与待裁决摘要。

## 先验设计：Blocker

区分“生产reset修正后，合法source-PD确能完成已声明任务/抬升持握”与“该物理链仍不具备阳性”。
结果决定是否可交付同τ executor校准输入；source-PD不代替GTτ→R资格。最便宜的方法是复用
原完整记录的PD targets、6条既有固定面板和生产RPC host，不加载/训练任何executor/WAM。

- 固定旧6条面板全部保留，不按新结果选样：airplane、cubemedium、cubesmall、cup、
  cylindermedium、cylindersmall各1条。完整task取原raw-task，不重标HOLD/RELEASE。
- 从原501对应source读取**全部**PD targets；airplane为652controls，不用前轮600截断版本。
- Native production reset，continuous腕角等价规范；inverse用live q及最近2π代表，真实目标
  必须可由12个主动action及原被动指耦合表达。实质超范围>1e-6明确拒绝，不以clip执行。
  记录边界舍入≤1e-6；生产原target和实际PD最大误差目标≤2e-5。预期饱和执行0。
- 每例源参考结束后继续至少120个（共同batch时域，短序列更长）**真实**物理controls，保持最终记录PD目标；不是复制末状态、
  延长教师真值或追加原录制标签。参考最长652，总772×6=4632envcontrols，≤6000。
- GPU只用空闲4/5/6一张；native Torch2.0.1+cu118，client graspenv；seed195沿用旧配对
  正控。wall≤600s、产物≤2GiB、磁盘余量≥20GiB；未重置任何他人作业，不做学习更新。
- 原CAD/table及actual limited-DOF安全代理不放宽；违规或不可表达command标该例失败，
  停止该例source指令；其余共同仿真继续。inactive开手中性控制不计作source-PD成功证据。
- 同保存trace比较旧v2/新v3路径合同。预期接触单帧空档撤回几何pointer由2→0改为2→2；
  无接触仍不得认证grasp/task，空档仍不得补造跨越。Task schema升v3，旧RL不可静默续训。
- 分开报告完整TaskSpec、实际抬升≥5cm+真实force/coupling/离桌连续**elapsed≥1s**。
  目标至少一个物理阳性；正式完整任务资格仍需同协议source-PD与GTτ→R各≥95%，不能
  因窄任务阳性放行。若失败则保留具体链/动作/状态，不把失败泛化为v6方法否决。
- 不再运行上一轮已用完的2000-update微拟合；本轮只准备完整同τ参考/state-action输入，
  ≤2000updates/30min校准的实际脚手架/是否复跑由S2接手，避免双重花费。

## 复现差异（已定位）

迁移默认旧manifest可读。第109行比较失败不是源语义变化：旧raw JSON缺新增
`annotation_invalid=false`及`annotation_invalid_reasons=[]`，其余字段逐项一致。
解析器补默认字段后不应与旧字典比较。本轮比较解析后语义，并字节复制完整raw作为task；
原始raw/admission/source不修改，缺省字段不反写。CLI从fresh output执行，不幂等；重跑同
output必须显式拒绝。`--refresh-audit`仅刷新已有完整audit汇总，不修补半成品目录。

`88 passed in17.15s`来自`fix-review-20261022-r1/final-tests.log`及旧提交1f41bcdb。
该提交的`test_ppo_model_identity.py`没有Hermes点名的actual predict新测试；后续新增这条
测试后旧清单总数增长，不能沿用“88全过”的预期。Hermes的89项/1失败属于新增测试后的
合同差异，不能被旧88项日志覆盖。S2负责KeyError；本线程不修改v6实现或fixture。

## 已执行工程验收（先于正控制）

- `v3-full-tests.log`：**193 passed / 0 failed / 0 skipped，17.80s**，完整v3 tests目录。
- migration CLI原bug实际红：`prefix-red.log`及`prefix-test-red.log`均在109行复现失败；
  新schema默认字段差异已逐键核实。fresh目录全501完成5.53s，task/raw/admission三个字节
  保留检查均501/501；`byte-preservation.json`。同output重跑拒绝由回归实际验证。
- pointer红：`path-red.log`，1 failed/1 passed；单帧gap观察指针2→0。修后2→2，
  `phase=ACQUIRE`、`task_success=false`、未确认阶段进度0，抓稳后才提交阶段进度。
- 源码mutants第一版3/4被拒，`gap-supplies-crossing`存活：测试选的跳跃位置未跨未决里程碑。
  改成空档前后确实跨越未决milestone的负控制，最终`mutants-reviewed/summary.json`
  **4/4被断言拒绝，0存活、0收集错误**。保留第一版日志，不拿import错误当kill。
- 旧“88 passed预期”的命令未随新增测试更新，是本线程复核文档错误；旧88项成功日志
  不能回答新增测试是否成功。以各运行明确commit+文件清单+全量passed/failed为准。

从仓库根目录执行，原始输入必须已存在：旧审计manifest及其每条sidecar的raw-task、task、
admission.npz；manifest中的绝对`source`目录必须可读（trajectory.npz及其原记录）。这是
**从已有原始产物、干净新输出目录**重建，不是从空数据仓库凭空生成机器人录制。

```bash
# fresh输出；期望501完整声明，490 RELEASE /11 HOLD /0 UNKNOWN；禁止重复使用同一输出。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/repair_prefix_task_semantics.py \
  --audit-manifest outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json \
  --output outputs/wam-rl-policy-v3/hermes-prefix-independent-review

# 最终194 passed/0 failed/0 skipped；后续新增测试必须报告新总数。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=tmp/hermes-v3-independent-review src/task/wam-rl-policy-v3/tests

# 期望5/5被断言拒绝；必须用新输出目录。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/check_hermes_fix_mutants.py \
  --output outputs/wam-rl-policy-v3/hermes-mutants-independent-review

# 真物理正控，需nvidia-smi确认GPU4空闲；最大4632envcontrols，不启动学习。
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/replay_source_pd_positive_control.py \
  --declarations outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json \
  --output outputs/wam-rl-policy-v3/hermes-source-pd-independent-review --gpu 4

# 纯保存状态评分；不重复花费物理预算。source-PD三种链身份绝不互借成功。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/score_source_pd_positive_control.py \
  --run outputs/wam-rl-policy-v3/hermes-source-pd-independent-review \
  --output outputs/wam-rl-policy-v3/hermes-source-pd-independent-review/scoring
```

正控输入补充：固定旧6条面板全部为原`val`，不在迁移的TRAIN501中。r1因此在native启动前
KeyError中断，0envcontrols；`failure.json`与client log保留。r2保留六条原面板，从原660
manifest读取完整raw-task，绝不用其旧截断task.json。没有换选TRAIN或静默删病例。
其中cylindersmall在前轮已微拟合，是已见样本；这批只能作已知序列诊断正控，不作未见DEV
方法评估。原报告501指迁移cohort，不能套成此6例的采样身份。

## 生产source-PD结果（r2，代码3c5f54b2，0学习更新）

实际GPU4，54.985s，6×772=**4632envcontrols**；native正常退出0，GPU4/5/6均释放至6MiB。
原660完整raw-task、源PD目标与canonical mesh只读；六条原val面板全部保留。动作实质超界
拒绝0、实质clipping0，只有source边界舍入（最大2.384e-7），实际PD最大误差2.424e-7。
q初态相对原录制按连续轴mod2π最大误差7.153e-7，dq0，hand1.788e-7；不是逐位初态相同。

| sequence | 完整source controls/是否执行完 | 原完整RELEASE成功：v2→v3 | ≥5cm/force耦合/离桌最长elapsed | 物理/执行中止 |
|---|---:|---:|---:|---|
| s1_cylindersmall_pass_1 |166/是|0→0|1.9667s|无|
| s2_cubesmall_pass_1 |232/是|0→0|3.0000s|无|
| s2_cup_pass_1 |206/是|0→0|2.5333s|无|
| s7_airplane_lift_Retake |652/是|0→0|3.9333s|无|
| s7_cubemedium_pass_1 |193/否，106步中止|0→0|1.5667s（中止前）|q[16]=-0.109389 <lower0−0.1|
| s9_cylindermedium_inspect_1 |165/否，36步中止|0→0|0s|q[16]=-0.120949 <lower0−0.1|

全6分母不变：**窄任务ever阳性5/6；完整任务ever/final均0/6**，不宣布完整教师资格通过。
4例无安全中止、执行了完整source与至少120个真实终态后缀；两例中止后的惯性状态不作
source-PD成功证据。所有case都观测至772tick；短序列后缀最多607controls，是实际积分，
不是末帧复制。5/6是固定已知面板的阳性见证比例，不是泛化成功率或完整任务成功率。

同一保存trace比较v2/v3：已观测pointer撤回次数**2→0**；cubesmall确认进度最大0→0.260006，
其余5例不变，full成功0→0。不能把这项工程修复说成成功率提升。旧完整raw intent未独立
认证，源PD按绝对关节目标开环跟随也不保证逐帧复现原物体SE(3)；因此此6例仍不能给完整
RELEASE评分/可执行性或GTτ→R发资格。物理限位是当前代理：两个负q有实际dq/PD反向偏差，
但solver瞬时过冲与真实危险程度没有独立认证，阈值未因本次失败放宽。

S2交接：`hermes-source-pd-20261022-r2/scoring-final/calibration-inputs.json`指向完整
trajectory.npz与controls.npz。固定cylindersmall无安全中止、源参考166controls全执行；
真实窄持握区间tick51..110（59/30=1.9667s），首个elapsed1s在tick81。可用完整保存未来，
不会再因首个阳性即截断source导致尾8帧GT缺失。所有772个control的state/action均保留，
建议仅按预注册窄任务/相同τ通道构造监督；**tau[t,0]是t+1，不是当前帧**；t≥3且t+8在
保存状态范围内才valid。所有进入状态必须因果使用，部署不得读取实测未来。
本线程训练0updates；前轮2000updates已耗尽，新校准由S2按授权≤2000updates/30min执行，
不可混成未见DEV、不可借source-PD阳性宣布GTτ→R成功。

## 最终复核与待裁决

- 新增秒制时长负控制后，全v3目录最终 **194 passed /0 failed /0 skipped，17.55s**；
  源码mutants最终 **5 killed /0 survived /0 collection errors**（mutants-final/summary.json）。
- `scoring-final/scores.json`含完整任务ever/final；`task-traces.json`逐帧old/new/反馈；
  `calibration-inputs.json`仅准备数据，不运行训练。旧scoring/保留首轮结果，最终复现建议用
  新输出目录`scoring-review`，当前命令行为可重复重算，不追加物理预算。
- 差异解释：已核88日志对应旧文件版，旧1f41bcdb/新8fe1a29f同一文档实际各列19个文件，
  新版多了actual-v6-predict用例。Hermes称18文件的确切argv/working-tree快照本线程没有，
  **无法自证其失败时使用哪份中间fixture**；不归咎环境、不撤销Hermes的失败。S2负责
  KeyError闭环。旧命令88期望过期是本线程文档错误。
- 待Hermes裁决：接触gap“几何保留/阶段未认证/无跨gap插值”合同是否接受；source-PD
  5/6窄阳性足以交S2做单轨迹上限，但完整RELEASE0/6不能放行正式RL；两条q限位代理
  的物理含义仍需独立复核。外部审计未复跑本次新diff，不宣称三方一致。

本轮diff入口：`git diff 8fe1a29f HEAD -- src/task/wam-rl-policy-v3`；工程提交3c5f54b2。
主要位置：`tools/audit/repair_prefix_task_semantics.py:110`（解析语义比较）与`:114`（原raw
字节保持）；`src/wam_rl_policy_v3/task_scheduler.py:177`（不撤回几何）、`:183`（跨gap无
插值）；`src/wam_rl_policy_v3/source_pd.py:6`（生产PD逆映射/超界拒绝）；
`tools/audit/score_source_pd_positive_control.py:9`（真实elapsed秒计时）。

动作边界补充：`action-boundary-audit.json`核到active source3230envcontrols；其余1402为
中止后inactive中性控制的物理积分，全部计入4632总预算。source指令中有2个合法手指端点
命令达到±1，不等同inverse clipping；Cartesian/腕角到边界0，最大abs0.964920。因此
“clipping0”不能读成“所有源命令从未取端点”，实质超界指令始终拒绝。

S2执行身份提醒（未改其范围）：共享TaskScheduler已升`preserve-observed-geometry.v3`；
官方v3 PPO动态绑定该schema，但v6三臂resume identity尚只绑定其本地TASK_CONTRACT等，
需要S2把scheduler_schema纳入身份/升版，避免旧奖励/截断训练恢复到新语义。当前P1 HOLD、
无可放行物理资格，因此本轮没有利用该缺口启动/续训。评价性旧checkpoint可按新协议另起
独立run，但不能称原训练已被事后修复。
