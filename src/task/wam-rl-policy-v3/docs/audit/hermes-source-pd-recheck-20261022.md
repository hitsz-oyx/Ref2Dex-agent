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
- 每例源参考结束后继续120个**真实**物理controls，保持最终记录PD目标；不是复制末状态、
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

# 193 passed/0 failed是本轮运行数；后续新增测试应报告新总数，不能沿用。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=tmp/hermes-v3-independent-review src/task/wam-rl-policy-v3/tests

# 期望4/4被断言拒绝；必须用新输出目录。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/check_hermes_fix_mutants.py \
  --output outputs/wam-rl-policy-v3/hermes-mutants-independent-review

# 真物理正控，需nvidia-smi确认GPU4空闲；最大4632envcontrols，不启动学习。
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/replay_source_pd_positive_control.py \
  --declarations outputs/wam-rl-policy-v3/hermes-prefix-fixed-20261022-r1/manifest.json \
  --output outputs/wam-rl-policy-v3/hermes-source-pd-20261022-r1 --gpu 4

# 纯保存状态评分；不重复花费物理预算。source-PD三种链身份绝不互借成功。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/score_source_pd_positive_control.py \
  --run outputs/wam-rl-policy-v3/hermes-source-pd-20261022-r1 \
  --output outputs/wam-rl-policy-v3/hermes-source-pd-20261022-r1/scoring
```
