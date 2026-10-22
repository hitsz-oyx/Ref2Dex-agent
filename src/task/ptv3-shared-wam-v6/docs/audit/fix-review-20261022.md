# 修复共同复核包（2026-10-22）

## Hermes14:1x复核回填（仅v3结算；v6实现由S2负责）

本段按机主点名要求回填；本线程没有修改S2的v6实现/fixture，也不改S3的STATE。
完整diff定位、fresh命令、前置输入、红绿回归和六例逐项结果见
[v3独立复核与source-PD报告](../../../wam-rl-policy-v3/docs/audit/hermes-source-pd-recheck-20261022.md)。

- **88的差异已解释，旧预期撤回**：旧日志确为88 passed/0 failed，但对应未加入actual predict
  新测试的文件版。新增用例后的Hermes 88 passed/1 failed不能被旧日志覆盖；旧复现预期
  没更新是本线程文档错误。其失败时确切working-tree快照/argv未取得，不归咎环境；S2
  负责KeyError闭环。本轮仅v3全目录最终**194 passed/0 failed/0 skipped**，不包含v6用例。
- **迁移可独立复现**：旧JSON无新增annotation_invalid两个默认字段，原有字段相同；
  `repair_prefix_task_semantics.py:110`比较解析语义、`:114`字节复制原raw。fresh完整迁移
  **501/501**；task/raw/admission字节保持均501/501。使用原660审计manifest为输入，
  fresh输出；不幂等、不覆盖旧产物，已有目录明确拒绝。参数与输入根在链接报告。
- **接触gap合同**：`task_scheduler.py:177`保留已观测几何，`:183`禁止跨gap插值；
  `source_pd.py:6`按live q反解最近连续角目标、拒绝实质饱和。故意源码错误最终
  **5 killed/0 survived/0 collection errors**；第一版存活的gap负控制及后续收紧日志保留。
- **生产source-PD已跑**：6条旧固定val面板，4632envcontrols/54.985s；实质clipping0、
  饱和拒绝0，PD最大误差2.424e-7。原完整RELEASE old/new均**0/6**、final0/6；窄任务
  ≥5cm+真实force耦合+离桌连续elapsed1s的ever见证**5/6**。两例actual q限位代理中止，
  不放宽阈值；另4例完整source与真实终态后缀观察。已观测pointer撤回次数2→0。
- **交接S2、不重复训练**：完整原始trace/control及`scoring-final/calibration-inputs.json`
  在`outputs/wam-rl-policy-v3/hermes-source-pd-20261022-r2/`。固定已见cylindersmall
  无中止，持握tick51..110=1.9667s，首个1s阳性tick81；完整未来保留，准备同τ校准。
  本轮0学习更新，GTτ/WAMτ executor不继承source-PD阳性；完整RL资格仍未通过。

**待裁决**：Hermes是否接受“几何保留、阶段不认证、跨gap不补轨迹”；两条q限位代理的
物理危险含义；原完整RELEASE声明/开环PD重现协议仍不具备正控。外部审计尚未复跑本次
新diff，不宣称三方一致。S2还需将共享scheduler v3纳入三臂resume身份；当前仍HOLD、无续训。r1因TRAIN501不包含该旧val面板在native前失败、0物理步，已保留；
r2读取同六条的原完整raw，未换样本。下方13:0x/12:4x为旧时点，旧88/64数字不作当前预期。

## 13:0x v2：本轮收口与复核入口

本节比较基线`1f41bcdbd613335b11d7accca9b2ec1035ec6e24`，覆盖下方前轮待推进状态。
机主v2(A–G)已逐项处理，窄路径的正负结果均保留：
[一页结算](negative-result-audit-one-page-20261022.md)。**审计收口，正式RL保持HOLD；
source-PD阳性不能替代GT-τ executor资格，未启动v6大训练。**

| 修复/核实 | 当前diff定位（仓库相对路径） | 修完期望与实测 |
|---|---|---|
| A 完整任务/无效标注 | `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/task_spec.py:49`、`task_annotations.py:92`、`task_admission.py:31` | 完整录制重建501/501一致，0/prefix造成新终态翻转；旧64 UNKNOWN逐条恢复62R/2H。缺原始force或validity字段/未定义终态显式annotation_invalid并拒绝完整评分，不改HOLD/RELEASE |
| A raw invalid与dropout | `src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/data.py:64` | raw-invalid/dropout原先只有合并valid，现两个独立boolean；原始无效、主动drop、同时发生三种状态可区分，完整TaskSpec不依赖它们。输入条件mask数值保持原算法，未据此宣称v6必须重训 |
| B 复制到三臂的voxel误终止 | `src/task/ptv3-shared-wam-v6/tools/run/samples_efficiency_env.py:118`、`:131`、`:198`、`:209` | 注入编码故障额外终止1→0，τ零/valid=false；物理穿透仍终止；同tick成功不被encoder失败否决。CPU真实Env边界与实际三臂native60controls均通过 |
| B 三信号/可恢复编码失败 | 同文件`:159`、`:225`；`src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/online_runtime.py:14` | info独立提供encoder_available/physical_safety_abort/task_success、ever/final；模型域/非有限预测用EncodingUnavailable降级，不结束物理回合。注入异常修前抛出、修后继续；schema/shape错误、OOM/坏CUDA上下文仍为run失败，不能假写物理失败 |
| B 真物理约束 | `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/physical_safety.py:9`；两个native host导出actual joint bounds | 保留原voxel网格；另查真实CAD/table穿透>20mm与实际limited-joint越界（平移>100mm/转动>100mrad），强制安全abort。两种encoder状态下穿透负控制都拒绝。它们是工程安全代理，未证明覆盖所有手物碰撞 |
| C1 adapter已存在 | `src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/online_runtime.py:103`、`:111`、`:136`；`:53`注入event | **确认外部审计**：已有v6独立adapter及三臂C路；“未接线”仅指官方V4 PPO。真实predict接口验证21/11、左右slot/anchor/world/event/goal；真实C路WAM覆盖13/20controls。没有硬造新桥 |
| C2 synthetic任务来源 | `samples_efficiency_env.py:108`、`:212`；`src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/execution_contract.py:14` | Source TaskSpec保留，episode另建synthetic +8cm位置HOLD1s；source_full_task_success=null，不能混成501完整任务率。新run绑定task/encoder/fallback/safety/reset身份；旧训练不按新协议续跑 |
| 教师资格硬门 | `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/executor_qualification.py:8`；`train_samples_efficiency.py:35`；`inspire_measured_official_ppo.py:38` | 正式RL必须匹配任务/executor/执行策略/序列范围的物理receipt；source-PD与GTτ→R各≥95%，WAM结果单列。实际本轮两份receipt均因GTτ→R 0/2明确拒绝；单序列证据不能放行未知初态cohort |

### 固定无τ后备模式及覆盖率

选择**同executor、只用当前已观测状态，τ=0且tau_valid=false**。这个输入规则在运行前固定；
保持同一actor的行为分布，不在env.step里替换已经被PPO计算logprob的action。没有把未训练
的fallback称为稳定控制器；它也必须受物理安全和资格门约束。A本来没有τ，B/C在不可用tick
按同规则提供零τ。覆盖率分母是实际执行controls的输入状态，不把未执行的最后一次预测混入。
实际三臂各20controls：不可用7（启动6+故意outage1），额外encoder done0；C有效WAM13/20。
65%是工程注入场景的覆盖率，不能报告成自然部署覆盖率或任务成功率。

可恢复错误限定为模型域/非有限预测；输入合同错、missing checkpoint、OOM等显式停止run，
不静默丢掉任务，也不把基础设施失败升级成方法负证据。错误摘要最多8条、每条512字节。

### E 单轨迹校准：预算已用尽，分别报告三链

固定`cylindersmall/s1_cylindersmall_pass_1`，本轮起该序列为已见训练序列。
预注册新窄任务是升高≥5cm、两指真实force/耦合、离桌且连续elapsed≥1s；它不同于三臂
精确8cm终点与完整RELEASE。9个共同初态字段逐数组差0；同轨迹重复不作独立样本。

| 执行链 | 窄任务成功/复跑 | 最长连续持握 |
|---|---:|---:|
| source-PD（canonical clipping0） |1/1|1.000s|
| GTτ→原R |0/2|0 /0s|
| GTτ→同架构校准R |0/2|0.633 /0.467s|
| WAMτ11000→原R |0/1|0.9667s，仍失败|
| WAMτ11000→校准R |0/1|0s|

80条新source-PD真实状态动作，Adam1e-4/batch64/seed196；**2000updates、5.3758s**，
128000次监督样本呈现不是环境交互。动作RMSE0.103540→0.001849，下降98.21%；不能当物理
资格PASS。基线权重不改，校准只更新同Scheduled686 actor，shape encoder冻结，无DAgger。
总校准1488物理controls+三臂接口两轮120=1608，所有自有native退出，GPU4/5/6核查各6MiB。

独立重算不用在线LiftHoldWitness，直接从保存state重建force/support/coupling及整数连续区间：
7条trace verdict全部一致；coupling仍复用FeedbackObserver，所以不是第二种接触探测真值。
原R-WAM最长只有30状态=29/30s，拒绝。校准GT两次于66/64帧已失持，当时GT仍有效；
参考共81状态，只覆盖70个有效GTτcontrols，之后fallback限制后段归因。一次GT校准和两条
WAM链触发真实joint16或joint7限位代理，具体轴/q/上下界保存。没有为凑成功改阈值。

### 可复跑命令与期望输出

```bash
# 当前diff；复核提交后HEAD指交付版本。新文件也包含在commit中。
git diff 1f41bcdbd613335b11d7accca9b2ec1035ec6e24 HEAD -- src/task/ptv3-shared-wam-v6 src/task/wam-rl-policy-v3

# 501完整实录逐条重建；期望PASS、501、64→62R/2H、raw-invalid0、意图冲突11。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/audit_annotation_provenance.py \
  --output outputs/ptv3-shared-wam-v6/negative-audit-v2-review/annotation-provenance.json

# 实际三臂+一次编码outage工程smoke；需先看GPU4/5仍空闲。期望PASS、60controls、每臂unavailable7/20、C covered13/20、额外done0。
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/smoke_samples_signals_native.py \
  --output outputs/ptv3-shared-wam-v6/samples-signals-native-20261022-review

# 已完成的单轨迹实验重算，不开新训练/仿真；期望PASS、source1/1、GT原/校准均0/2，WAM均0/1。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/summarize_single_tau_calibration.py \
  --run outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1

# 最终定向回归；期望64 passed（实测18.86s，1个PyTorch弃用警告）。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=src/task/wam-rl-policy-v3/src:src/task/trajectory-policy/src:src/task/ptv3-shared-wam-v6/src \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=tmp/audit-v2-external-pytest \
  src/task/wam-rl-policy-v3/tests/test_task_annotation_validity.py \
  src/task/wam-rl-policy-v3/tests/test_executor_qualification.py \
  src/task/wam-rl-policy-v3/tests/test_task_annotations.py \
  src/task/wam-rl-policy-v3/tests/test_prefix_task_semantics.py \
  src/task/wam-rl-policy-v3/tests/test_terminal_release_timing.py \
  src/task/wam-rl-policy-v3/tests/test_task_scheduler.py \
  src/task/wam-rl-policy-v3/tests/test_executable_task_admission.py \
  src/task/wam-rl-policy-v3/tests/test_model_domain_admission.py \
  src/task/wam-rl-policy-v3/tests/test_continuous_wrist.py \
  src/task/wam-rl-policy-v3/tests/test_measured_episode_logging.py \
  src/task/ptv3-shared-wam-v6/tests/test_ppo_model_identity.py \
  src/task/ptv3-shared-wam-v6/tests/test_v6_training.py \
  src/task/ptv3-shared-wam-v6/tests/test_samples_execution_signals.py

# 11个共享源码回归+10个三臂/adapter/时长回归；每个都必须由断言失败拒绝，import错误不计。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/run_fix_mutants.py \
  --output outputs/ptv3-shared-wam-v6/audit-v2-shared-mutants-review
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/run_signal_mutants.py \
  --output outputs/ptv3-shared-wam-v6/audit-v2-signal-mutants-review
```

单轨迹原始运行命令为`run_single_tau_calibration.py --output ... --updates 2000 --train-seconds 1800`；
完整入口、环境与预算在[P单轨迹卡](../experiments/probes/P-20261022-single-tau-calibration.md)和
run manifest，主机native Torch2.0.1+cu118、策略/拟合Torch2.4.1+cu121。**本轮2000updates
已经用完，不自动再执行这个训练命令**；外部复核应先使用保存权重/trace/源码冻结和上面纯重算。

产物根`outputs/ptv3-shared-wam-v6/negative-audit-v2-20261022-r1/`：73项首轮回归、最终64项定向复核（含新增缺validity及资格范围），
annotation-provenance-final.json逐501/旧64说明；final-reviewed-tests-2.log记录最终64项；shared-mutants-final/（11/11）、signal-mutants-reviewed/（10/10）、测试日志。
最初三臂native r1因harness给错输出namespace而被拒，0物理步；native r2改用入口要求的
Task输出目录，完成60controls；最终r3运行前冻结v3/v6源码与manifest，再次60controls通过。
r1错误/超时与r2/r3结果均保留，没有静默换协议；r3 native_returncode=0。

最终已stage本轮33个文件后运行`python3 tools/verify.py --changed`：**VERIFY PASS**，
日志`negative-audit-v2-20261022-r1/verify-staged.log`。这覆盖当前实验卡/schema/seed/链接及
仓库自动选择的合同检查；下方旧metadata阻断描述仅属于前轮历史。未提交`docs/PROJECT_MEMORY.md`。

### 三方待裁决与当前证据边界

1. raw-invalid0/501只指原始force/tick有效与完整重建一致；11条rawH/referenceR、29条疑似
   release-loss仍无独立计划意图认证。不能以0 raw-invalid宣称501份真实完整任务GT；正式完整
   任务正控认证分母仍0。所有数据保留，未把UNKNOWN强改HOLD或依据proxy删样本。
2. 同τ校准未得到阳性。可见初态相同而物理后缀略异、短GT参考和state-only fallback都限制
   因果归因；当前只能阻止扩大正式RL，不能否决τ表示或预训练路线。
3. 新安全规则是有明确单位的工程代理，不是完整碰撞认证：limited-joint余量平移100mm/转动100mrad、CAD/table
   20mm与force/geometry误差需Hermes/外部审计复核；不因本次0.9667s失败放宽它们。手物
   pairwise penetration和GPU隐藏warm-start未自证。
4. 第一版adapter接口测试曾让一个错误21→11 mutant存活（9/10）。已补独立的21槽valid mask
   与坐标断言，不能只拿由同一坏映射生成的输出自证；保留survivor日志，最终结果另列。
5. Hermes/外部审计尚未给这次diff的复跑结论。任何分歧保留双方版本、命令、指标和原始trace，
   不宣称三方已经一致。候选几何进度回退、完整RELEASE及独立样本效率转future evidence；
   按机主F本轮收口，不再无限扩大审计。

若重训v6，最小清单仍是**数据/标注**分离完整TaskSpec、训练可用mask及raw-invalid/dropout，
重建受旧task10/τ条件污染的cache；**条件**对齐spatial waypoint与t+24 endpoint的语义，保留
已有21/11 adapter与观测mask；**预算**由Hermes另行批准，先固定DEV与短资格预算再定主预算。
这次robot后处理bug不自动证明全部v6预训练数据有错；当前GTτ→R未资格，追加v6预训练
无法替代该前置条件。本轮只微拟合executor，v6权重/原flowmatching进程均未改。

## 前轮12:4x修复记录（保留原时点范围）

本轮以修好实际合同为目标。比较基线为已推送的
`60efadce2444709bbbf9802107de7bc5e3f9eef6`；历史训练冻结代码为`3a0779f6`。
数据、旧模型、旧结果均保留。下面区分修复后的实测变化、必须达到的验收目标、仍未通过的资格。
Hermes/外部审计者尚未独立复跑；没有把单方测试通过记为三方一致。

## 逐项改动、预期数值与实际证据

| 项目 | 可复核实现位置 | 修完期望哪个数怎样变 | 本轮实测及边界 |
|---|---|---|---|
| 安全prefix改写任务 | `wam-rl-policy-v3/src/wam_rl_policy_v3/task_annotations.py:17`；`tools/audit/audit_teacher_rollout_corpus.py:118`；`teacher_dataset.py:31` | 与完整声明不一致473/501→0/501；任务数501、可用控制56418都不变；恢复490R/11H/0U | 501/501新task packet等于原raw-task，473次模式恢复；1221事件、81794有序source_frames保留，491删失、709个超prefix事件保留。完整成功自动认证仍0/501 |
| UNKNOWN和旧坏sidecar准入 | `task_admission.py:13`；`task_spec.py:50`；`task_scheduler.py:57` | 真UNKNOWN进入可执行任务0；旧截断task显式setup失败；修复完整声明后501→501，不能用437筛选制造收益 | 64条旧UNKNOWN按其原声明恢复62R/2H，全部留在501中；新完整cohort准入501/501、原初态voxel通过501/501，旧manifest明确拒绝。保留对真UNKNOWN的拒绝及序列清单。新policy v2绑定resume身份 |
| 释放时序与剩余路径 | `task_scheduler.py:60`、`:108`、`:124`、`:198` | 两条已定位真实记录由0/2成功→2/2；释放授权分别应在171/233帧附近，不额外要求释放前0.2s持握 | 两例实际2/2；501全量同轨迹复评分full 0→77成功（15.37%），safe prefix 0→9（1.80%）。full授权RELEASE 1→205、pathcomplete 1→204；没有新执行或阈值放宽 |
| voxel域与物理任务终止分离 | `model_domain.py:7`；`measured_official_env.py:90`、`:114`、`:168`；runner身份字段 | 注入一次在线越界：额外task done1→0，tau_valid1→0、invalid计数0→1；初态越界拒绝。今后该模型域gate直接生成的任务终止必须为0 | 实际真实Env边界回放满足这些数；原grid/点mask/index检查均保留。reset只用当前真实点检查，不伪造H4送模型；在线坏样本跳过PTv3并返回invalid τ。旧27792终止不能变成新的成功数 |
| 动作与reset | `tools/audit/validate_native_action_reset.py`；详见工程报告 | 12轴×正负24/24 PD与实测运动通过；第2/4步indexed reset未重置环境差0 | 24/24通过，独立PD误差0；响应98.1494%–99.2358%，误差≤1.8506%；两次13字段差0。初态腕角和sourcePD超界限制另列 |
| 连续腕角reset | `continuous_wrist.py:12`、`:29`；生产`tools/run/inspire_physics_host.py:110` | 连续rpy只在reset选中env规范到[-π,π)，同姿态不改平移/手指；零腕动作首tick旋转<0.1rad；24/24与indexed隔离不回退 | 生产r4真实刚体quaternion首tick airplane0.002433885rad/cubemedium0.000435070rad；24/24、PD误差0、两次14字段untouched差0。17项CPU语义回归及noop源码mutant检测通过；source-PD平移饱和与完整任务仍未修好 |
| teacher/source-PD实际上限 | `collect_bounded_teacher.py`、`replay_bounded_source_pd.py`、`score_teacher_action_reset.py` | 注册完整任务≥95%；6例需6/6，不降低门槛 | 当前新旧均0/6，**上限资格FAIL**。teacher5例终点后观察不足，sourcePD也有跟踪/动作范围阻断；两臂抬升持握代理6/6不能替代完整任务成功 |
| v6真实前向到PD | `smoke_v6_ppo_feasibility_cpu.py:29`（新增显式checkpoint/device选项） | 真step11000输出[1,24,2,21,3]，稀疏投影[8,11,3]，18个PD目标finite | GPU5真实PTv3 smoke PASS，WAM单次前向0.490903s；官方PPO仍v4/UNWIRED，不自动获得v6 τ执行资格 |
| scale与奖励日志（上一提交） | `wam_v4_runtime.py:124`；`measured_official_env.py:150` | 缺请求scale误收1→0；reward分项之和−实际reward=0；不改变reward数值 | 请求不存在scale明确拒绝；真实Env 2×5controls守恒通过。历史50物体请求scale均匹配，0/50，不能用潜在scale bug解释旧零成功 |

表中未写Task目录前缀的生产文件均位于
`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/`；审计工具均位于
`src/task/ptv3-shared-wam-v6/tools/audit/`。

### 释放时序为什么这样改

旧full raw 490/490 RELEASE都早于完整录制终点，间隔中位2.0s。旧代码把释放阶段放在完整
路径最终settle之后，导致按声明正常释放的两条真实记录被当成失握，退回ACQUIRE。
新实现把事件位姿纳入有序milestone，按序到达声明末次release位姿时授权RELEASE；接着
继续检查录制声明中的剩余几何路径和终态条件。没有截掉post-release路径或放宽空间/时间门槛。
RELEASE实测是持续状态，授权后不再要求重新出现一次上升边沿；提前掉落仍要求恢复。
原测试“到释放位姿后还必须继续持握0.2s”编码了旧错误，已改为“未到释放位姿的掉落必须恢复”，
并新增真实时序的独立正控制与跳路径/无支撑/偏离终点负控制。

全量复评分只变TaskScheduler，其他活跃task/feedback/path代码逐文件相同，保留完整diff。
旧full：0成功/492删失/9path stall；新full：77成功/415删失/9path stall。新prefix为
9成功/490删失/2path stall。授权后未成功128条仍删失，其中116条结束时终态settle不足1s；
不能把415删失解释成415次物理失败，也不能把full的unsafe suffix认证为新的高质量正例。
77成功全部为RELEASE，均有真实终态连续1s观测，不通过末帧后复制状态补时间。
完整产物`outputs/ptv3-shared-wam-v6/prefix-task-rescore-20261022-r1/{old,new,comparison}.json`；
具体命令在[prefix修复报告](prefix-task-semantics-fix-20261022.md)的全量复评分补记。

### voxel修复的协议身份

`initial-geometry-admission.mask-invalid-online-tau.v1`保持原网格与模型输入gate：初态域检查
不通过即报setup错误；在线域检查不通过不送PTv3，τ及valid置0，单独记invalid steps/episodes。
TaskController的真实反馈、路径、恢复和超时仍决定任务终止。没有用voxel域代替机械工作空间，
也没有额外引入未经校准的物理阈值。保留旧`workspace_failures`聚合字段供旧reader兼容，但
本协议不再递增它；看新`model_domain_invalid_*`，不能把旧字段0解释成物理不越界。
新runner同时绑定task admission、scheduler、model-domain三种身份，旧checkpoint不能无声恢复
成同一实验。invalid τ长期fallback的实际控制质量尚未认证。

### 连续腕角修复的范围

真实URDF与PhysX属性均确认joint4/5/6为continuous且无角限，独立FK证实2π规范化保持各link
姿态（double计算≤3.3e−16；真实float32输入约4.5e−7）。生产wrapper先规范传入`_set_env_state`
的18维q中3:6，再由原indexed reset/FK链处理；不修改平移、手指、dq、动作或保存的source PD。
create后再次核实际joint layout/hasLimits，manifest与PPO resume绑定reset_angle_policy。
extended_state新增实测`palm_pose`，供刚体姿态独立检查。

原raw-neutral首tick 2.734649rad与audit normalized-neutral0.002620rad是**实测q经独立FK**的
旋转估计，且两者env数2/1，不能冒充严格同批刚体对照。最终生产r4重新测原两环境、同24轴/
reset协议，真实rigidbody quaternion显示两例均<0.003rad；初态source-PD的1.815rad旋转/
0.6268m位移目标属于另一个强动作控制，不能当零命令reset效应。所有混杂和原始差异均保留。
最后生产回归776envcontrols，含全部尝试本轮物理总计9366；所有自有进程已退出，GPU4释放。

## 复现命令与期望输出

运行环境：CPU检查为`/home2/wyy/miniconda3/envs/graspenv/bin/python`，Torch2.4.1+cu121；
真实native物理为`/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`，Torch2.0.1+cu118。
原生仿真用GPU4，v6前向用GPU5，每次用前检查nvidia-smi；GPU0–3/7其他作业未动。
没有新RL、没有启动v6大训练。命令均从仓库根目录执行，输出必须用新run目录避免覆盖。

```bash
# 逐文件diff；新文件也在最终commit中，可用git show <交付SHA> -- <path>复核。
git diff 60efadce2444709bbbf9802107de7bc5e3f9eef6 HEAD -- src/task/wam-rl-policy-v3 src/task/ptv3-shared-wam-v6

# 若在另一目录重做501迁移，换未存在的output；r1现存产物不覆盖。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/repair_prefix_task_semantics.py \
  --output outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-review

# 期望完整声明501/501准入；旧坏manifest明确被拒绝。initial voxel实际数见fixed-admission.json。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/audit_fixed_admission.py \
  --output outputs/ptv3-shared-wam-v6/fix-review-external/fixed-admission.json

# 最终回归：期望88 passed。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/task/wam-rl-policy-v3/src:src/task/trajectory-policy/src:src/task/ptv3-shared-wam-v6/src \
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
  src/task/wam-rl-policy-v3/tests/test_prefix_task_semantics.py \
  src/task/wam-rl-policy-v3/tests/test_task_annotations.py \
  src/task/wam-rl-policy-v3/tests/test_teacher_goals.py \
  src/task/wam-rl-policy-v3/tests/test_terminal_release_timing.py \
  src/task/wam-rl-policy-v3/tests/test_task_scheduler.py \
  src/task/wam-rl-policy-v3/tests/test_executable_task_admission.py \
  src/task/wam-rl-policy-v3/tests/test_model_domain_admission.py \
  src/task/wam-rl-policy-v3/tests/test_measured_episode_logging.py \
  src/task/wam-rl-policy-v3/tests/test_continuous_wrist.py \
  src/task/wam-rl-policy-v3/tests/test_inspire_reset_submission.py \
  src/task/wam-rl-policy-v3/tests/test_measured_official_reset.py \
  src/task/wam-rl-policy-v3/tests/test_resume_identity.py \
  src/task/wam-rl-policy-v3/tests/test_measured_reward_bridge.py \
  src/task/wam-rl-policy-v3/tests/test_measured_execution.py \
  src/task/wam-rl-policy-v3/tests/test_inspire_task_reward.py \
  src/task/ptv3-shared-wam-v6/tests/test_negative_result_invariants.py \
  src/task/ptv3-shared-wam-v6/tests/test_ppo_model_identity.py \
  src/task/ptv3-shared-wam-v6/tests/test_voxel_harm.py \
  src/task/ptv3-shared-wam-v6/tests/test_continuous_angles.py \
  --basetemp=tmp/fix-review-external-tests

# 期望8/8故意production回归均被实际pytest拒绝，不能由import/collection错误冒充。
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/run_fix_mutants.py \
  --output outputs/ptv3-shared-wam-v6/fix-review-external-mutants

# 期望19个不变量及各自错误注入PASS；覆盖时空/slot/mask/归一化/reward/reset/eval identity。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/task/wam-rl-policy-v3/src:src/task/trajectory-policy/src \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/run_negative_result_invariants.py \
  --output outputs/ptv3-shared-wam-v6/fix-review-external/invariants.json

# GPU编号先重新查看，以下GPU5只在仍空闲时使用。期望PASS、step11000、official UNWIRED。
nvidia-smi
CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 TMPDIR=$PWD/tmp \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/ptv3-shared-wam-v6/tools/audit/smoke_v6_ppo_feasibility_cpu.py \
  --device cuda:0 \
  --checkpoint outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt \
  --output outputs/ptv3-shared-wam-v6/fix-review-external/v6-flow11000.json
```

最终回归88 passed/17.15s；19不变量PASS；最终8/8独立production源码mutants被拒绝。
故意错误包括：完整任务误改UNKNOWN、漏UNKNOWN前置断言、跳过旧sidecar一致性检查、
恢复voxel每tick任务终止、绕过初态voxel准入、恢复延迟释放授权、漏一个reward分项、跳过连续腕角规范化。
这些在独立源码副本和进程中运行，live源码不被改写；每例均保留源码和pytest断言失败日志。
这证明验收能检测这些具体错误，不证明没有其他错误。物理mutants与未杀死的幂等double-couple
单列在[动作/reset报告](teacher-action-reset-validation-20261022.md)，不凑PASS数量。

产物：`outputs/ptv3-shared-wam-v6/fix-review-20261022-r1/`有combined-tests.log、
invariants.json、mutants-final/summary.json、fixed-admission.json、真实v6-smoke JSON；最终扩大回归为final-tests.log。
501迁移/完整性/逐例事件审计见[prefix修复报告](prefix-task-semantics-fix-20261022.md)；
物理复跑的精确命令/源码冻结/初态与动作轨迹见[工程报告](teacher-action-reset-validation-20261022.md)。

仓库`tools/verify.py --changed`仍FAIL：仅6张既有trajectory-policy历史卡缺v2元数据，未伪造
claim/seed/date替它们补齐。47个本轮变更路径未新增verify报错，Python/Markdown/structured与
索引检查通过；原始完整日志`repo-verify.log`保留。全库门禁不伪报为PASS。

## 危害与应撤销结论

| 缺陷 | 已有产物中的数量与预算 | 结论处置 |
|---|---|---|
| prefix语义改写 |473/501=94.41%终态翻转，491/501=98.00%删失；候选56418controls；未保存BC每minibatch序列，精确实际曝光未知 | `P-20261020-inspire-measured-official-ppo`归因降级；继承cache/评分链的`P-20261022-wam-v6-executor-qualification`旧全任务结果须复核；`P-20261022-wam-v6-samples-efficiency-3arm`保持HOLD；局部抬升现象保留 |
| UNKNOWN无成功分支 |旧标签64/501=12.77%，draw9314/56449=16.50%，名义16.95%；不是原声明中64个真UNKNOWN | 撤销“同任务语义有效”的推断。上次437准入方案被完整声明修复取代；真UNKNOWN仍拒绝，不删除旧64样本 |
| v4身份错认 |53347结束回合、16515072已更新环境步都属于v4链 | 撤销v6 PPO方法否决；保留这条v4链未观察完整成功 |
| voxel终止 |1570为27792/53174=52.2661%；最终27869/53347=52.2410% | 撤销机械工作空间解释；没有历史状态/episode长度，不能认定全部误杀或算出修后成功数 |
| 日志缺项 |3rank×1344=4032条curve无return/phase，53347结束episode缺逐例长度/归属 | reward确实进入GAE，保留训练中0成功的描述；未来按实际controls/分项记录 |
| 释放时序 |490/490声明release早于全endpoint；同501真实轨迹full 0→77、prefix 0→9；两已定位例0/2→2/2 | 撤销旧0/501作为评分器可靠/参考不可执行的依据；保留旧物理轨迹本身，不外推物理新执行 |

旧native实际16527360controls，12288未更新。精确UNKNOWN/voxel吃掉多少步无法从未保存日志
恢复；在冻结调度链假设下，UNKNOWN已结束回合关联18636–16382973步，voxel83607–16438638步，
并集83607–16466580步。这些是宽集合界，不能当可省预算或损失梯度量。
从旧已发生回合仅剔除两类，观测写成0/N、N∈[16164,25478]；只剔voxel为0/25478。
它们不等于修复后结果，也不能把UNKNOWN draw占比直接乘训练总步。
完整历史卡的逐张影响面仍见[负结果审计](negative-result-audit-20261022.md)及
[危害预算](negative-result-harm-20261022.md)。

## 待裁决条目与无法自证之处

1. **计划意图**：保留的490R/11H只是完整历史声明，force消失不是天然计划释放。
   501条reference标签均R，11条rawH与其不一致；37/490有掉落代理、29条还缺±3帧reference
   release见证。没有以此删样本或重标成功。Hermes/外部审计需裁决应使用哪个明确任务意图来源。
2. **上限资格**：新teacher/sourcePD6例完整任务目前都0/6，没有接近95%；新评分器在真实旧记录
   上修好两例，并不保证新执行精度、重置物理状态或所有阶段已正确。保持P1 HOLD。
   0/6物理采集发生在生产角代表修复之前；修复后仅验收动作/reset，尚未重跑完整teacher，
   不能把旧0/6套成最新完整链结果。
3. **角代表与动作范围**：reset角代表已修并完成生产native验收；sourcePD raw首tick强动作
   的旋转2.484rad是q-derived FK估计，不能解释成无命令跳变。原330个clipping controls，
   normalized/nearest-angle单例源参考期10→9；余9均平移Y，canonical最低−1.04015，完整任务
   仍0/1。不能把修角代表说成修好了动作范围或teacher上限。
4. **voxel fallback**：网格未改，τ invalid的输入合同已核；长期在线fallback的物理代价/收益未知。
   历史27792首个点未保存，现有P0两个远离目标约2.2m的腕点不能代替其分布。
5. **v6**：真实GPU forward可组合，官方PPO仍未接线。BC训练曾用旧task10/τ cache，shape可用
   不代表适配新goal/event，不能把单H4 finite输出当执行或泛化验收。
6. **复核分歧**：目前没有收到Hermes/外部审计复跑结果。其发现与本报告不一致时，保存命令、
   初态/版本/指标和双方数值为具名待裁决项，不能覆盖旧产物后宣称一致。
7. **候选进度回退**：原档案新scorer尚有76条末态pointer0，initial_acquired则501/501从tick0
   已真。stamp28帧provisional0→1、29帧force0但无grasp_lost又退0；binoculars43–84帧到30，
   85帧force1却无grasp_lost退0；恢复耦合时已离首目标410/270mm。是否允许短接触缺口保留
   独立几何见证需明确接触/路径阶段合同；本轮未调dwell、阈值或自动补进度，保留trace待裁决。

## 若重训v6：最小改动与启动前条件（尚未启动）

1. **数据**：先版本化501完整声明+prefix mask，保留旧数据和cohort；标清491删失/unsafe future，
   不把超prefix事件当NONE或补控制。不默认这次robot task bug污染了所有v6预训练human数据，
   逐数据源核构造器、有效mask和释放意图。原11H/参考R冲突先裁决。
2. **标注**：固定30Hz秒制、未来τ[0]=t+1/30、source帧/手slot/21→11合同；事件意图来源与观测
   接触反馈分开。缓存身份包含完整task、mask、scheduler版本，重建旧task10/local_goal/τ cache。
3. **条件注入**：部署用spatial waypoint及已知planned event，训练要么提供同语义任务条件，
   要么显式区分实际t+24 endpoint条件；不能用GT实际未来替代部署可获得信息。保留invalid
   event、恢复CONTACT、左右slot与缺点mask。验证真实goal/event模块和未来label隔离。
4. **执行器与上限**：先让明确意图/合法reset/合法sourcePD的同指标正控制通过，再判断是否需要
   重训v6、BC适配或两者；旧53347个v4回合不提供v6重训理由。
5. **预算**：先做有上界的label/cache审计和一个固定teacher/GT小panel；若标注确有系统污染且
   Hermes确认，重训建议先沿已保存step11000配置同288样本/全局更新，固定训练/DEV/随机种子，
   提出500更新（144000样本曝光）短资格检查；通过后再由Hermes批准主预算。不能把500更新
   当大规模效果否决。预先记录GPU456分配、wall上限、checkpoint周期/中断保存、固定DEV停机规则。

下一项最便宜的可证伪工作是先对stamp/binoculars保存状态明确“几何路径进度与接触认证”合同，
构造单帧force缺口和真实失握两类相反控制，确认可修复性；随后只复跑一条无动作饱和的sourcePD，
固定新合法reset/任务声明、完整观察终态，要求同指标完整成功。未过不恢复RL三臂或大训练。
