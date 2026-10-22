# v10：同 R 消融、接近可达性、精确到位修复

## 结果解读表

| 冻结验证结果（每臂每子集40） | 结论 | 下一步决定 |
|---|---|---|
| seen A1 **35/40**，A2 **0/40**；Fisher **2.27289e-17**，Wilson区间不重叠 | 本面板工程目标通过，τ内容对同一R链有贡献；不证明预训练收益 | 保留修复；新对象资格仍单独验收 |
| 原unseen A1 **3/40**，A2 **0/40**；Fisher **.240506** | 新对象尚不合格，τ收益不成立 | 三臂大RL继续HOLD；另按n40扩覆盖 |
| 原unseen A1限位 **40→0/40**，但桌面穿透 **17/40** | 限位局部改善不等于动态安全通过 | 保留全部失败，修接近与接触下压力，不放宽gate |
| seen末态持握但未到位 **16→0/40** | live目标伺服已修本面板主要到位问题 | 保持旧15mm评分；新尺寸协议另报，不能重标旧分母 |

状态：**COMPLETED，160/160回合及raw独立重评分完成**。零训练更新；不重训v6，本卡不运行RL。
全量逐臂CI、最长持握/误差分布、失败分类、82347controls（含881中断）和可复现命令见
[完成交付报告](grasp-handoff-results-20261022.md)。seen Wilson95=[.738879,.945405]，
A2=[0,.087622]；原unseen A1=[.025836,.198642]。35/40是观察值≥80%，不代表95%置信下界≥80%。
原unseen现为开发面板，不冒充本轮未调过的holdout。当前完整v3测试：**257 passed /0 failed /0 skipped，19.43s**，
日志`outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/unit-tests-current.log`；
初始v10冻结版本244项通过，新增几何/快照/注册测试使数量不同。PASS不代替物理资格。
后续[独立接口未见n40覆盖](experiments/probes/P-20261022-grasp-coverage40.md)已预注册并排队，
每链40个不同初态、20新物体，0updates；与并行12例工程检查不合并。

冻结目标：seen 严格成功由15/40提高到至少32/40；v9未见面板 A1 关节中止40/40下降，目标≤4/40。
严格成功仍为完整600控制/20秒末，S2c实测接触耦合、离桌抬升≥5cm且连续持握≥1s，物体XYZ目标误差≤15mm，无物理中止。
HOLD任务 release 不适用。不能把持握31/40当作精确成功。

A2 保留同一个冻结 R、同一个 intent reducer、相位机和伺服，只把全部24帧未来替换为当前手点（即位移τ=0），共同状态的valid不变；不留 WAM keypose 泄漏，不再旁路 R。
两臂都计算 WAM 可用性，参考内容干预之后再进入同一路径；闭环运行后的状态差异是干预结果，不要求不同轨迹的观测相等。

开发上限24个工程触发样本/14400控制，无成功率推断；随后冻结实现，在原v9同初态、同seed、同预算上两臂各40 seen/40面板复验（160回合，≤96000控制）。
八个v9未见物体已用于本轮诊断，因此本轮只称“原未见面板复验”，不能再称未调参的新holdout。不得根据结果补抽/删除物理失败。

已保存触发：`outputs/wam-rl-policy-v3/grasp-v9-stat40-20261022-r1/`；临时诊断 `tmp/v10-diagnosis/first-failures.json`。
DOF15/16与10的具体原生名称将从新host回执核对，禁止按XML文本顺序猜。

复现预注册：
```sh
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/wam-rl-policy-v3/tools/run/register_grasp_repair.py --engineering --output outputs/wam-rl-policy-v3/grasp-v10-engineering-20261022-r1/registration.json
# 正式验证去掉 --engineering；工程运行不是n<40的效果对照。
```
所有运行从新目录启动，保存代码快照、初态、实测PD、原始状态、阶段和物理中止；原安全gate/奖励评分阈值保持。

工程r1回退与修正：统一25mm/0.10rad限速使已知可行seen触发样本无接触，主动SIGINT停止三个自有worker，保存3个完成+3个partial（2813 controls）。R输出是PD偏置，不能当作立即到达的运动学位姿。新过滤仅在**实测**手点/掌mesh进入CAD桌面有限支撑范围且向下侵入超过15mm时介入，正常R命令逐字节保留。原物理安全中止不变。此开发数据不进入正式n40统计。
新host原生DOF名称已确认；旧未见面板：31/40 approach、9/40 close中止；thumb_proximal_pitch_joint 11、thumb_intermediate_joint 25、pinky_proximal_joint 6（42个关节事件/40回合）。39/40不是腕pitch近奇异构型，故不能用Euler奇异性解释大多数中止。

工程r2：两条seen触发轨迹（其中p006旧版误差1649.6mm）20秒末均精确持握，误差约0.01mm与0.00mm；这不是n40成功率。三条未见触发中2条完成全时域无中止、1条延迟至460控制后拇指中节限位；未建立持握。不能把延迟中止或无中止算作成功。
下一修复对实际“接近桌面仍要求向下大力压”或有限关节受压状态，先卸载，再从当前物体位姿、实测手形与τ抓姿提示生成通用候选；无对象/初态ID分支。R仍调用，相位由实测推进。正式A1/A2共享此逻辑。
工程r3预定p004/p024/p036/p050/p058（旧seen五个限位触发）与p001/p005/p013（未见三个触发），8次/≤4800控制。加此前6+5次工程初始化，累计最多19/24次，剩5次；全部保留，禁止把开发样本成功比例冒充验证。

冻结验收前工程r3全部8例结束：旧seen五个限位触发中4例恢复严格成功、1例仍失败；三个未见触发均未成功（两个原物理桌面穿透中止，一个无持握且位置失控）。此处只用于修复开发，不能把4/5当作验收率。工程累计19次、所有partial计入；不继续扩充开发采样。
回归检查：243 passed /0 failed/0 skipped；实际注入六种错误（τ泄漏、mask变更、弱伺服、漏积分、总是过滤R、忽略物体中心）全部被拒绝6/6。首版“总是过滤”变异并未真正改变行为，记录为5/6；已修正变异位置后6/6，不能用无效变异吹PASS。
正式冻结批次：`grasp-v10-stat40-20261022-r1`，80配对初态×2臂=160回合，≤96000controls、0updates。逐子集n40，20s不因成功停止。当前物理资格以正式结果为准。

正式r1写盘中断：新增饱和次数在进入lift/hold后由`sum(np.bool_)`成为`np.int64`，JSON拒绝。三个worker均在已保存完整物理轨迹之后退出；g6还有一个先完成的A2结果。保持原FAILED manifest，`recover_grasp_serialization.py`只恢复完成的记录（600步时域/原gate终止、native exit0、全量tick/阶段均验证），随后独立raw-state重评分；绝不重跑这些样本。修复仅为JSON原生int，消费者/评分模块字节不变。用`--carry-run .../gN-recovered`在新`gN-resume`目录续剩余固定任务，已有2400控制入原预算，不重复采样。新增回归覆盖实际worker计数的JSON序列化。

## 可复核 diff 与预期

| 修复 | 实际入口/代码 | 期望改变的量 | 不允许升级的结论 |
|---|---|---|---|
| A2消融干净化 | `grasp_repair.py:10`；`run_grasp_repair_worker.py:173`、`:188` | A2每控制步调用R（旧为0）；τ最大绝对值=0，24帧抓姿输入也同步去内容；共同状态421个非τ字段/valid逐字节一致 | 对输入τ的依赖，不等于已证明预训练或完整24帧表示不可替代 |
| 受压构型恢复 | `grasp_repair.py:53`、`:115`、`:168` | 原未见面板关节中止40/40下降，目标≤4/40；必须同时报告其它中止和实际成功，禁止只换失败名称后称通过 | 减少限位≠未见抓取资格；候选只覆盖实测点/掌面近似，不能保证整手无碰撞 |
| 抓后精确伺服 | `grasp_repair.py:21`、`:156` | seen严格成功15/40→至少32/40；所有末态误差中位79.6mm下降，单报全20s生存者分布 | 命令修正变大、离线PASS都不等于已物理到位；原评分/安全gate保留 |
| 计数序列化 | `run_grasp_repair_worker.py:15`；`recover_grasp_serialization.py` | JSON类型错误归零，完成轨迹不重跑，分母/预算保持 | 结果恢复不能更改控制器、评分或失败分母 |

代码diff：
```sh
git diff 76e400fe bdf580a7 -- src/task/wam-rl-policy-v3
git diff bdf580a7 3f226f8d -- src/task/wam-rl-policy-v3/tools/run
```
正式控制代码冻结于bdf580a7；3f226f8d只修计数写盘与无重仿真恢复。三卡快照中consumer/pose/stage模块与当前源字节相同；原native安全gate不改。

## 从干净输出目录复跑

环境：模型/统计用`/home2/wyy/miniconda3/envs/graspenv/bin/python`（Python3.8.20、Torch2.4.1+cu121）；host由worker自动启动`/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`（Python3.8.10、Torch2.0.1+cu118、IsaacGym），模型与host分进程同卡。使用前查`nvidia-smi`，4/5/6有冲突则排队；不占0–3/7。

必要输入均为既有产物：注册器读取v9原始registration/panel，R=`inspire-scheduled686-bc-20261020-r4/latest.pt`，WAM=`flowmatching-overnight-20261022-r2/training/step011000.pt`，native cache=`inspire-shadow-parity-20261019-r1/cache`；完整绝对路径写在冻结manifest/worker中。无新下载、训练或凭据操作。

```sh
nvidia-smi
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
REPRO=outputs/wam-rl-policy-v3/grasp-v10-stat40-repro
$PY src/task/wam-rl-policy-v3/tools/run/register_grasp_repair.py --output "$REPRO/registration.json"
# 三个命令分别在空闲GPU4/5/6启动，可并行；目录必须是全新路径。
$PY -u src/task/wam-rl-policy-v3/tools/run/run_grasp_repair_worker.py --registration "$REPRO/registration.json" --worker 0 --gpu 4 --output "$REPRO/g4"
$PY -u src/task/wam-rl-policy-v3/tools/run/run_grasp_repair_worker.py --registration "$REPRO/registration.json" --worker 1 --gpu 5 --output "$REPRO/g5"
$PY -u src/task/wam-rl-policy-v3/tools/run/run_grasp_repair_worker.py --registration "$REPRO/registration.json" --worker 2 --gpu 6 --output "$REPRO/g6"
$PY src/task/wam-rl-policy-v3/tools/run/summarize_large_grasp.py --registration "$REPRO/registration.json" --manifest "$REPRO/g4/manifest.json" --manifest "$REPRO/g5/manifest.json" --manifest "$REPRO/g6/manifest.json" --output "$REPRO/summary.json"
$PY src/task/wam-rl-policy-v3/tools/run/verify_large_initials.py --arms A1,A2 --manifest "$REPRO/g4/manifest.json" --manifest "$REPRO/g5/manifest.json" --manifest "$REPRO/g6/manifest.json" --output "$REPRO/initial-parity.json"
$PY src/task/wam-rl-policy-v3/tools/run/rescore_large_grasp.py --worker "$REPRO/g4" --output "$REPRO/g4-rescore.json" --verify-mutant
# g5/g6同样重评分。期望160回合无重复，初态80/80一致，raw重算160/160一致；实际成功数以物理复跑为准。
PYTHONPATH=src/task/wam-rl-policy-v3/src $PY -m pytest -q src/task/wam-rl-policy-v3/tests
$PY src/task/wam-rl-policy-v3/tools/audit/run_grasp_repair_mutants.py --output outputs/wam-rl-policy-v3/grasp-v10-mutants-repro
```
期望回归故障注入6/6被拒绝；测试数以本轮最终日志的passed/failed/skipped全量为准。PhysX接触过程非逐位确定，冻结seed/初态不保证独立复跑计数逐位相等，故保留n40及CI。

无法自证/边界：实测力与支持是现有引擎与observer合同，未获得外部真实接触标签；几何候选不保证任意物体力闭合；本轮原未见面板已用于开发；完整RELEASE和GTτ上界未由本轮建立。三臂RL仍HOLD。若独立复跑与此报告不同，保留原始记录并列为待裁决，不修改过去结果。

第二次技术中断（21:19）：三个policy进程均消失、manifest停在RUNNING；native分别明确记录`Connection reset by peer`/`IPC peer closed`，不是物理安全中止。原始进程退出原因无回执，不能断言是GPU被抢或OOM。39个完成回合全部保留；没有遗漏的完成result。三个未完成回合原始轨迹未落盘，但native错误回执记录精确161/510/210控制，共881控制。仅这三个固定初态/seed需从reset重试，不计为物理失败、不补抽。技术开销明确额外881，原有效验证上限96000控制不变，总执行上界96881；0updates。`prepare_interrupted_grasp_carry.py`保留原RUNNING/FAILED文件并生成独立carry收据，完成记录不重跑。改由关闭stdio/独立session的detached launcher续跑，避免依赖短会话exec句柄。
