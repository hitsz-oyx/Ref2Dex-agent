---
schema: ref2dex.probe.v2
probe_id: P-20261022-grasp-coverage40
experiment_id: P-20261022-grasp-coverage40
date: 2026-10-22
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-grasp-coverage40
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 139, 180, 196, 219]
decision_changed_if_positive: retain geometry and SE3 repair on new interface initials and budget drift recovery residual separately
decision_changed_if_negative: repair approach or qualify physical witness before attributing failure to WAM
status: RUNNING
run_id: grasp-v11-coverage40-20261022-r1
---

# v11 扩大初态覆盖：每链40个接口未见初态

## 结果解读表

| 冻结结果 | 结论 | 下一步决定 |
|---|---|---|
| A1>A2，Fisher p<.05且Wilson95区间不重叠 | 本批接口未见初态上，τ内容对同R执行有贡献 | 保留接口；不升级为预训练收益 |
| A1/A2相当，A0物理见证成功 | 控制接口/参考意图仍待修；不能声称τ收益 | 在漂移状态上训练意图修正，PPO另卡限预算 |
| 三链均失败或A0命令准入失败 | 没有可执行正见证，尚不能归因WAM或任务不可能 | 保留全部失败，修物理接近/动作合同 |
| A1仍有关节/桌面中止 | 静态候选准入不足以保证动态安全 | 修接触下的压力与速度控制；不放宽gate |

Result: registered 40 distinct initials /20 unexposed objects x3 arms=120 episodes; resource queue starts GPU5/6 after preceding COVERAGE, GPU4 after preceding residual check, 0 updates.
Decision: n12 engineering coverage remains separate; use n40 per chain and retain every physical failure.

分类Decision：区分“接口只适合旧圆柱初态”与“新几何/SE3合同可扩到新初态”，决定继续修接近还是可保留接口。
最便宜符合机主统计要求的批次为40个不同初态×三链，不追加v9重复或训练。

**抽样在任何新成绩产生前固定**：剔除v8/v9/v10及并行n12覆盖已暴露的18对象，
余下缓存scale=1且engineering PASS、至少两条序列的对象中，按对象字典序取20个、每个取最前两条序列。
共40不同初态，20对象×2序列；不看source是否成功/最终误差，不以失败换样本。
这些是接口未见，BC/WAM预训练是否见过未知；不是数据集完全未见。物体及精确序列写进registration。
每链一个同质`interface_unseen`子集n40；与旧seen结果、12例工程覆盖分别报告，禁止合并。

三链：A1=原R+WAMτ；A2=同R同相位/伺服，只清零全部24帧τ；A0=source-PD接近，实测抓住后同SE3伺服。
**A0是不同动作链的物理见证，不是GTτ→R上界**。A3/A4上界仍未建立，本卡不补造。
每初态native seed=100+i，推理seed=180+i，初始化seed196，三链共用九字段初态；Latin rotation平衡顺序。
每次fresh物理进程、env1、30Hz×600controls=20s，不成功即停。
最多72000环境控制，0训练更新，PPO未授权；每worker≤14400s，等卡≤10800s，磁盘余量≥20GiB。
只用4/5/6；已有handoff的COVERAGE结束后后续仅用4，故5/6在各自≤64MiB时先启动，4等handoff退出。
冲突不杀进程、不偷卡、不换seed。此资源排队修订发生在本卡任何仿真开始前，
原QUEUED记录完整保留`queue-before-resource-amendment.json/log`，总排队截止仍沿用最初时间；
科学协议、源快照、初态/种子/臂顺序与预算不变。新supervisor/PID/确切worker命令见`supervisor.json`。

新协议主指标：完整20s末、实测接触耦合/离桌≥5cm/连续hold≥1s、
XYZ目标误差≤物体最小凸包宽度的0.10且无物理中止。旋转目标仅在任务显式指定时评估；
当前HOLD无姿态任务，不任意添加评分姿态。release不适用，不做完整RELEASE资格。
逐条另报≤15mm比例/误差分布，但**不与v10固定15mm协议直接混算改善**。
逐臂计数、Wilson95、两两Fisher及配对McNemar；20对象内两序列相关，
trial区间/检验限于固定面板，不能充当40独立物体的总体推断。

期望改变：旧接口unseen关节中止40/40→目标≤4/40；A0目标≥32/40作物理工程见证，
A1目标≥32/40且胜过A2。均为预注册目标，实际结果不能预填，不作为启动有界PPO的通用80%门槛。
静态PD目标准入不保证接触动力学安全；旧40中36条末次PD目标合法仍触发实测限位。

## 来源与仍不一致

Hermes提出尺寸化几何、SE3伺服、候选限位预检；GPT反对把它们当统一答案，并主张扩大覆盖与有界残差RL。
采纳：代码修已定位接口，新增覆盖，BC转漂移恢复，PPO限预算另注册，不要求seen80%才可启动。
仍不同意把n12工程样本当正式覆盖，故本卡n40；不因单元PASS宣称动态安全。
v6当前物体坐标系、goal_valid全True已独立核实；剩余是训练t+24实际终点 vs 在线scheduler waypoint的语义/时间差，
本卡不更改WAM条件、不重训。结果不能归因坐标或mask错误。

## 可复核diff与从干净输出复现

```sh
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
TOOLS=src/task/wam-rl-policy-v3/tools/run
$PY -m pytest -q src/task/wam-rl-policy-v3/tests/test_coverage40_registration.py src/task/wam-rl-policy-v3/tests/test_geometry_runtime_snapshot.py
# 期望3 passed / 0 failed；scale .5不能冒充1，缺source_pd的快照被拒绝。
$PY $TOOLS/prepare_grasp_coverage40.py --output outputs/wam-rl-policy-v3/grasp-v11-coverage40-hermes-repeat1
# 期望REGISTERED 40 distinct initials /20 unexposed objects x3 arms=120 episodes; <=72000 controls; 0 updates
$PY $TOOLS/run_grasp_coverage40.py --run outputs/wam-rl-policy-v3/grasp-v11-coverage40-hermes-repeat1 --detached
# QUEUED→RUNNING→VERIFYING→COMPLETED；若FAILED，保留现场，不自动重采样。
```

环境：graspenv Python3.8.20/Torch2.4.1+cu121；native dexplore_v120_train Python3.8.10/Torch2.0.1+cu118/IsaacGym。
注册时冻结算法源码及worker；worker只更换冻结代码根，不改控制常数。
源码：`tools/run/prepare_grasp_coverage40.py`（暴露隔离、scale准入、120回合注册），
`tools/run/run_grasp_coverage40.py`（排队/预算/确切命令/重评分），
`tests/test_coverage40_registration.py`、`tests/test_geometry_runtime_snapshot.py`（负控制）。
算法diff属于并行实现，见[独立复核与已修缺口](../../grasp-v11-independent-findings-20261022.md)及
[三处修复](../../grasp-geometry-coverage-20261022.md)。不把共同作者实现归为本线程独立验证。
产物根：`outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1/`。
最后必须通过：40三链九字段初态一致、40干净A2、120 raw重评分与标签翻转负控制；
这些仍共用冻结反馈/几何模块，不等于外部接触真值。Hermes/GPT独立物理复跑尚待裁决。

已做注册器故意错误注入：允许scale不匹配、漏掉暴露对象筛选，**killed2/2**；
`outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/coverage40-registration-mutants.json`。
当前完整v3回归**257passed/0failed/0skipped，19.43s**；注册+快照定向**3passed/0failed**。
这些只核软件合同，不作为三处修复在物理环境有效的证据。
