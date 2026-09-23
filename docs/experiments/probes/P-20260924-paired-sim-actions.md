# P-20260924-paired-sim-actions

date: 2026-09-24
branch: agent/cm-paired-sim-actions

## Question

能否从**完全相同的仿真前状态**分别执行两个不同动作，得到可信的
一步物体效应差，而不是把观测动作置乱当作物理反事实？

## Hypothesis

H1: Isaac Gym 的 actor-root + DOF 状态快照/恢复，在一步内让同动作
重复结果的物体位移误差 p95 <0.05mm；在至少 40 个前接触样本中，
至少 20 个动作对的物体差异 >0.2mm 且 >同动作重复误差的 5 倍。

Alternative: 快照不恢复求解器接触历史，或一步动作干预效应小于数值噪声；
应使用同 run 配对环境或设计短 horizon 干预，不能用这些数据训练/评估 Cm。

## Decision

H1 成立：收集小量、分训练/测试 seed 的 paired physical transitions，
直接检验 Cm 对动作效应差的预测，再决定是否投入 PPO。

H1 不成立：停止该采集方式，记录具体失败源；不把不匹配的旧
first-grip 两臂文件当作配对数据。

## Minimal protocol

单个自训练 e260 actor，固定 s3 轨迹、seed，headless 16 环境。
在固定若干控制步，用同一快照运行 base、base-repeat、alt 动作；
alt 为 base 的腕部 z 命令 +0.1（clamp 到合法范围）。
每次恢复完整 actor-root 与 DOF 状态，严格检查恢复误差、
同动作重复误差、动作实际差异和物体一步差异。
只在通过重复性门后才解释动作差。工程 smoke 先取一个 schedule step。
GPU 忙时可运行 2 环境 CPU-only 工程 smoke，结果仅检查采样器能否运行，
不能作为上述 GPU 物理反事实主结论。

## Budget

GPU: 1 张真正空闲卡；wall: <= 60 min；输出 < 100 MB。
若所有 GPU 被占用，只做代码/CPU 单测，不抢占或干扰其他任务。

## Stop condition

输入 SHA 漂移、GPU 被占用、快照恢复失败、同动作重复误差过门、
非有限值或 wall 超预算立即停止。

## Result

Status: PENDING (GPU physical-pair probe has not run)

Key evidence: collector code `3e520c6` passes 3 CPU unit tests and Python
syntax checks. `agent_paired_cpu_smoke_s145_e260` FAILED before the first
simulation step: DExplore `_load_table` calls `.to('cuda')` even when CPU
simulation and CPU policy are requested. This is an engineering limitation,
not evidence for or against the physical-pair hypothesis. All 8 physical
GPUs were occupied by other users at the attempted smoke time.

## Decision update

Do not edit DExplore's basic data loader merely for this smoke. Wait for a
truly unoccupied GPU and run the fixed 16-env GPU smoke; if none becomes
available, report the resource blocker rather than treating observational
action shuffles as physical counterfactuals.

## Artifacts

`src/task/CmResidual/paired_sim_step.py`
`third_party/DExplore/dexplore/evaluate_paired.py`
`outputs/CmResidual/agent_paired_cpu_smoke_s145_e260/run_manifest.json`
