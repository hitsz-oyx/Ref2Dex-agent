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

Status: UNPROMISING (both exact-pair acquisition methods rejected)

Key evidence: collector code `3e520c6` passes 3 CPU unit tests and Python
syntax checks. `agent_paired_cpu_smoke_s145_e260` FAILED before the first
simulation step: DExplore `_load_table` calls `.to('cuda')` even when CPU
simulation and CPU policy are requested. This is an engineering limitation,
not evidence for or against the physical-pair hypothesis. All 8 physical
GPUs were occupied by other users at the attempted smoke time.

`agent_paired_gpu_smoke_s145_e260` later ran on previously idle physical GPU6
and FAILED at the first scheduled physical fork: actor-root/DOF tensors were
restored exactly, but rigid-body state still differed by 8.03048 in tensor
units. No pairs were accepted. This rejects the assumed sequential
root+DOF snapshot/restore contract, not the Cm hypothesis.

Parallel-env smoke `agent_parallel_pair_smoke_s145_n3` also rejected its
scheduled pair: root/DOF/rigid prestate gaps were 2.79/1.81/2.79 after
80 common policy steps. From identical frame-zero starts, diagnostic runs
showed at step1 a DOF gap of 0 and rigid gap ~1.2e-6; by step2 the
DOF/rigid gaps were already 0.184/0.225 while policy action maximum gap
was 0.00159. The gaps grew through contact. No action-effect pair was
accepted. These are engineering rejection diagnostics, not Cm metrics.

## Decision update

Stop sequential snapshot/restore collection. A single run with three
identically initialized environments (base, same-action repeat, alternate)
can avoid restoring solver state. All arms must start at reference frame zero
and evolve under identical policy commands until the intervention step.

H2 engineering gate: immediately before the fork, all root/DOF/rigid states
and reference-progress metadata must match within 1e-4; after a common
action, the repeat arm must match the base object displacement within 0.05mm
and joint positions within 1e-4. The smoke uses 3 env, seed145, step80,
alt wrist-z +0.1. If H2 passes, expand to small matched triplets and measure
whether any pre-contact action effect exceeds 0.2mm and 5x replay error;
otherwise stop this physical-pair route and do not label observational action
shuffles as counterfactuals.

H2 failed at the prestate gate. Stop exact-pair acquisition for now.
Randomized actions actually executed in the simulator can identify an
average intervention effect without pretending to provide individual
same-state counterfactuals. The next Probe should test that route.

## Artifacts

`src/task/CmResidual/paired_sim_step.py`
`third_party/DExplore/dexplore/evaluate_paired.py`
`src/task/CmResidual/parallel_sim_pair.py`
`third_party/DExplore/dexplore/evaluate_parallel_pairs.py`
`outputs/CmResidual/agent_paired_cpu_smoke_s145_e260/run_manifest.json`
`outputs/CmResidual/agent_paired_gpu_smoke_s145_e260/run_manifest.json`
`outputs/CmResidual/agent_parallel_pair_smoke_s145_n3/run_manifest.json`
`outputs/CmResidual/agent_parallel_pair_drift2_s145_n3/run_manifest.json`
