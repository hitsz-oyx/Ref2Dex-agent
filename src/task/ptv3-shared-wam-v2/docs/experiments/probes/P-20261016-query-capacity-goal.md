---
schema: ref2dex.probe.v2
probe_id: P-20261016-query-capacity-goal
experiment_id: P-20261016-query-capacity-goal
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-query-capacity-goal
probe_index_in_family: 1
seed_pool: probe
seeds: [221, 222]
decision_changed_if_positive: retain larger or explicitly goal-conditioned query actor for matched native qualification
decision_changed_if_negative: prioritize control-path and objective diagnostics instead of increasing shared control-head width
status: UNCLEAR
run_id: query-capacity-goal-20261016-r1
---

# Inspire query控制头容量、直接goal与当前观测瓶颈

Class: Decision。服务Objective A的可执行BC substrate，以及Objective B的shared readout定位。
用户明确授权并行控制头128/512×直接goal开关，并同步Shadow BC→RL。此前811D MLP
pooling对照未检验原query头的721→128融合，也不能排除直接计划goal的作用。

## 协议与最便宜的可判别读出

冻结原shared7k WAM，沿用strict TRAIN93680/DEV10376、原固定TRAIN/DEV256及对象面板。
缓存每窗口完整shared tokens和有效mask，不用mean-pool替代query memory。FP32/TF32编码，
FP16保存以限定IO/存储；每个最终actor用留存FP32 tokens重放，首拍XYZ偏差须<0.1mm。
TRAIN-only state87/current721/goal3及8×15 action归一化；goal=已有计划reference_obj[t+24]
位置减当前物体位置，和MLP直接goal完全同义，不读未来实测物体。

五臂：w128_g0/w128_g1/w512_g0/w512_g1，以及w512_g1_bypass。goal编码器均存在，
off仅在标准化后置零；同宽臂model221逐参数同初始化。原state/current融合与两层query
decoder不变，预测完整8×15 geometry-lift q_pd标签。bypass为721→120零初始化线性
直连标准化输出，同宽普通臂也分配但关闭该支路。因此这是增加直接读出能力的Probe，
不能把任何增益完全归因为信息损失，而忽略可训练容量变化。

全部fresh控制头、冻结WAM，half-startup/sequence-uniform同draw222，batch128，
AdamW LR5e-4/WD1e-4、clip1、BF16 actor，6000步固定终点，不挑选DEV checkpoint。
与旧joint7k、原MLP并非单变量比较，五臂内部才做因素归因。width改变也改变参数和
优化难度；报告参数数、TRAIN拟合及DEV分层，不能把wide更好直接称容量假设被正式证明。

主要读出原DEV256首拍XYZ mean/p95与startup，TRAIN256检验拟合，完整对象轨迹面板辅助。
同宽goal on相对off、同goal wide相对narrow分别比较；改善>=10%且startup不恶化才
进入少量native资格评估。无明显信号不自动扩大训练。Probe仅PROMISING/UNPROMISING/UNCLEAR。

## 资源、停止条件和追溯

输出`outputs/ptv3-shared-wam-v2/query-capacity-goal-20261016-r1/`。
完整token缓存<=52GiB；单臂<=2400s、最终权重/optimizer<=0.5GiB，磁盘始终保留20GiB。
缓存GPU2；训练最多并行GPU2/3/4/5（其余臂排队），Shadow可用GPU1，GPU0外部作业不动，
至少GPU6/7保持空闲。远端就绪后可以在专用目录复现相同代码/缓存协议，不改变实验因素。
非有限、缓存覆盖/身份/归一化/量化检查不通过、资源超限立即停止并保留错误记录。
命令、commit、PID、GPU、进度和ETA记录在各run manifest/process/progress。

## Limitations / future evidence

单seed因素筛选，不是正式Validation；未检验joint WAM更新与其他goal姿态表示。
绕过支路是直接线性读出，未单独对齐有效参数数；若正向，再做匹配容量验证。
首轮FP16量化重放仅覆盖首16条DEV的XYZ，不代表全缓存/旋转已获验证；正式归因与native
资格前需在固定面板补查。所有臂共享同一量化cache，内部配对条件相同。
