# P-20260923-mixed-cm-sim-adapt

date: 2026-09-23
branch: agent/cm-mixed-pretrain-sim-adapt

## Question

GRAB MANO + Inspire 几何轨迹预训练，是否帮助 ObjectInteractionCm
用少量仿真真实执行转移适配一步物体效应？

## Hypothesis

H1: 在完全相同仿真转移、模型结构、优化器和更新步数下，预训练初始化
比随机初始化在未参与微调的 seed95/96 接触转移上 EPE 低至少 20%，
并且低于零运动基线。

Alternative: 预训练只是运动相关先验，在物理执行域没有样本效率优势；
应改 Cm 表示/动作执行模型，暂不接 PPO。

## Decision

H1 若成立：扩大至新评估 seed 和更多训练数据，并做同状态动作反事实
物理 rollout 诊断；随后才考虑 PPO 接法。

H1 若不成立：先改模型输入/结构，或训练动作到真实手运动的执行模型；
不以混合预训练权重作为既定方向。

## Minimal protocol

训练转移：自训练 s3 策略 e160 seed74 与 e180 seed78，各随机固定接触 /
非接触 16 条，合计 64 条。评估：e260 seed95/96，各接触/非接触
8 条，合计 32 条。所有动作条件输入均来自 pre-action q/action 和
nominal hand flow；next object state 仅作目标。预训练与 scratch
结构相同，使用同一 mini-batch index schedule、AdamW、LR、40 更新步。
比较初始/10/20/40 步的接触 EPE、无交互门控 EPE、零运动 EPE。
这是同轨迹、极小数据的路线 Probe，不是正式泛化或策略效用验证。

## Budget

GPU: 0（当前 GPU 均有他人任务）；CPU threads: 2；wall: <= 60 min；
storage: < 10 MB。优先 1-step/1-sample 工程 smoke，再运行固定 Probe。

## Stop condition

输入 SHA 漂移、非有限 loss、运行显著超出 wall/内存预算立即停止。

## Result

Status: PENDING

Key evidence: pending

## Decision update

pending

## Artifacts

`src/task/CmResidual/tools/probe_mixed_cm_sim_adapt.py`
