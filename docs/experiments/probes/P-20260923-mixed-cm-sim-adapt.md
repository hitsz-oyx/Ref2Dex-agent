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

训练转移：自训练 s3 策略 e160 seed74 与 e180 seed78，各随机固定
真实物体平移 >2mm 的接触转移 / 非接触转移 16 条，合计 64 条。
评估：e260 seed95/96，按同样分层各 8 条，合计 32 条。固定分层是
为了避免静止物体主导小样本 Probe；并不改变下一状态只用于采样/监督的边界。
所有动作条件输入均来自 pre-action q/action 和
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

Status: UNPROMISING（当前 64 样本 / 40 更新配方）

Key evidence: run `agent_mixed_cm_sim_adapt_40step` 完成，CPU 2 threads、GPU 0、
约 67 秒，输入 SHA 与提交固定。训练 64 条（运动接触 32 / 非接触 32），
未见评估 32 条（运动接触 16 / 非接触 16）。

| 接触点流 EPE | 初始 | 10 步 | 20 步 | 40 步 |
| --- | ---: | ---: | ---: | ---: |
| 混合预训练 | 34.34 mm | 15.16 mm | 14.89 mm | 10.65 mm |
| 随机初始化 | 10.44 mm | 10.27 mm | 10.33 mm | 11.22 mm |
| 零物体运动 | 10.44 mm | 10.44 mm | 10.44 mm | 10.44 mm |

40 步预训练相对 scratch 只低 5.1%，未达 ≥20% 预设门槛，
且仍未优于零运动。两个模型都未获得有用的物理效应预测。
非接触样本只有 1/16 被几何交互激活；显式门控后预训练仍有
0.24mm EPE，零基线约 0.006mm。训练集极小、同轨迹、
评估 seed 已用于前一冻结迁移 Probe，不能推出预训练一般无用。

## Decision update

不扩大当前 V1.3 全量微调，不接 PPO。下一 Decision Probe 应先分开
“动作→实际手运动”失配与“接触→物体运动”建模：训练/校准可在线使用的
一步手执行预测，或重设计更小的局部几何 Cm，并在新的未见 seed 上
与 scratch/零运动对照。混合几何数据可保留作表示预训练，但穿模需
另做 signed-SDF 质量审计；目前无证据证明它的物理效应标签可信。

## Artifacts

`src/task/CmResidual/tools/probe_mixed_cm_sim_adapt.py`
`outputs/CmResidual/agent_mixed_cm_sim_adapt_40step/report.json`
`outputs/CmResidual/agent_mixed_cm_sim_adapt_40step/run_manifest.json`
