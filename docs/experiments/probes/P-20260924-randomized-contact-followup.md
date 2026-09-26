# P-20260924-randomized-contact-followup

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Decision

## Question

随机 +0.1 抬腕的一步物体 z 位移有利，但在线重复上抬未提高抓取。
是因为处理破坏随后数步的手物接触，还是单步目标本身对长期抓取
信息不足？

## Hypothesis and decision

在新的 seed151/152、64 env、每 10 步随机 ±0.1 的接触状态，
记录处理后 5 个控制步。若两个 seed 的 5 步接触比例 `plus-minus`
均为负、合并方向一致且环境聚类 95% CI 不含 0，同时一步抬升
仍为正，则优先训练“动作物体效应 + 接触保持”Cm 并做一次新的
在线 matched Probe。否则停止局部抬腕排序路线，转向训练期
Cm 表示/辅助目标；不在 seed149/150 上重调阈值。

该 Probe 只决定下一种目标，不证明策略效用。短期接触无差异
也不排除更长期的破坏机制。

## Minimal protocol

固定自训练 e260 actor、s3 Inspire 几何重定向轨迹和仿真参数。
仅全局步 50..150、stride10、实际接触且未 reset 的环境参与
分步平衡随机化。干预一次后恢复 actor，记录一步物体 z 位移、
接下来 5 步的接触比例、末步接触、累计物体 z 位移和存活。
分步加权比较 ± 两臂，按 environment ID 聚类 bootstrap 2000 次。
无逐状态配对，也不把复用环境行当独立 episode。

## Budget and stop

每次 1 张空闲 GPU，64 env、单 run <30 min；两 run 合计 <60 min，
总输出 <200MB。固定代码和输入 SHA；GPU 冲突、输入漂移、
非有限值、记录不完整或超预算即停。

## Result

Status: PROMISING（短期接触损失机制；非策略效用）

两次 GPU 仿真均 `COMPLETED`，每次 64 env、11 个随机处理时刻，
各 634 条接触处理。分步加权 `+0.1` 减 `−0.1`，environment-ID
聚类 bootstrap 2000 次：

| outcome | seed151 | seed152 |
| --- | ---: | ---: |
| 一步物体 z 位移 | +12.19mm [9.57,14.71] | +13.44mm [10.40,16.64] |
| 5 步物体 z 位移 | +9.09mm [5.61,12.77] | +11.46mm [7.68,15.73] |
| 5 步接触比例 | −7.58pp [−10.31,−5.19] | −4.70pp [−6.77,−2.73] |
| 第 5 步仍接触 | −5.62pp [−9.26,−2.22] | −3.38pp [−6.29,−0.53] |

两 seed 的接触比例 95% 区间均严格为负，且一步物体上抬
均严格为正；满足预设的“优先接触稳定 Cm”分支。5 步存活差
两 seed 均为 0。本结果不是“+0.1 相比不干预”的直接 RCT，
也不是逐状态反事实或长期策略收益。

产物：
`outputs/CmResidual/agent_randomized_followup_s{151,152}_d01_h5_n64/`
中的 `run_manifest.json`、`transitions.pt`、`followup_report.json`。

## Decision update

训练一个动作条件的局部交互 Cm，使其同时预测一步物体效应
和短期接触保持/持续物体位移；在未见 seed 上检验动作效应排序
及简单 raw-action 模型对照。只有模型可判别这项权衡，才做
matched 在线策略 Probe。停止纯一步 z 分数直接上抬路线。
