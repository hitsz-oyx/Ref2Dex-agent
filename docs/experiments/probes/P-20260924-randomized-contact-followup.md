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

Status: PENDING

Evidence: pending

## Decision update

pending
