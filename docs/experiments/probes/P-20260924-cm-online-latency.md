# P-20260924-cm-online-latency

date: 2026-09-24
branch: agent/cm-randomized-action-effect
classification: Blocker

## Question

六区域 Cm（包括手/物几何和在线执行手流）在 DExplore 的 GPU
状态张量上，能否足够快地为少环境的两个候选动作打分？

## Decision

16 环境、每环境两候选动作，warm GPU 推理 `p95 <=33ms`
（DExplore 30Hz 控制步时长）且输出有限：做一次少环境在线
Cm-on/off Probe。超过：先优化几何/缓存或减少候选，不做在线
性能宣称。64 环境延迟仅作扩展容量参考。

## Minimal protocol

固定已训练 Cm checkpoint、校准和未见 seed148 的当前状态；
每次合并 2×16 或 2×64 候选，GPU 上从 `q,dof_vel,action,
object_state` 生成 token 并前向。预热后记录 20 次同步 wall
的 median/p95，不包含一次性 URDF/mesh 初始化。输出有限性
和 checkpoint/输入 SHA 固定。

## Budget and stop

一张真正空闲 GPU，wall <=5 min，输出 <1MB；占用或漂移停止。

## Result

Status: PENDING

Evidence: pending
