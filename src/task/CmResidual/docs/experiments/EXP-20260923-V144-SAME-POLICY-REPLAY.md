# V1.44: 同 seed/同 GPU/同策略重跑的物理重现性

- experiment_id: `EXP-20260923-V144-SAME-POLICY-REPLAY`
- branch: `agent/v144-replay-audit`
- run_status: `PLANNED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

V1.43 在干预前同一 seed/env 的 off 与 always 状态
几乎完全不匹配，同 GPU 顺序执行亦然。为了判明
是评测本身不可重现，还是两个动作分支扰动调度，
冻结 V1.43 的 seed98、路由/专家/环境/代码/
首次干预记录条件，在同一 GPU5 上按顺序各重跑
一次 `off` 与 `always`（无 Cm 在线使用）。

分别比较原 run 与重跑 run 的64环境起始帧、
运动编号、第一次满足干预条件的进度、q、原动作
和13维物体状态；同状态阈值沿用 V1.43：q 与
动作最大绝对差≤1e-4，物体位置差范数≤1e-5m，
完整物体状态最大绝对差≤1e-4，同时进度一致。
每臂≥40/64 严格匹配才称“此配置可重现”，
否则重现性失败，V1.43 不匹配不能归因于干预
分支本身。还报告两次完整首 episode 的成功数
与逐环境成功交叉计数，但不把后者解释为因果。

若两臂自身均可重现而 off/always 互不匹配，
再考虑分支内核调度或动作评测代码；若任一臂
自身也不匹配，后续须使用多 seed 的分布级比较
或同一次物理状态克隆，不再使用跨 run 的逐环境
反事实。重跑不训练、不选新策略、不用官方 actor；
最多GPU5一张卡，产物远低于300GB。
