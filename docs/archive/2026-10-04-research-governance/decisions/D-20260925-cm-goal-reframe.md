# D-20260925 — Cm 目标改为固定任务内的 policy utility

状态：已由用户授权，执行中

日期：2026-09-25

## 新的阶段目标

在固定的自训练抓取任务分布内，使用接触后的随机化动作干预和完整
episode 的 held-lift 标签，学习动作/历史条件的 Cm value 表示，并用
matched `Cm-on` / `Cm-off` 检验它是否改善后续抓取决策。

当前第一阶段固定为自训练 airplane 路由。跨物体迁移、Apple 留出和
object-disjoint 泛化不再作为本阶段的继续门槛；它们保留为后续研究债务。

## 为什么改目标

已有同一 airplane 分布的新 simulator seed 上，Cm 按预测 `+z - -z`
排序的五步物体 z 高低四分位差为 `32.96 mm [27.98, 37.85]`，说明短期
动作效应存在可学习信号。此前的 Apple 失败回答的是未见物体迁移问题，
没有回答固定任务内的 policy utility。

当前 blocker 是效应预测还没有连接到最终 held-lift，而不是缺少跨物体
泛化证据。因此监督目标改为：

* 接触后候选动作的多目标短期物理结果；
* 完整 episode 的 held-lift、最大接触支持抬升和接触持续时间；
* 小型 value head 对最终 held-lift 的预测。

## 预注册最小成功条件

1. 随机干预记录必须包含 pre-action 历史、实际执行动作、完整 episode
   held-lift 结果，并通过环境 ID/干预时刻分层审计。
2. Cm-aware value/ranking 在同一物体分布的新 simulator seed 上优于
   state-only 和 action-shuffled 控制。
3. 冻结模型接入候选动作选择后，matched `Cm-on` 相比 `Cm-off` 的
   held-lift 增益在至少两个新 seed 上方向一致；合并环境聚类区间的下界
   高于零，或达到预先固定的最小实际增益门。

单纯训练集拟合误差、五步位移预测或一次成功 rollout 不满足目标。

## 执行边界

本阶段只使用自训练 actor；官方 actor 不能进入最终 Cm-on/off 比较。
优先一 GPU、小环境 pilot 和 CPU 离线分析。2026-09-25 22:00 前由 AI
自主推进，不新增路线选择请求。
