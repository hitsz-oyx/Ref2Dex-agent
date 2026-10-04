# D-20260925 — Cm 下一条研究路线

状态：需要用户选择
分支：`agent/cm-conditional-grasp-value`
日期：2026-09-25

## 当前需要决定的问题

在“随机干预上的条件效应排序”已经在单一飞机轨迹上出现信号、但
跨物体留出失败之后，下一笔 GPU/实现预算投向哪一层 Cm？

## 关键证据

* `P-20260925-cm-history-grasp-value`：s151–154 训练、s155–156
  未见 seed 的历史动作模型把五步 z 处理效应分成 **32.96 mm
  [27.98, 37.85]** 的高低四分位差；动作置乱为 **−13.67 mm
  [−20.40, −6.67]**。独立拟合 seed 复现方向。
* 同一 Probe 的当前时刻动作模型为 **31.78 mm [26.40, 37.28]**，
  所以历史编码的额外价值没有被分离出来。
* `P-20260925-cm-crossobject-history-value`：s186 训练物体到未见
  apple 的五步 z 高低差为 **−6.00 mm [−15.68, 4.02]**，预注册
  迁移门失败。
* 已有 initial option-value 两个留出 seed 合计 action-aware 离线
  选择 **40/128**，固定真实路由 **41/128**；接触后 source→balanced
  短/长 option 也未过抓取门。当前没有 Cm policy utility 证据。

## Option A — 任务对齐的 post-contact 随机干预 + value head（推荐）

在现有自训练专家路由的首次稳定接触点，随机分配两个候选动作/option，
继续原策略到完整首 episode，记录多目标物理结果和最终 held-lift。用
短历史 Cm 预测各物理头，再用很小的 value head 从这些预测和观测历史
学习最终 held-lift；比较 action-aware、state-only、action-shuffled。

预计成本：先做 1 个少环境 pilot，再做 2 个新 seed；约 1–2 张 GPU、
每组 <=30 min，加上离线 CPU 分析。需要实现新的完整 episode 记录和
严格的随机分层/环境聚类分析。

成功后：冻结模型，做 matched Cm-on/off 在线候选选择；仍需正式多 seed
Validation。失败后：停止当前物理 selector，转向 critic/representation
级别，不再调同一动作族。

## Option B — 跨物体 Cm 表示重设计

保留随机干预监督，但加入物体相对几何/形状 token，重新做 object-disjoint
leave-one-object-out；只有至少两个留出物体的效应排序通过才进入策略。

预计成本：需要审计可用物体几何、重建输入和多折 CPU/GPU Probe，约
1–3 个工作日和 2–4 个 GPU 运行。成功后：再回到 Option A 的任务对齐
value head；失败后：放弃当前局部几何 selector。

## Option C — 直接接入现有专家 PPO critic/auxiliary

复用现有专家路由和已有短期 Cm 特征，把 Cm 作为 critic/advantage 辅助
目标，做 matched on/off。预计实现最快，但此前 PPO/effect-rank 和多种
辅助接法已有不稳定或负结果；若失败，新增信息量有限。

## 当前建议

选择 **Option A**。它直接测量 Mission 要求的后续抓取价值，遵循附件中
“随机统计反事实 + 多目标物理头 + 小 value head”的方向，也能避免把
单一飞机的物理排序误当成跨物体通用表示。Option A 需要用户确认后才
启动新的 GPU 数据采集；在确认前不启动不可逆或高成本运行。
