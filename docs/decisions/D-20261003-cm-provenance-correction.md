# Decision Memo：修正 Cm decision/residual Probe 的实现契约

Date: 2026-10-03

## 当前需要决定的问题

审查发现当前两条近期路线存在结果生成错误：decision-interface 的 `shuffled` 控制与
`direct_q` 相同，residual policy 将 local-frame `z` 当成 world-frame 高度，两个 native
collector 在 reset 后复用了 stale motion/start/rest，residual 记录把 `PYTHONHASHSEED`
当成 simulator seed，且 checkout 缺失 `src.task.Cm.dataset` 与 `src.task.Cm.tools.data`。

## 关键证据

直接对 score permutation 与原始 `argmax` 做了随机复现，选择结果逐行相同（只可能在并列
时改变 tie-break）。`consequence_score` 原先直接使用 output 的第三个 local 坐标；旋转
物体后它与 native world-height 指标不等价。当前 checkout 不存在 `src/task/Cm/dataset` 和
`src/task/Cm/tools/data`，
而现有测试和训练入口直接导入该模块。

## 选择的行动及理由

接受审查结论，保留旧 records/checkpoints 但冻结受影响的控制和科学解释。代码修正为：

* `cm_residual_policy.v2` 将 local residual 旋转到 world 后再计算高度效用；旧 `shuffled`
  policy mode 和 v1 checkpoint fail closed。
* decision-interface v2 移除伪 `shuffled` arm，只保留 Cup、direct-Q、Cm 和 random，并将
  propensity 改为四臂值。
* residual source/probe/decision-interface 在每个 reset 后刷新当前 metadata，在 trigger
  保存 motion/start/rest 快照；DExplore 的 simulator seed 必须唯一、显式并写入 manifest
  与 records。
* 从 canonical read-only checkout 复制 `src/task/Cm/dataset` 与 `src/task/Cm/tools/data` 的
  Python 源码到当前仓库，使主 Cm 训练和数据工具入口可独立导入；不修改外部 checkout。

修正版统一使用新的 v2 schema 和 `P-20261003-...-corrected` experiment identity，避免
把修正后结果与旧 records 混合。修正后 Probe 尚未运行，因此本次只恢复代码与可复现性，
不形成新的 Cm utility 结论，也不启动 PPO。

## 成本、停止条件和边界

本次修改是可逆的代码/文档操作，不扩大 GPU、时间或数据预算。下一步只允许先做小规模
engineering smoke 和 corrected Probe；只有同时满足 action ranking、action coverage 和
真实局部效用门，才重新讨论策略训练。旧 v1 结果不能再升级为科研结论。
