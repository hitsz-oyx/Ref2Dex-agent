# D-20260927 — C1 观测路由 Validation 结论

状态：Decision Checkpoint 已完成；接受预注册的窄范围 `SUPPORTED`。

Root 对 [VAL-20260926-observation-six-expert-c1](../experiments/validations/VAL-20260926-observation-six-expert-c1.md)
建议 Option A。用户回复“你是主agent有最高权限，按照你的想法来”，将这次
路线选择交由 root；root 据此接受 Option A。该回复不是新的实验、资源或
更广科学 claim 的授权。

五个预留 holdout seed（400–404）的 10 个 native arm 均 `COMPLETED`，输入
hash、命令、Cm-off、64 个有限完整首回合及 paired motion/start frame/episode
length 审计均通过。每 seed 专家选择一致为 63、62、64、62、60/64，cup 为
30/30；观测路由 held-lift 为 123/320，固定物体身份参考为 118/320，前者
每 seed 均至少 16/64，且相对参考总差 +5/320。所有预声明门槛通过。
描述性的 seed-level bootstrap 差值为 +1.5625 pp，95% percentile 区间
[−1.8750, +4.6875] pp；区间跨零，不支持观测路由优于固定路由的主张。
完整审计和工件 SHA256 见[机器可读结果索引](../experiments/validations/VAL-20260926-observation-six-expert-c1-results.json)。

处置：冻结该六专家观测路由为**这 12 条 motion、五个 holdout seed 上通过
C1 路由选择与非零 held-lift 联合门槛的自训练层级 substrate**。总体
Self-trained grasp 仍是 `PARTIAL`：这里没有单一共享 GRAB actor、未见物体
泛化、相对固定路由的优势或 Cm policy utility 证据。Cm 路线保持冻结；
任何后续 Cm 方案仍需新的高层决策和独立的 matched Cm-on/off 验证。
