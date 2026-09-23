# V1.41 接触后抓握增强 / CmLite 门控

- date: `2026-09-23`
- branch: `agent/v141-contact-reflex`
- run_status: `RUNNING`
- evaluation code commit: `0a0cfa2`
- official actor checkpoint: `null`

实验设计与预注册停损门槛见对应 experiment card。准备
在现有路由评测器中加入显式关闭、无模型增强和 CmLite
门控三种互斥模式；默认关闭，确保 V1.40 路由行为不变。

seed90 三臂均已完成，64/64环境按起始帧和轨迹编号配对：
`off` 46/64，`always` 47/64，`cmlite` 51/64。
相对 `off`，`always` 修复12/破坏11；`cmlite`
修复16/破坏11，动作改写率7.79%（always31.39%）。
Cm 臂比 off 净增5/64，且高于 always4/64，恰达到
预注册 heldout 扩大门槛。此 seed 是筛选集，不能算
泛化证据；接下来按冻结参数在 seeds91–94 测三臂。
