# V1.41 接触后抓握增强 / CmLite 门控

- date: `2026-09-23`
- branch: `agent/v141-contact-reflex`
- run_status: `COMPLETED`
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

四个未见 seed 的全部三臂均 `COMPLETED`。按
seed91/92/93/94，off为52/52/55/49，always为
51/54/52/52，cmlite为53/55/50/52；合计分别
208/209/210（分母均256）。Cm 相对 off 共40修复、
38破坏，仅净增2，seed93反向净损失5。所有输入
起始帧/轨迹配对64/64，不是初始化错配。故 V1.41
预注册的稳定及 Cm 一致增益假设均被否定，停止扩大
此冻结模型和门控参数的评估。
