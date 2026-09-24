# P-20260924-contact-cm-online-down

date: 2026-09-24
branch: agent/cm-contact-aware-action-selection
classification: Decision

## Question

重新设计的动作条件 Cm（短期物体效应 + 接触保持）能否通过
选择性小幅下压，在完整 DExplore 闭环中提高 held-lift 成功率？

## Hypothesis and decision

冻结 seed151/152 训练的 raw-action Cm，固定自训练 e260 actor
与 s3 Inspire 轨迹。新评估 seed157/158 各 64 env，三臂：
`base` 不改动作；`always_down` 当前手物接触时 wrist-z −0.1；
`cm_down` 仅当模型预测的五步接触比例相对原动作增益 ≥.05，
且五步物体 z 位移预测损失 ≤10mm，才 wrist-z −0.1。

窗口固定 global steps50..150 的偶数步。先做 seed157 ×16
工程 smoke；接线/有限性过门后才做上述六次完整 Probe。
只有 Cm 相对 base 和 always_down 合计均 ≥8pp、两个 seed
都不负、且 Cm 确实选择了足够动作，才继续更大样本；
任一门失败就停止局部 wrist-z 动作接法，不用本次 seed
调 .05/10mm/窗口阈值。完整 held-lift 是主指标，reward、
接触比例和最大抬升为安全诊断。

## Minimal protocol

仅线上当前 `q,dof_vel,object_state,actor action` 输入冻结 Cm；
不使用未来状态或真实五步随访标签。raw MLP 是重新设计的
短期交互/物体效应 Cm，不再声称六区域几何结构独有。
always_down 区分模型状态选择与固定下压。独立 GPU PhysX
环境不作为逐 episode 反事实配对；这是小 Probe，不是正式
policy-utility Validation。

## Budget and stop

单 run 1 张空闲 GPU、≤64 env、<20min；九次总 <90min，
总输出 <200MB。模型/actor/motion SHA 漂移、GPU 冲突、
非有限预测、执行次数过少或运行失败即停止升级。

## Result

Status: PENDING

Evidence: pending

## Decision update

pending
