# Decision Memo：冻结强 Cup/rotation 后学习受限 Cm residual

## 当前问题

固定 `rotation_cup7` 候选已经有稳定的局部机会信号，但已有 Cm 完整动作/程序
控制没有超过该强参考。需要区分：Cm 是否能在强参考附近预测并利用小幅、物体坐标
系的局部动作后果，而不承担完整动作生成职责。

## 关键证据

HF15 rotation-cup 候选在留出 clear 状态相对 base 的支持保留抬升为
`+26.509 mm`，原候选机会门通过；HF15/HF16 的完整 Cm 控制门仍未超过强
`fixed7`，但 HF16 一步执行器条件物理信息通过了信息门。随机动作路线的线性
响应只保留小幅速度信号，不能直接承担控制。

## 选择行动

新建 `agent/cm-residual-policy` 路线。冻结 HF15 `rotation_cup7` 反馈律和所有
输入专家；先只允许三维 object-local translation residual，Cm 预测真实两步
接触/抬升/几何后果，策略只输出该 residual，并在不确定或有风险时严格回退零
residual。使用真实 native source、Cm/shuffled 同构训练和 baseline/residual/
shuffled 三臂 held Probe。

## 成本与停止条件

首个 Probe 使用单空闲 GPU、有限 airplane 源和有界 H10 native 窗口；不启动完整
PPO，不改旧 checkpoint，不追加旧 HF15/HF16 slot。支持不足即 `UNCLEAR`；支持
充分但收益或风险门失败即 `UNPROMISING`；不调阈值、seed 或 residual 范围追门。
只有所有预注册 residual-vs-baseline 门通过，才设计下一轮正式策略验证。

## 边界

本路线不改变 Mission、C3 claim 或最终稳定抓取协议。外部项目只读，所有新输出
写入本仓库拥有的 research output 目录。当前工作区既有未提交修改不纳入本路线
提交，也不覆盖。
