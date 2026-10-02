# Decision Memo：把 Cm 变成短期候选动作排序器

当前需要决定的问题：Cm 是否值得继续作为策略输入，还是应当退出策略决策路径。

关键证据是当前 direct-Q 终点 222/384，高于 Cm 和 dynamics-off 的 188/384；现有
Cm 只作为 actor 辅助信息，未形成可检验的相对动作优势。物理价值模型已有冻结的
direct-Q、V 和三成员动力学 checkpoint，但没有测试它们对同一接触状态的候选排序。

选择执行一个小型真实候选面 Probe。固定 rotation-Cup 为 `pi0`，候选为 Cup 动作
周围的 37 个 `±0.1` 局部动作；冻结现有物理价值 checkpoint，分别计算 direct-Q 和
短期 `local reward + gamma * Q_base(predicted state)`，Cm 分数使用成员均值减标准差，
不活跃时回退 Cup。每个接触状态在完整候选面形成后，按固定概率随机执行 Cup、direct-Q、
Cm、打乱排序和随机候选之一，记录 10 步真实局部结果。这样能用同一状态类的 IPW
比较，且不扩充普通 Cm 数据、不启动 PPO。

最便宜的判别方法是 96 环境、每个 clear/non-clear 状态各一个窗口，优先检查：
候选排序是否有可用 margin、Cm 是否改变足够多动作、改变臂的真实 10 步局部效用是否
为正。支持不足记为 `UNCLEAR`；有支持但 Cm 不优于 Cup/direct-Q 记为 `UNPROMISING`。
只有三项同时通过，才讨论训练策略；否则关闭此接口路线并保留现有 direct-Q 结果。

预计成本为一次冻结模型 native Probe（不超过 240 秒）和 CPU 统计。代码、checkpoint、
环境和 motion 均只读；输出写入本仓库专用 research/output 子目录，不覆盖现有产物。

结果：`r1` 共 212 个完整窗口，五臂支持 39/37/47/47/42，Cm 在 47/47 个 Cm 臂窗口
改变了动作，说明接口被真正使用；但末三步最小正抬升的 Cm-Cup cluster-bootstrap
90% 下界为 -38.27 mm，局部 reward 下界为 -0.965，均未通过正效用门。该路线标记
`UNPROMISING`，不启动 PPO、不扩大普通数据，后续回到更高层的 representation/claim
复盘。
