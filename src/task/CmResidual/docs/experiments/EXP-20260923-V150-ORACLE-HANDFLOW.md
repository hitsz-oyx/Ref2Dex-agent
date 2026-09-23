# V1.50: 原版 Cmv2 一步失准是手流输入还是表征问题

- experiment_id: `EXP-20260923-V150-ORACLE-HANDFLOW`
- branch: `agent/v150-oracle-handflow`
- run_status: `NOT_STARTED`
- conclusion: `INCONCLUSIVE`
- official actor checkpoint used: `no`

V1.49 在自训练 s3 PPO 接触转移上，原版 Cmv2 V1.3
使用动作映射出的**名义 PD 目标手流**时平移 EPE
为39/53mm，明显差于零位移和 CmLite；contact token
却激活>93%。当前不能分辨是原版表征/跨手泛化差，
还是一个控制步内实际手并未移动到 PD 目标。

冻结 V1.49 完全相同的原版 checkpoint SHA256
`371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591`、
相同 seed95/96 已记录首 episode 转移及 RNG149
每 seed 接触/无接触各64个索引。参考 V1.49 card
的完整 SHA。唯一输入变量是 hand flow：

1. `nominal`：前状态+当前动作→DExplore PD 目标→
   Inspire URDF 目标手点，与 V1.49 相同。
2. `oracle`：前状态到真实 `next_q` 的 Inspire URDF
   手点差。`next_q` 是仿真后的未来状态，**只能作
   离线诊断，严禁在线 PPO 或部署时使用**。
3. `zero_hand_flow`：保持当前手/物几何但手流全0，
   分离接触几何先验与动作条件流。

三者共享同一前帧几何、模型参数、GT 下一物体位姿、
microbatch≤8/object chunk32。逐 seed/接触分层
报告物体局部平移 EPE、零物体位移 EPE、模型
预测平移幅度、手点流均值与 p95、contact token
激活和几何/模型延迟；不按结果改 checkpoint 或
样本。先8样本工程 smoke 确认 strict load/finite
与 `next_q` FK；正式全256样本。

预注册支持“名义手流失配是主要原因”的强门槛：
两个 seed 接触子集 oracle EPE 均比 nominal 低
至少50%，且均比零物体位移 EPE 低至少20%；
另报告 oracle/nominal 手流比值。若仅前半通过，
可称手流失配有影响但不足以使该模型可用；
若不通过，不能排除模型训练域/坐标/几何等其他
原因。本离线实验**无论结果如何，都不能证明 Cm
提高 PPO 抓取成功率**。

GPU5 空闲时单卡，smoke+正式预计<10分钟，新增
产物<1GB、总产物<300GB。输入 SHA、模型加载、
有限性或资源冲突失败即停；外部项目只读。
