# V1.50: 原版 Cmv2 一步失准是手流输入还是表征问题

- experiment_id: `EXP-20260923-V150-ORACLE-HANDFLOW`
- branch: `agent/v150-oracle-handflow`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（oracle 手流能显著缩小误差，但不足以使原版权重优于零位移）
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

## 完整256样本结果

代码 commit `fd89143`，8样本 smoke 和正式256样本
manifest 均 `COMPLETED`。两 seed 各自固定64接触+
64无接触索引，与 V1.49 完全相同；两次模型输入
除了手流外共享同一前状态几何、原版权重和 GT。
接触子集结果：

| seed | 手点流均值：名义/真实 | 零物体位移 EPE | Cmv2 名义 / oracle / 零手流 EPE | oracle 降误差 |
| ---: | ---: | ---: | ---: | ---: |
| 95 | 52.8 / 8.2 mm | 5.36 mm | 39.05 / 9.02 / 8.10 mm | 76.9% |
| 96 | 57.1 / 12.0 mm | 10.35 mm | 52.74 / 14.60 / 12.72 mm | 72.3% |

名义手点流约是真实一步的6.5倍/4.7倍；其
p95为147/150mm，真实 p95仅19/37mm。
oracle 手流在两 seed 都把模型 EPE 降低超过50%，
证实名义 PD 目标到真实一步执行的手流错配是
重要原因。但 oracle EPE 仍**比零物体位移差68.2%
和41.1%**，所以预注册强门槛不通过。
零手流 EPE 8.10/12.72mm，也未优于零物体位移，
提示冻结原版权重/几何域本身还有偏差；不能
把全部误差归给 PD 控制滞后。

无接触子集名义→oracle EPE 为11.10→8.67mm、
8.00→6.07mm，仍分别高于零物体位移7.61/
4.85mm。全样本名义→oracle 为25.07→8.84mm、
30.37→10.34mm，仍高于零物体位移6.48/
7.60mm。上述 oracle 使用未来真实 `next_q`，
**不是可部署输入，也不能支持在线 PPO 增益**。

结论边界：原版 Cmv2 在这一 s3/Inspire 转移上
的失准有明确的手执行动力学因素，但即便给它
真实下一步手运动也无法超过零预测；下一步
应考虑从当前 PPO 物理转移做 Cm 适配训练，
再用完全 matched 的 Cm-on/off PPO 接入检验。
证据在 `outputs/CmResidual/agent_v150_oracle_handflow{_smoke,}/`。
