# 负结果审计结算：一条窄路径（机主13:0x v2）

**结算：已修已定位的实现/合同缺陷；物理抓取—抬升有source-PD阳性，但当前GT-τ executor仍未资格。
停止扩展本轮审计，正式RL三臂保持HOLD，不启动v6大训练。** 不能从此推出路线或预训练无效。
差异、行号、命令与待裁决项见[共同复核包](fix-review-20261022.md)。

| 外部修法 | 独立核实与落地 | 验收数字 |
|---|---|---|
| A 完整任务与训练mask分离 | 从完整实录重建501份，prefix不改TaskSpec；新增annotation_invalid及原因，未定义/无效任务禁止进入完整任务评分。raw_invalid与训练dropout分开 |501/501重建、原终点/事件/source-frame一致；旧64 UNKNOWN逐例解释并恢复62R/2H。原始force观测无效0/501；11条计划意图冲突另列，完整任务正控认证0，未宣称其已可执行 |
| B 编码/安全/成功三信号 | 三臂复制路径已补修；encoder失败τ invalid/不终止，同executor固定state-only模式；真实CAD/table穿透与实测有限位关节检查独立；后执行状态更新ever/final |实际A/B/C各20controls，启动+注入共7次不可编码均继续；C实际WAM覆盖13/20=65%（含故意outage），额外编码终止0；安全错误仍触发终止的负控制通过 |
| C1 已有v6 adapter | **确认已有**21/11、坐标、goal/event、A/B/C接线。此前“未接线”仅指官方V4 PPO入口；解读为全仓没有v6接线不成立 |真实OnlineV6Runtime.predict接口回归通过，故意错21点映射/丢event均被拒；实际A/B/C共60controls工程smoke通过，其中C路step11000/WAM→旧R→物理20controls |
| C2 合成HOLD任务 | **确认**reset另建当前pose+8cm、精确位置HOLD1s；原始TaskSpec保留为来源，不覆盖文件；新任务身份/encoder/safety/reset策略绑定新run与resume |结果明确source_full_task_success=null；旧A/B训练交互286720/149504受旧reward/终止影响，不能事后改分洗掉或续作新协议 |
| D/E 先判据、再单轨迹校准 | 采纳；同Scheduled686架构、同GTτ通道、只80条新source-PD状态动作，2000updates/30min硬上限 |2000updates、5.376s；动作RMSE0.103540→0.001849（下降98.21%）。未加RL、DAgger或WAM训练 |

本次校准的**独立窄任务**是上升≥5cm、真实force/耦合、离桌且连续elapsed≥1s；
它不认证三臂精确8cm终点，也不认证501完整RELEASE任务。固定一条`cylindersmall`已见序列，
9个公共初态字段逐数组差0；重复不是独立样本。

| 不可混同的执行链 | 窄任务成功/复跑数 | 最长真实连续持握 |
|---|---:|---:|
| source-PD（合法动作，clipping0） |1/1|1.000s|
| GT-τ→原R |0/2|0s/0s|
| GT-τ→校准R |0/2|0.633s/0.467s|
| WAM-τ11000→原R |0/1|0.9667s，仍未成功|
| WAM-τ11000→校准R |0/1|0s|

总校准1488物理controls，另三臂工程两轮120，共1608；全部自有进程退出，GPU4/5/6释放。
原/校准GT链皆未获得阳性见证，禁止用source-PD的1/1替代。校准两次在GT仍有效时已于
第66/64帧失持；随后参考81状态结束、70controls有有效GTτ，其余进入预注册fallback，
限制后段归因。另有真实有限位关节超界，已保留具体轴/位置/边界，未放宽安全阈值凑成功。

**自己的判断**：历史结论首先被TaskSpec语义改写和voxel误终止破坏；后者占1570结束回合52.27%，
不能等同52.27%可回收步数。修后窄路径当前阻塞是GTτ→R的物理资格，追加v6预训练不能替代它。

**共同把关未决项**：raw force事件不是独立计划意图；11条rawH/referenceR及候选进度撤回
尚未裁决。可见初态相同仍有GPU物理后缀差异；无τfallback稳定性、完整RELEASE、独立泛化/
样本效率均未认证。本轮窄审计收口；这些不继续扩成今天必须解决的研究假设。
下一步若获新实验预算，先补更长、完整末端的同轨迹参考并做可达的GTτ闭环资格；
原2000update预算已经用尽，不自动加训。正式RL入口现要求匹配任务和executor的物理资格
receipt，source-PD和GTτ两链≥95%，WAM结果单列；本轮不生成可放行receipt。

证据目录：`outputs/ptv3-shared-wam-v6/negative-audit-v2-20261022-r1/`（501逐例/回归/mutants）；
`single-tau-calibration-20261022-r1/`（权重、7条物理trace、独立重算）；
`samples-signals-native-20261022-r3/`（运行前冻结源码的真实三臂入口smoke）。本轮最终64项回归、21个故意源码回归
均被拒，说明这些特定错误可检测，不代替实现正确或物理资格。
