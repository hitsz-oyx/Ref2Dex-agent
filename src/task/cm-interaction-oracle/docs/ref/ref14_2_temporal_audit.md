# ref14_2 P1：固定短 Y 的 temporal audit

2026-10-05；Task `cm-interaction-oracle`；实现路线 `agent/cm-interaction-oracle`。
本独立诊断属于原 `P-20261005-rolling-gt-y` 的追加离线分析，沿用其保存输入与
固定合同，不领取 seed、不创建新科学实验。判断为 **UNCLEAR**：信息归属较清楚，
连续化是否增加可部署决策价值仍未识别；不升级任何 Gate 或全局 claim。

## 问题、最小设计和资源

服务于 ref14 的 rolling GT-Y 必要性链：区分初始 ties 更符合尚未覆盖失败、
二值阈值压缩，还是没有真实干预效果。结果决定后续应先保持当前 Y 推进 oracle，
或另立协议审查表示分辨率；当前不根据 test 调阈值、改标签或训练模型。
最便宜方式是复用 ref13_1 的 `rolling.npz` 和 `result.json`，逐 channel 统计
首次分叉与 failure lead-time，并只读已完成 rolling group 的计划记录。
这是 Decision 诊断，非 Validation。

设计先发送给 root 后执行，父任务授权 CPU 统计 <=10min、<=200MiB。
实际主计算约 1 秒、两个 JSON 合计约 1.1MiB；没有 GPU、神经计算或 PhysX。
运行身份 `ref14-temporal-20261005-e`，实际 git commit、命令和输入/代码 SHA256
见 `outputs/cm-interaction-oracle/ref14-temporal-20261005-e/manifest.json`。
主结果见 `outputs/cm-interaction-oracle/ref14-temporal-20261005-e/result.json`。

## 固定合同与统计边界

Y8/U/Z 保持原定义。重算 Y 和 risk 与保存数组严格一致，U 也严格一致。
局部16/32 head 分别读 future steps9..16/9..32；failure 只在当前 at-risk 时激活，
局部 contact-lost 计数从0重新开始。channel 分叉阈值沿用 `.02`，没有优化阈值；
初始 utility tie 使用精确相等。对 Z-discordant pair 按 good−bad 方向比较，
failure channel 翻转符号。主要结果要求两条 factual 路径当前都 at-risk，
并且 query 严格早于 bad 的首次 raw failure。首次 raw event 与
post-qualification drop 分开记录；没有事件保持 null，不伪造 lead-time。

主要 cadence tau=0,8,...,56，必须有完整32步 future。dense tau=0..58 是既有
边界诊断，独立输出，不代替主 cadence。后续不同 candidate 已有不同 H；
这些数字描述 factual-path observability，不能形成共同当前状态的动作收益。
124个 pair 不独立；原32 anchor 又属于4个 shared-solver group，不作独立样本推断。

## 结果：32-step contact/held 比 height-failure 更早分叉

共124对最终 Z 不同的 candidate。50对初始 utility ties 全部同时是完整 Y8 ties；
这50对 bad 的首次 raw failure 均在 step32 之后。主要 cadence 上46/50可正确分开，
utility 提前29–47 steps，中位34，与原 ref13_1 数字一致。

| Channel | 全部124对：正确分叉 | 初始50 ties：正确分叉 | 全部正确分叉的 lead 中位数 |
| --- | ---: | ---: | ---: |
| contact16 | 116 | 42 | 18 |
| held16 | 116 | 42 | 18 |
| failure16 | 89 | 23 | 11 |
| contact32 | 120 | 46 | 31 |
| held32 | 120 | 46 | 31 |
| failure32 | 120 | 46 | 26 |
| height_failure32 | 36 | 20 | 23 |
| height_fraction32 | 44 | 28 | 25.5 |

50个初始 ties 中，46对的最早正确分叉同时包含 contact32/held32，11对同时包含
failure32，4对同时包含 height_fraction32。归属计数可以重叠。
这批样本中，没有满足主要 eligibility 的反向 channel 分叉。
contact16与held16、contact32与held32计数相同，表明此样本的 channel 信息高度重叠；
不能由此断言其他分布中 held 恒等于 contact，或 contact proxy 是稳定抓取充分条件。

dense诊断把剩余4对也分开，50/50，utility lead32–51，中位36。
典型 env36 baseline vs thumb− 首次 utility 分叉 tau57，bad failure step89，lead32；
主要最后 query56 的 future 只到88。query64必须有96步数据，旧90步轨迹不支持，
所以这4对的主 cadence 无信号应记为 horizon/cadence 边界，不能记为 short-Y 无效。

## Tie 与连续化：兼容解释，不作因果归因

初始50对 ties 的连续 height 诊断（local9..32 的最小离2cm高度margin、32-step
height change、最小逐步垂直位移）全部已有至少一个数值差异。分叉仅采用工程
数值公差1e-9米，没有拟合“有用差异”阈值。这说明二值标签确实压缩了一些
物理变化，但不同高度也可能同样安全，不能把连续差异当正确风险排序。
逐步垂直变化以米/控制步计，不将未知控制时长硬换成 m/s。

同时，所有50个 bad 的 raw failure 都在初始32-step horizon外，支持 horizon
覆盖不足的解释，但两种解释在这批保存轨迹上无法唯一分离。因 Y 的 late-head
只读9..32且 contact-loss有局部计数，failure进入几何 horizon也不保证每个 head
立刻变化。组合 utility 的 tie 与完整Y tie均记录，避免混淆 utility相互抵消。

此输入只有 height 和二值 forcepair proxy，没有逐步连续 force/proximity 或
完整 hand-flow，所以不能归因为“动作真的没影响”“实际 flow 太相似”或
“连续 contact margin 更早有效”。本次另读P0资产补充下述same-state连续诊断；P2 flow审计可补充运动多样性；
未记录的 surface flow 仍为 missing。不得回填、拼接或按最终 test Z 搜新Y。

## 当前 rolling 完成组快照与连续 contact 诊断

只读取 `s263-g0-result.json` 及其12个 frozen plan，7 anchors，tau0..88。
计划中的固定Y→U和 baseline-first argmax严格重放；所有12 rounds均有7候选。
各round全utility tie anchor数为4,4,4,5,4,6,5,3,2,2,4,6，且对应完整Y tie数相同。
因此该完成组没有观察到只因 utility抵消的全候选 tie。

这些 round 内候选才来自同一当前状态；跨 round H 变化，不计算“同一候选”的
首次时序分叉或共用某条 candidate 后续失败。此快照不重读/认证大panel的hash，
只有completed group/plan的输入hash和上述标签算术重放；raw prefix/PD/PhysX验证
仍由主运行审计承担。未完成组、`interrupted-attempts`和剪枝缺失标签不纳入。
工具可再次指向新唯一run-dir补入后续完成组，不覆盖旧快照。

另读 P0 `ref14-assets-20261005-completed-groups-v3/s263-g0-asset.pt`，
[字段合同](ref14_2_data_contract.md)与其manifest保留上游原panel审查。12轮共有49个
utility-tied round×anchor（重复观察，不是49个独立anchor）；所有49条观察均有候选间
late9..32 min-height-margin、平均hand-force norm和最小body-object proximity的数值差异。
工具保存每个tied观察的spread，仍仅用工程数值公差1e-9，不拟合正确风险margin。
这排除了这些连续观测“数值完全相同”的解释，但没有证明更强的Y排序、提前量
或物理contact/friction安全性。32-step candidate无90-step Z，不能对这些候选建立
成功/失败归因；cross-round lead-time也不能从候选branch拼接。

## 交付与下一步边界

工具 `tools/audit/audit_y_temporal_channels.py`；4个synthetic语义边界测试通过：
pre-event/joint-risk、无事件删失、正确/反向与epsilon严格边界、late-window高度
和逐步单位。首次工程执行a因 float64重算plan U与原float32不位等失败；修复为
保持float32后b成功；c追加独立dense诊断成功，d补continuous资产，e使用最终P0 v3资产重跑成功。没有污染原科学数据或标签。

保持当前短Y/U，继续父任务 ref14_1。连续量仅作下一阶段协议输入；若要比较
新的连续Y，需要另行冻结目标与same-state评估协议。不得据这里的数字启动训练、
解释rolling policy Z增益、替代主控制gate或宣称全局 Cm causal utility。
