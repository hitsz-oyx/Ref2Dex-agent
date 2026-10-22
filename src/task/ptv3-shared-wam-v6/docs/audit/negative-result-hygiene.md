# 负结果证据卫生：七类必备协议

适用于本次历史负结果审计及以后需要否决研究分支的运行。它不是新增实验授权，也不要求
为手续补齐而重跑所有旧实验。优先修正已有结论范围；必要证据若缺失，结果降为待定或
实现/协议局部现象，不能把缺口自动升级成下一轮训练。

## 1. 否决范围

在看结果之前写明被检验的最窄命题和可改变的路线决策，例如“固定checkpoint+合同+预算
在这条序列没有达到hold45”，而不是“τ不可用”“v6失败”“RL无效”。同时写明结果不涉及
哪些主张：执行器资格、生成器准确性、完整任务完成、接触持握、样本效率不能互相替代。

结果归档为可同时存在的三类：

- **实现无效**：目标方法没有按合同真正执行。移出方法比较，保留故障轨迹与修复证据。
- **协议未合格/不可比**：执行真实，但正控制、初始化、评价可达性、预算或统计单位不足。
  保留描述性数字，不能产生原本的方法否决。
- **在声明范围内的真实负结果**：实现和相关控制合格，未达到预注册目标；只停止该配方/
  预算/分布的投入，不自动否定整个科学主张。

后验诊断不得偷换初始gate；要改指标则并列原gate、新诊断及改动原因。对多个指标分别判定，
不能只留一个UNPROMISING标签。不同WAM版本、executor身份、任务或动作合同分别否决。

## 2. 实际执行身份与入口

每个参与结论的run写清：experiment_id、run_id、实际代码commit、实际argv入口、冻结目录，
runtime解释器与包版本、模型schema/checkpoint/step、BC及预训练继承、数据split、物理backend，
以及变换/normalizer/手序/控制频率。实验阶段、实现版本与run身份不要复用同一连续编号。

身份核查从**运行manifest/argv → 当次冻结入口 → loader → checkpoint**正向追踪：

- Task目录的v3不等于WAM v3；Historical PPO的`V4Runtime`不能被今天的v6工具替代。
- checkpoint由BC config间接给出时，也要展开真正路径；记录step0 DEV-best与final不同。
- DDP的多个rank是同一模型的执行拓扑，不是多seed复现。迁移CPU/GPU PhysX必须显式标注。
- 卡头更新提交、运行时HEAD、主报告HEAD、后来修复提交不一致时全部保留，并说明哪个
  证明了什么。没有保存旧源码，就写无法完整认证；不能把main修复倒灌到旧commit。
- stale顶层RUNNING与最终rank STOPPED冲突时，依最终run/child产物说明，不臆测进程仍活跃。

不要求重复极严格的内容哈希体系；版本受控的情况下保留manifest、冻结路径和关键载入证据
即可。不能打印凭据或把无关用户修改带入提交。

## 3. 真实入口的合同验收

验收必须命中实际runner调用的函数，不能只验证一个未被使用的helper或离线shape测试。
已有日志/保存数组能回答的先CPU核对；不能回答的列UNKNOWN，不擅自再开新方法实验。

- **τ/goal**：手点顺序、世界/物体坐标、reference-current/live-current原点、未来tick间隔、
  startup validity、越界/末帧padding，以及horizon是否被mean-pool。训练的t+24实现终点
  与空间waypoint必须区别；同shape不能通过语义验收。GT与pred对照的startup要一致；
  额外oracle单列，不能混为单一变量对照。
- **动作**：十二active到十八native的唯一展开点、wrist增量与finger绝对目标、clamp/scale/
  coupling只发生一次；query动作须用该learner state的q构造PD。实际applied、PD检查与
  episode ID应能对齐，不只比较网络requested输出。
- **时间/重置**：每条transition用source episode观察产生动作，在同episode执行，再记
  terminal reward/done；reset清理对应history/reward/previous，不伪造H4；reset不是隐藏
  physics step；force reset帧invalid，post-step force才可用。天然终止与强制回放越界区分。
- **RL**：真实post-control observation bootstrap、terminal mask、GAE、旧策略logprob重放、
  policy/value更新与optimizer恢复一致；KL回退要恢复weights和moments，不能只改LR变量。
- **评估**：实际执行最后一个动作后的观察可计结果，但最后一帧未执行的decision不能计
  action clipping或训练交互；有干预的rollin不能充作自主policy evaluation。
- **模型/数据**：checkpoint真正加载，normalizer与几何来源一致；loss-only未来标签不能
  泄漏进在线输入；WAM无动作条件时不能冒充候选动作反事实模型。

每项写实际证据路径、通过/失败/未测及适用run。旧运行未通过后，修正版另立run，保存旧结果。

## 4. 与被评指标匹配的正负控制

控制的指标、初态、动作/物理/任务时长要与结论相符：

- 要否决完整placement，正控制也须完成placement；teacher仅hold45合格不能给完整任务
  校准背书。若teacher placement门槛未过，则完整gate不可判，learner hold45缺失仍可描述。
- 要研究预测τ执行差距，先验证同一已适配executor能用可执行机器人GTτ；原动作replay
  是执行环境正控制，GTτ是执行桥正控制，二者职责不同。GT也失败优先说明桥未合格，
  不能因此判生成器主瓶颈；同理GT通过不等于任务样本效率已改善。
- 所谓hold须核实际定义：目标附近连续位置、lift-only、手物接近、实测指force、物体
  unsupported、长持握、完整放置均应分别命名。ballistic抬升是hold代理的负控制；
  恒定几何目标是稳定阶段预测/奖励的强对照。仅指标变量名含hold不算抓持认证。
- B几何与C预测臂使用同一可用几何/goal/H4、奖励、BC、预算及初始化；弱B或不同分辨率
  不能支持“胜过强几何基线”。压缩/no-prior必须从训练起适配，推理置零仅是OOD诊断。
- 奖励随真实next-state变化；如果WAM信号几乎恒定或未启用，负结果只约束该奖励接法，
  不能否决模型表示。原位静止baseline、未准入计数、启用覆盖及WAM与几何信号差异应可读。
- 显式teacher warmup/β混合、oracle目标、训练态teacher forcing分别单列；含干预保持率
  无论多高都不是自主执行成功率。

UNKNOWN必须有自己的分母，不能编码成0误称无穿透/无扰动恢复；没有loss opportunity时
自然恢复率也是UNKNOWN。关键正控制未合格就停止方法归因，先保留真实轨迹及失败范围。

## 5. 影响面清单

每个发现必须列“受影响run / 不受影响run / 受污染指标 / 保留事实 / 应撤销推论 / 补救”。
尤其区分：

- 源坐标错误污染整个teacher校准；runner末端reset错误污染对应片段；producer保存失败
  不等于策略物理失败；丢失milestone导致queue FAILED不等于学习无收益。
- 接口修复只影响新run；历史轨迹仍可作故障行为描述，但不能进入清洁方法汇总。
- gate复合失败时逐项保留：例如15/16 hold45而clip失败，不能改写成不会抓取。
- 旧版本训练return、后来版本deterministic eval不能作为单变量改进配对；重置状态也变了
  的前后比较不具因果单变量资格。
- phase62已经持握的reset能证明保持/恢复，不证明frame0起抓；分phase给出，不混成总体收益。
- 只有offline PD/动作误差的卡不能影响“自主执行是否成功”；只有旧v4 PPO的卡不能影响
  “v6 PPO是否失败”；只有预测误差/mesh proxy的卡不能影响物理可控性结论。

对每条历史卡优先追加scope修订，原始数值、run路径、旧状态不覆盖；归纳主报告链接到该清单。

## 6. 统计单位与可比性声明

每个指标同时写清分子/分母、单位和独立性：物体/序列/capture/窗口/点帧/episode/训练seed/
采样draw/并行rank不可互换。多环境复制一条motion不是多条未见序列，同序列重叠窗口不能
当作独立样本，K8噪声抽样不是八个模型，三DDP rank也不是三个训练seed。

声明TRAIN/DEV/TEST分界及选checkpoint规则；没有祖先预训练manifest时不得写终身未见。
同面板已经反复探索则称DEV/probe，不变成新TEST。选择成功teacher参考的资格面板须报告
完整候选数和筛选率，不能只报被筛选面板的高成功率。

配对比较须核实际首帧而非只比seed：q/dq/object/hand/sequence/物理观测、tick、table、
force-valid、episode等。跨checkpoint/arm使用共同初态；失败的初态配对标记不可比，不静默丢除。
有动作干预、后端迁移、任务时限或reward不同时，表中明确不能直接相比。

样本效率的横轴用实际环境control transitions，native.steps×envs为物理成本总账；
STOP前已采集而未更新的rollout也计成本，评估交互另账。达到成功阈值要求预先约定的连续
评估达标；未达写右删失，不是0步。缺seed/C臂/checkpoint不填0，也不计算完整AUC或宣布
无收益。示教、预训练、BC、几何优化与推理耗时各自列账，不能把继承成本写0。

## 7. 停止条件

**本次审计**：只CPU读取/汇总/修改scope，不启动新方法、GPU、仿真或训练；不改变原数字；
不杀他人进程、不碰凭据，不以“还差一个实验”绕过owner HOLD。缺证据就报告读不到及影响。

**以后重新获准的实验**：卡内先定设备、实际运行环境、wall/interaction/checkpoint/disk
上限、资源被占的处置、关键合同失败即停条件。新run开始前重新核设备，受保护GPU绝不迁移占用。
只用自己的有界STOP/保存退出机制；保留已有checkpoint，不覆盖失败证据。

科研停止和工程停止分别记录：合同错误/正控制不合格应停方法归因；数值非有限、reset/动作
错配、磁盘/时间边界应保存中断点；合理负结果达到预注册判断后停该配方。预算不足不是
已证明无效，工程修复也不是研究收益。最终报告须写“已完成 / 未测 / 中断 / 可保留结论 /
应排除结论”，避免由单一状态标签代替证据。
