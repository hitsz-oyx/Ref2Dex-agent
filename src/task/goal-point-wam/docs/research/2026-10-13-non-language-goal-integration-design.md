# 非语言物体运动 goal 在当前系统里的实际接法

日期：2026-10-13。状态：源码核对与接入建议，**尚未实现新模型或训练**。
用户要的是直接声明物体期望运动，由策略产生手动作；允许参考示范。
本分析针对现有 `goal-point-wam`，不将其他任务或作者机器人策略默认为兼容。

## 结论与当前缺口

应借鉴 Dex4D 的配对目标表示，继续使用我们的数据加载、手点输出和机器人R。
COIL的时空目标编码可作为后续短序列版本的参考；暂不搬它的Panda feature
extractor、action head、action normalization、仿真tracker或整个训练器。
Im2Flow2Act的2D/固定相机与当前3D/masked hand合同差别更大，不作为第一改法。
来源与实际组件检查见[源码兼容性报告](../../../../../paper/point-motion-goal-policy/egodex-compatibility-summary-20261013.md)。

当前三个问题必须分开：

1. **视频训练尚缺任务目标。**已审计的22个原始compact窗口 `object_goal_valid`
   全空。其World仍有scene-point监督，Actor仍有手轨迹监督，但这不能训练
   “不同物体goal决定不同手行为”的视频条件关系。此事实不能解释所有训练负结果。
2. **审计goal尚未进入训练。**`object_motion_goal.py`产物不是当前loader/model
   消费的合同。仅在output目录写goal文件或更换编码器，不会使远端旧数据获得目标。
3. **执行还需机器人桥接。**人体视频的未来手标签不是机器人command；当前机器人
   端已有独立适配和Chunk-R，训练/缓存与human预训练之间不是自动同步的。

## 真实 loader 的可达性检查

使用已有r03真实clip与goal，在CPU调用原版 `VideoPointWindows(engineering=True)`
和 `video_collate`；临时view只软链接原clip，原文件字节保持不变。
没有模型推理、训练、伪造机器人状态，未将审计样本升级为训练数据。

| 操作 | 实测 |
|---|---|
| 读取原clip | batch中有效object goal点0 |
| 把pack函数的字段直接加入sample后collate | `desired_object_points/current_object_valid/desired_object_valid/goal_offsets/goal_times_s/selected_query_indices`全部被丢弃；旧goal有效点仍0 |
| 显式映射末端到旧 `goal_object_points`，并设置旧mask | 24个有效点进入原batch合同 |
| 原人体监督/World action | 分别 `[1,24,2,11,3]`、`[1,24,22,9]`，均保留原含义 |

记录：`tmp/point-goal-integration-analysis-20261013/loader-goal-reachability.json`。
**r03是静止底座，不是合格运动任务样本**；24点进入batch只证明字段连接可行。

## 最小第一版：可靠对象终点 + 显式点对

暂保留当前0.8秒Actor标签/输出，先回答“对象目标与点对应是否实际有用”，
不同时引入长目标时间、序列编码和新的机器人执行器。

### 数据

- 输入保持当前scene、H4手历史。
- 明确当前目标物体/部件mask，不能用最大未来位移自动推断任务身份。
- 在当前目标内固定点索引；从同一轨迹取该点+24位置，写入旧字段
  `start_object_points/goal_object_points/object_goal_valid`。
- mask为当前点已知且目标标签已知；失踪目标不补静止位置。选择当前点时不根据
  未来可见性筛点；缺少未来标签只屏蔽该goal pair。
- 原生三域已有同canonical point配对目标，不需要EgoDex重建即可先测试编码。
  但source anchor是否真是操作目标仍须使用已有target-relevance审计选择panel。
- EgoDex只有验证过的运动对象窗进入goal-conditioned Actor对照；null-goal原样本
  可以继续用于既有World/无目标辅助监督，但比例要固定，不能当作有目标样本。

### 编码与模型接入

每个对应点构成一个token输入：

`feature_i = [current_xyz_i, desired_xyz_i - current_xyz_i] / geometry_scale`

共享MLP映射到128D；保留逐pair mask与始终有效的null token，再交给现有
trajectory decoder的 **memory** 路径。同步重排point pairs不应改变预测；
打乱两侧配对会改变任务含义。这里使用delta是本地选择，Dex4D源码用current/target
拼接；两者保留配对，具体哪种更好尚未实验。

当前默认endpoint把两个无序集合分别tokenize，缺少显式pair身份；这已被排列
检查确认，但不意味着它在非对称几何上完全没有位姿信息，或导致所有失败。
`start_delta`已有隐式对应信息，应作为可选低成本对照而不是忽略它。

原decoder的memory可接不同token长度。`query`路径则假定前半start、后半end；
N个pair tokens即使碰巧通过偶数长度校验，语义也不满足该假定。因此第一版固定
memory，不将新的pair tokens直接塞入旧query pooling。

新goal MLP参数与旧3D MLP不同，不能严格加载旧goal权重。应明确迁移现有scene/
decoder/World权重、初始化新goal encoder，并给予encoder和decoder短适配。
原checkpoint与旧默认路径保留；不能只换class后沿用旧缓存宣称升级成功。

## 第二版：多时刻目标，保留物理时间

只有第一版出现可用信号后，才将同一点在多个waypoint的目标作为序列输入：

`token_(h,i) = MLP([current_xyz_i, desired_xyz_(h,i)-current_xyz_i]) + TimeEmbedding(tau_h)`

`tau_h`是相对当前的秒；point valid和goal-time valid分别保留。先用+8/+16/+24
（约0.267/0.533/0.8秒）。这些数字已经存在于审计packet，但尚未接入模型。
COIL的有效槽顺序编码不是严格秒数，所以不能直接拷贝来兑现此时间合同。

原生数据由 `effect[slot,7/15/23]` 作用于同组当前canonical点产生三个目标。
视频数据从同ID `point_target[7/15/23]`产生。机器人数据从预先给定参考物体
位姿作用于同canonical点产生；不是自主rollout实际未来物体或未来手。
point pair token经time编码后仍通过memory进入Actor。

需要改的实际边界：

| 位置 | 必要改动 |
|---|---|
| `tools/audit/prepare_video_points.py`及新目标准入 | 生成有对象/部件身份的目标；与原World标签分别声明可用性 |
| `src/goal_point_wam/video_data.py` | loader读目标packet，collate保留desired/masks/time；不能只放sidecar而不读取 |
| `src/goal_point_wam/data.py` | 原生多时刻goal，保留原手标签和当前target选择 |
| `src/goal_point_wam/model.py` | 成对/时序GoalEncoder与Actor goal分派；原query合同不能默默沿用 |
| `src/goal_point_wam/point_supervision.py` | forward向Actor传新goal，World仍只读scene/action |
| `tools/run/train_video_points.py` | 新checkpoint身份/权重迁移；评估新增正确、null、static、错配goal及运动目标分层 |
| 机器人sample/server/cache | 根据已知plan生成同合同goal；显式重编码context，再适配robot actor |

这些是待实施改动清单，不是目前已经完成的功能。

## EgoDex能训练什么

训练数据的一行是：

`(当前scene, 当前/过去手, 声明的物体目标) -> 演示未来手位移`

EgoDex原生手标签足以监督该**人体轨迹Actor**，不需要编造机器人qpos或动作。
目标可从演示未来物体运动hindsight构造；部署时必须由用户/参考/规划提供同格式
目标，不能偷读实际执行未来。Human Actor不接受未来手，未来手只作为loss标签。

视频World继续学习：

`(当前scene, 演示中执行的未来手轨迹) -> 实际未来scene点`

训练时未来手作为动作条件是动作条件模型的监督合同；部署若用World评价动作，
必须给候选手轨迹。期望物体目标不能直接喂给World当实际后果。
两者共享backbone，若以后训练backbone，World参数/输出可能随训练改变；
goal-blind只约束固定模型的输入依赖，不保证跨训练版本输出不变。

当前没有足够经过确认的EgoDex运动目标held样本。标签缺口不是新编码器能修复的。
目标内补点/重跟踪和held-session覆盖应先完成，不能对静止底座的三个数值通过窗
做训练再声称物体运动goal有效。16点门是此前刚体标签审核配方，不是Actor必要条件；
少量可靠点可以保留mask使用，但不能伪造整物体姿态或失踪目标。

## 手远、物体静止时怎么办

短控制horizon与任务goal horizon分开。当前0.8秒内物体仍静止，并不意味着任务
是保持静止；抓起前的接近阶段应保留。若给的是短期静止终点，不能完整表达后面的
抓起意图；优先给包含后续运动的已知参考目标，而不是自动添加未来手oracle。

这需要更长目标标签。现有 `point_target[24]` 不能凭padding生成2秒后的物体运动，
`pack_object_motion_goal`也明确拒绝超出24的offset。视频需更长同ID track span；
原生完整pose序列也要检验目标时刻覆盖、mask和当前anchor。达不到则先承认该
pregrasp任务的信息缺口，不把所有静止段删掉或虚构phase。

严格时间跟踪与“达标后推进”是两种控制合同。当前系统按参考tick给目标；第一版
保持该合同。是否改为进度推进应另做对照，不能直接搬SimToolReal阈值而改变
任务时钟。对象goal通常不唯一确定每根手指的动作，仍依赖当前观察、训练中的
接触策略及机器人可执行性；这不要求强制输入人手goal。

## 从人体Actor到机器人，已有链仍要适配

机器人当前链是：

`自身scene/H4 + 已知物体goal + q/dq/上一动作 -> robot-adapted Actor -> robot link位移tau -> R(当前机器人state,tau) -> command`

模型张量都是2×11点，但human wrist/joints不等于Inspire link几何；EgoDex使用
当前相机frame，机器人使用当前物体frame。**只要求一个sample内scene/手/goal/label
共同坐标还不足以证明跨域迁移**，坐标分布和形态差异需机器人监督适配。
第一步不凭空为EgoDex估计物体6D姿态；保留其已定义共同camera参考，明确域差异。

当前robot stage冻结goal，contexts还会缓存。引入新encoder后，需要显式加载新版
权重、重算cache、用本体实际状态和执行过的action/FK重新适配actor；旧adapter
不会自动获益。详细证据见[机器人桥接审查](../../../../../paper/point-motion-goal-policy/robot-bridge-design-audit-20261013.md)。

先用同一个已校准R做比较，防止同时换R导致无法归因。但R不是永久正确：若GT
robot tau经R也失败，先处理其状态覆盖；若GT校准通过、Actor预测失败，才更能
定位到Actor/goal/访问状态分布。人体手EPE变好不能替代接触、持续抓持和物体跟踪。

## 真正能决定投入的顺序

1. **工程**：先把可靠目标映射到loader，确认goal传到Actor、mask与pair/time不丢。
   同步重排等价、null/static不同、改变goal不改变固定World输入依赖。工程检查
   不证明goal改善行为。
2. **已知标签上的编码Probe**：先用原生可信交互窗的精确目标比较旧endpoint、
   start_delta及显式pair编码，固定训练预算/数据/可训练模块。数据划分和目标
   relevance遵循原审计。正确goal需改善held手误差，特别是运动/旋转panel；
   敏感性上升或训练loss下降不能独立过门。
   不同编码模块的参数与token数也须报告；这是方法对比，不声称仅改变配对身份。
   同模型的重排/错配检查才直接核对配对机制，且没有可执行反事实动作标签。
3. **EgoDex条件Probe**：有运动目标held覆盖后，用同一新模型/数据流做correct/null
   goal短对照，保留相同World监督和null辅助比例。未验证的错配目标可作输入依赖
   检查，不能给它编造“正确手动作”反事实标签。
4. **机器人**：重新适配Actor/cache，分开校准GTtau→R和预测tau→R，再做实际闭环。
   阶段2/3出现正信号才支持这一执行迁移投入；goal不能自动解决独立R问题。

当前建议的下一项工作是步骤1和步骤2的最小合同/编码对照；不是直接恢复旧远端
长训练，也不是把作者完整策略当成即用替代。这里给的是可审查的具体接法，收益
仍待这些实验区分；本轮仅完成设计与loader工程核查。
