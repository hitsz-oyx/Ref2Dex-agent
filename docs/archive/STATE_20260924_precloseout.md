# Ref2Dex Current Research State

Updated: 2026-09-24

本文件只保存会影响下一步研究决策的当前事实。

历史细节见 experiment cards 和 Git。

## 当前决定（2026-09-24）

用户已选择 `D-20260924-after-task-aligned-option` 的 Option A：本阶段
停止新增 Cm 训练、仿真和 Probe，保留可复现实验产物，整理负结果。
单轨迹 self-trained 固定路由抓取为 307/320（95.94%），但该路由
读取仿真器起始帧选择整段使用的 BC/PPO 专家，不能代表单一观测驱动
actor 的成绩。PPO 专家使用过 CmLite 奖励训练，缺少同路由 matched
Cm-off 消融，不能把成功率归因于 Cm。Cm 对最终策略的独立、稳定增益
仍未建立；当前实验只否定已测试的具体配方与动作族。跨物体抓取迁移
仍弱，未见物体上的 Cm 策略收益也未建立。下一步仅保留未来若重启
研究时重新设计时序 credit assignment 的方向；当前不执行。
证据与边界集中记录于 `docs/handoffs/CM_CAMPAIGN_CLOSEOUT_20260924.md`。

## 跨物体研究证据（当前阶段已收尾）

用户确认“跨轨迹”指**未见物体身份**，论文核心是证明 Cm
带来稳定抓取的独立作用，而非只提高单物体 baseline。
当前任何 Cm 方案均**未证明**跨物体策略收益，不能进入正式 claim。

已构造并哈希固定的 train3（airplane/mug/toothpaste）—apple
物体不相交 Probe。自训练 e320 在 apple 仅约 2–4/64 持握抬升，
官方 actor 仅作诊断时约 56–61/64；相同初始运动下失败回合
通常能接触物体，但抓握不持续、几乎无承重抬升。接触力的
五手指链路计数虽在全回合与成功相关，**抬起前的早期接触不相关**，
因此已排除“仅预测多链路接触数量”的 Cm 目标。当前简单单步
动作效应、接触分数、短动作 splice 和轻量几何 Cm 接法均无可靠
跨物体策略增益，不能继续微调它们来冒充机制证明。

数据侧已确认坐标对齐的 Inspire 几何重定向来源有 22 个额外
S1-lift 物体通过离线基础门；旧 `inspire_rl_partial_filtered` 来源
有明显相对坐标偏移，不直接使用。经同一校正转换与仿真诊断，
cubesmall、waterbottle、alarmclock 均 64/64 官方诊断持握抬升；
torussmall 仅 6/64，暂列困难 stress case。已建立 pilot split v2：
train5 = airplane/cubesmall/mug/toothpaste/waterbottle，heldout =
alarmclock；apple 已探索，仅作诊断。自训练 airplane e260 在
alarmclock 为 0/64。train5 Cm-off e320 随后完成：train 身份
19/64，alarmclock 仍为 0/64，未过迁移门。扩展到 train8 的
六区域几何 Cm 在对象留一效应误差上仍差于 raw head；raw head
虽能预测 alarmclock 的平均 H5 效应，但不能可靠排序其状态内效应。
最新 train8 H10 物理机会筛查也只有 cup 和 waterbottle 两个对象
出现可靠的 >=10mm wrist-z 效应，且均同号，未过“至少三个对象且
正负并存”门。当前不再训练 z-only 对象选择器，转回策略状态覆盖
与 Cm 表示/信用分配的更高层问题。

随后对 self-trained train5 e320 导出完整转移并做 H10 承重覆盖审计：
airplane/cubesmall/mug 通过预设样本门，toothpaste 只有 2 个环境、
waterbottle 只有 28 个正窗口，最终 3/5 < 4/5。当前不能直接用这批
分布训练更大的跨物体 Cm。上游 hard-object 规则因名字含 `small`
把 cubesmall 重复为 21/64 环境，其余对象仅 10–11；下一步先做一次
有界的对象均衡 continuation，区分采样失衡与更深的策略表示问题。
该 e320→e360 均衡 Probe 已完成：总体 held-lift 22/64，但覆盖仍为
3/5；toothpaste 提升至 11/13，cubesmall 与 waterbottle 均为 0/13，
且 cubesmall 的 H10 正窗口降为零。简单均衡只重新分配了技能，未
扩大对象覆盖，标 `UNPROMISING`。下一步以 waterbottle-only 短程
continuation 区分共享策略干扰与该对象的 reward/curriculum 难题。
waterbottle-only e320→e360 也已完成，但 e360 为 0/64 held-lift、
接触比例仅 .0169、H10 正承重窗口为零，固定门失败。移除多对象梯度
仍不足以学习该物体；在改 curriculum 前，先用现成 e320/e340
checkpoint 判断是初始状态覆盖不足还是 continuation 训练塌缩。
该诊断显示 e320 waterbottle 已有 contact .131、138 个正 H10 窗口
覆盖 14 个环境；e340 正窗口骤降为零、contact .043，e360 继续降至
.017。因此是 PPO continuation collapse，不是源 actor 完全没有可学
状态。下一步冻结 e320 做按物体等量的表示数据池；后续 Cm policy
Probe 应检验表示能否阻止这种塌缩，而非继续裸 PPO。

---

## 1. North-star status

### A. Self-trained manipulation

当前已经存在不依赖官方 actor checkpoint 的较强单轨迹结果。

V1.28：

`s1_airplane_lift`

5 个未见 seed：

* 61/64
* 62/64
* 63/64
* 59/64
* 62/64

总计：

307 / 320 = 95.94%

因此：

> “自训练策略是否能够在固定任务上稳定抓取”已经不是当前最重要的未知问题。

该结果不等于多轨迹或泛化已经解决。

---

### B. Cm as PPO training reward

V1.46 使用冻结 CmLite 预测的一步目标进展作为低权重 PPO 训练奖励。

Matched evaluation：

Cm-on:

262 / 640 = 40.9%

Cm-off:

399 / 640 = 62.3%

difference:

-21.4 pp

五个 seed 均不利于 Cm-on。

当前判断：

> “直接把一步 Cm 预测进展作为 PPO reward”在当前配方下为明显负向路线。

除非出现新的理论理由或模型发生本质变化，不继续只调 reward coefficient。

---

### C. Original Cmv2 on current PPO distribution

V1.49：

原始 Cmv2 V1.3 使用 nominal PD target hand flow。

接触样本平移 EPE：

seed95:

39.05 mm

seed96:

52.74 mm

对应 zero-object-motion baseline：

5.36 mm

10.35 mm

CmLite：

3.48 mm

6.75 mm

虽然有效 contact token 激活率 > 93%，原始 Cmv2 在当前 PPO 分布上的一步预测仍明显失准。

---

### D. Nominal-vs-actual hand-flow mismatch

V1.50：

用真实下一步 `next_q` 构造仅用于离线诊断的 oracle hand flow。

接触样本 Cmv2 EPE：

seed95:

39.05 mm -> 9.02 mm

seed96:

52.74 mm -> 14.60 mm

下降：

76.9%

72.3%

说明：

> nominal PD target hand flow 与真实一步执行 hand flow 的失配是原始 Cmv2 失准的重要来源。

但是 oracle 结果仍然差于 zero-object-motion baseline。

因此：

> hand-flow mismatch 不是全部原因。

oracle 输入不可用于在线策略。

2026-09-23 对另一架构 ObjectInteractionCm V1.3 的混合 GRAB MANO +
几何重定向 Inspire 预训练权重做冻结迁移小 Probe：自训练 PPO 的
seed95/96 各 8 条接触真实执行转移上，nominal-flow 物体点流 EPE
分别 38.41/33.63mm，零运动 6.50/11.70mm；事后 next-q oracle-flow
降到 6.85/2.35mm。两个架构均提示 hand-flow 执行失配，但小样本
不能证明全部机制。该预训练权重不能直接用于 PPO；下一步做同数据
预训练初始化 vs scratch 的仿真转移微调 Probe。无交互时该模型也可能
输出非零流，需门控。混合预训练运动学标签可能含穿模，现有最近距离
字段非 signed penetration，不能据此声称已过滤穿模。

随后固定的 `P-20260923-mixed-cm-sim-adapt` 用 64 条仿真真实转移、
40 次相同 mini-batch 更新比较该混合预训练与同架构 scratch；
未见 seed95/96 的 16 条运动接触转移上，40 步 EPE 10.65mm vs
11.22mm，零运动 10.44mm。预训练相对 scratch 仅低 5.1%，
没有通过预设 ≥20% 且优于零运动的门槛；标 `UNPROMISING`
（当前极小样本配方），不继续全量微调或接 PPO。下一步优先
校准在线可用的动作→真实手执行模型，或改小型局部几何 Cm 表示，
再以新的仿真 seed 测动作效应。此结果不否定混合几何数据可用于
跨手表示预训练。

2026-09-24 `P-20260924-executed-handflow` 已验证在线可得的
`q_t,a_t,q_{t-1}` 足以在当前 DExplore 单轨迹分布预测下一步
手表面运动：逐关节 gain+momentum 在未见 seed95/96 的接触样本
EPE 2.70/2.88mm，静止手 10.46/8.26mm，PD 目标瞬时到位
45.93/43.43mm。训练仅用自训练策略的 seed74/78 转移；无
未来状态输入。此为 `PROMISING` 手执行模型 Probe，不证明物体效应
或策略效用。下一步先测试校准 hand flow 是否修复冻结 Cm 的物体
效应，之后再决定小型局部几何 Cm 的训练。

同日冻结混合 Cm 接入校准手流后，接触物体点流 EPE 从 nominal
的 seed95/96 41.02/35.25mm 降为 5.97/6.21mm，接近 oracle
5.29/4.92mm；零物体运动 5.87/9.14mm。输入失配是大因子，
但冻结物体效应在 seed95 仍不能稳定优于零运动。

随后 12,932 参数六区域局部几何 Cm 在真实仿真转移训练 300 步，
未见运动接触测试 EPE 5.75mm vs 零运动 11.61mm；然而同架构
零手流 6.04mm、动作置乱 5.93mm，动作特异增益极小，静止接触
还会假预测运动。`P-20260924-local-geometric-cm` 标 `UNPROMISING`，
不接 PPO。旧 first-grip 两臂文件的严格预干预状态匹配 seed97/98
只有 1/64、0/64（replay seed98 0/64），不能充当真实同状态
动作反事实。下一步先构建并验证配对仿真转移，再评估 Cm 的
动作效应；不能再凭观测转移离线 EPE 宣称 policy utility。
配对采样器的 CPU-only smoke 被 DExplore 基础层硬编码 CUDA 阻断；
后续 GPU 工程门结果见下段。

2026-09-24 后续审计：等待任务实际在空闲 GPU6 运行，但顺序恢复
刚体误差 8.03，逐样本物理 pair 被严格拒收；并行三臂从相同
frame0 开始，到第二步关节/刚体已经分化，也无法当作逐样本反事实。
该采集路线标 `UNPROMISING`，不是 Cm 假设被否定。

转向真实仿真执行的随机动作干预：接触状态中腕部 z 命令随机 ±0.3，
seed145/146 的物体一步竖直位移加减臂差分别 +29.24/+26.00mm；
较小 ±0.1 的未见 seed147 为 +9.16mm，均通过分步置换检验
p≈0.0002。实际手竖直运动差分别 +76.34/+75.91/+27.35mm。
这是 `P-20260924-randomized-action-effect` 的 `PROMISING` 平均
干预效应，不是逐样本反事实、更不是 Cm 策略效用。下一步须先
用随机干预数据重校准在线动作→实际手运动，再训练/检验 Cm 的
动作条件物体效应，只有过门才接 PPO。

后续混合 ±0.3/±0.1 干预训练了在线 `q,dof_vel,action→next_q`
逐关节校准：未见 seed146/148 的手表面 EPE 12.21/7.49mm，
旧窄动作 gain 为 20.37/8.49mm；腕部动作对比预测接近真实，
但小幅度 EPE 未过预设 30% 改善门。暂作 Cm 输入而非最终执行模型。
随后 12,932 参数六区域几何 Cm 在相同两 seed 的真实随机干预
转移上，物体平移 EPE 11.25/5.81mm；同架构零手流
16.79/7.66mm、raw state+action MLP 13.86/6.53mm。
小动作平均效应仍低估，故不直接宣称模型已校准。

更决策相关的 `P-20260924-cm-cate-ranking` 在未见 seed146/148
用 Cm 预测分数预先分组，真实随机处理效应最高−最低四分位
分别相差 +72.38mm (95% CI +61.27~+81.82) 和 +25.51mm
(CI +22.18~+28.18)，标 `PROMISING`；raw MLP 也接近，
不能说几何是唯一原因。

在线 `P-20260924-cm-online-action-boost` 固定 Cm 预测的单步物体
抬升差，选择接触窗口 +0.1 抬腕。新 seed149/150 各 64 环境：
base held-lift 87/128，always-boost 73/128，Cm-boost 86/128。
Cm 比盲目 boost 少伤害，但没有改善 base；两个 seed 的平均 reward
和接触比例均下降。这条直接上抬接法标 `UNPROMISING`，停止
调阈值；不能把离线 CATE 排序误当策略收益，也不能据此否定
Cm 的其他策略接法。随后新 seed151/152 的随机 ±0.1 五步随访
显示明确权衡：一步物体上移差 +12.19/+13.44mm，但五步手物
接触比例差 −7.58/−4.70pp，两个 environment-cluster 95% CI
均不含零；这为“只按一步抬升选择动作可能损害抓取”提供
了机制线索，但不是正式中介效应证明。
随后 contact-aware Cm 用 seed151/152 训练、未见 seed153/154
测试：六区域几何模型的一步物体 EPE 6.31/5.91mm，接触
RMSE .121/.143，优于同架构零动作手流；预测的接触效应
最高−最低四分位对应真实 RCT 差 +22.40pp（CI +15.87 至
+29.29pp）。但更小 raw state+action MLP 排序 +25.11pp，
接触 RMSE .115/.131，几何独立优势未证实。按预注册门槛
不升级在线候选控制，也不基于已看测试 seed 调网络。
新 seed155/156 的固定 .5/.5 geometry+raw 集成只比 raw 的
接触效应排序高 0.87pp（配对 CI −2.26 至 +3.98pp），且
一组 seed 接触 RMSE 略差；六区域几何互补性门未过，停止
这套结构的小幅微调。随后用动作条件 raw Cm 在真实闭环中
选择 `wrist-z −0.1`：seed157 ×64 原策略 held-lift 44/64、
始终下压 40/64、Cm 选择下压 40/64；Cm 实际选择 587/3199
个 eligible 状态。已触发预注册的每 seed 不负门失败，故省略
不会改变升级决策的 seed158，停止这一局部腕部动作接法。
下一步需返回更高层，评估训练期 Cm representation/auxiliary
与重新定义动作/长时域目标的成本和判别力；正式 Cm policy
utility 仍未证明。

---

### E. Cm PPO actor 权重：已有窄范围正证据，未达稳定目标

V1.51 在 s3 单轨迹、固定自训练 e260 源与训练 seed70 下，冻结 CmLite 的
动作条件接触概率 × 预测一步竖直位移，为 PPO actor 样本赋予有界权重。
真实环境 advantage 决定梯度方向，Cm 不改变奖励或最终推理。未见 seed119–123：
on 396/640、off 317/640，+12.34pp，通过该配方的预设增益门。

V1.52 在新 seed124–128 上，on 402/640、保持相同权重边际分布但置乱样本
对应关系的 placebo 345/640，+8.91pp，通过样本对应关系门；on 对 off
358/640 仅 +6.88pp，**未通过**预设的严格复制门。当前仍不能声称稳定抓取。

2026-09-23 三个短 Probe：已有自训练真实转移上，预测位移分量对真实一步
接触位移事件的排序强于接触概率；固定新 seed129/130 的在线消融中，
effect-rank 90/128、joint 83/128、contact-rank 45/128、off 72/128。
新 seed131/132 的动作置乱对照中，真实动作 effect-rank 76/128、动作置乱
42/128、off 69/128。动作对应关系具有方向性正证据，但 effect-rank 对 off
仍有 seed132 负差，且只有一个训练 seed。这些均是 Probe，非正式验证。

### F. 跨训练 seed 正式验证否定了当前 effect-rank 联合主张

`VAL-20260923-CM-EFFECT-PPO` 使用新训练 seed71–74、未见评估 seed133–138，
三臂各 1536 个严格完整首 episode；12 次训练、72 次评估全部有效。
effect-rank 882/1536、off 789/1536、动作置乱 1049/1536。
effect−off +6.05pp，但 95% 双维 bootstrap CI −14.71 至 +26.69pp，
训练 seed71/72 为正、73/74 为负，未过预设 utility 门。
effect−动作置乱 −10.87pp，CI −42.06 至 +20.18pp，未过动作对应门。
按预注册联合规则标 `REFUTED`：**当前 effect-rank 配方不能作为 Cm
稳定策略增益的证据**；这不否定 Cm 整体或其他接法。effect-rank 24 次
评估仅 1 次达到 ≥58/64，抓取稳定性也未达标。

`P-20260923-CM-SIGNED-UP-RANK` 的廉价诊断显示：冻结 CmLite 在失败
seed74 策略的接触转移上仍有 signed dz 相关性（r=0.664）；旧 `|pred dz|`
最高十分位的真实平均 dz 为 −5.06mm。但只把 PPO 排序改为预测正向 dz
并没有改善：同训练 seed74 的新策略在评估 seed133/134 为 24/128，
旧 effect-rank 37/128、off 68/128。该有向排序接法 `UNPROMISING`；
离线预测关联不能当成策略收益。

原 joint 权重在困难训练 seed74 的小 Probe 为 84/128，off 68/128，
通过该次继续门；在第二困难 seed73 为 79/128，off 71/128，
仅 +6.25pp，未过预设 +8pp 继续门。两 seed 的动作置乱参考分别为
96/128、98/128，均高于 joint。因 seed/评估选择已知且第二门未过，
不把 joint 推进正式验证，不能据此证明动作条件 Cm 的独立贡献。

`P-20260923-CM-CRITIC-SALIENCE` 将冻结 Cm 的一步效应只用于
minibatch 归一化的 PPO critic loss，actor loss 不变。已知困难训练
seed74 为 98/128 vs off 68/128，出现大正信号；第二困难 seed73
为 80/128 vs off 71/128，seed133 还低于 off，未过预定的
≥8pp/每评估 seed 正门槛。没有置乱显著性控制，故不升级正式验证，
不能称为 Cm policy utility 证明。

### G. 训练期 Cm 辅助表征：方向为正但未过升级门

`P-20260924-cm-ppo-aux-representation` 从同一自训练 e260 actor
出发，训练 seed75/76 的 Cm-on/off 四臂各微调到 e300。两臂
同网络、同 PPO 配方、同冻结 raw-action contact-aware Cm；仅
actor 共享特征的三维物体/接触效应辅助损失系数为 .002/0。
未见评估 seed159/160 的完整首 episode held-lift：训练 seed75
on 61/128、off 50/128（+8.59pp）；训练 seed76 on 75/128、
off 71/128（+3.13pp）。合计 on 136/256、off 121/256，
+5.86pp，低于预注册 +8pp 升级门。辅助目标确实被学习，但
Cm 特异且稳定的 policy utility 未经置乱目标对照或正式
Validation 证明。本 Probe 标 `UNCLEAR`；不在已见评估 seed
上调系数，也不把这条固定配方推进正式验证。

### H. 多轴、十步真实干预提供新的可学习方向

用户选择 B 后，`P-20260924-multiaxis-h10-effects` 在自训练
e260 actor 的实际仿真接触状态随机施加腕部 x/y/z 各 ±0.1
一步动作，并随访 10 步。新 seed161/162 的 x 轴对物体 x
位移效应分别 +21.15/+13.04mm；合并 +17.23mm，环境聚类
95% CI [12.30,22.06]，通过预注册非 z 轴继续门。y 轴
也同向，合并 +13.05mm [1.66,22.05]。这是人群平均物理
干预效应，不是逐状态反事实、Cm 预测成功或 policy utility。
下一步用两 seed 训练新的多轴/十步动作条件 Cm，在未见
seed 上与 state-only 对照比较动作效应预测；过门才接策略。
后续 `P-20260924-multiaxis-h10-cm` 用 seed161/162 的 1574
接触样本训练 500 更新，在未见 seed163 的 803 样本上：
真实 x 十步效应 +16.76mm；state-only/raw/六区域几何的
x 轴 factual RMSE 28.16/26.69/25.25mm，预测 x 对比
0/17.49/21.41mm。两动作条件模型过预注册门；几何在 y/z
误差较大，尚不能说它整体优于 raw。下一步用随机数据检查
跨轴效应的任务相关性，再用全新 seed 做最小 matched 在线
决策 Probe。该 `P-20260924-multiaxis-h10-online-choice` 在
新 seed164 的 64 环境完整首 episode 中：base 36/64、
固定 x+ 32/64、固定 z+ 39/64、Cm 双轴选择 34/64。
Cm 确实执行 x+ 156、z+ 217 次，仍低于 base；违反预注册
每 seed 不负门，省略 seed165 并停止该固定在线规则。多轴
真实效应和未见动作预测可学，仍不能推出策略增益；短期
单步候选的选择目标或接法与完整抓取尚未对齐。

### I. 两步动作有真实效应，但任务相关增益仍未成立

`P-20260924-cm-two-step-physical-effect` 在新 seed168/169/170
将 wrist-x ±0.1 随机施加一步或连续两步，十步物体 x 的
第二步增量分别 +18.67/+19.84/+18.96mm，环境聚类 95% CI
均为正；平均接触比例两步比一步约低 0.5–1.3pp。新架构
`StructuredTwoStepCm` 将第一/第二步效应分开，在未见 seed170
的十步 x RMSE 比同容量 state-only 低 11.84%，预测交互效应
为真实的 0.843 倍，模型 Probe `PROMISING`。原始拼接架构
虽然 x RMSE 低 11.43%，却仅预测 0.316 倍交互效应，
未过门。结构化模型同时使用了更多训练数据，不能把二者
差异单独归因于架构；两种模型在各自留出 seed 上的判别门
结果可用，但架构独立贡献仍需 matched 对照。
这也不等于策略有收益。seed170 的探索性 wrist-x 对十步物体 z
对比为负，不适宜仅凭 x 效应做抬升规划。

随后 `P-20260924-cm-two-step-z-effect` 在新 seed171/172
随机比较一步与两步 wrist-z ±0.1；合并第二步 z 增量
+6.33mm，环境聚类 95% CI [−0.93,+12.86]mm，
两步减一步平均接触比例 −2.61pp。点值有上移信号，
但 CI 跨零，未过预注册继续门；不在这两个 seed 上增加
重复剂量 z 模型、seed 或规划。尚无 Cm policy utility 证明。

新的混合轴 `P-20260924-cm-crossaxis-primer` 把 x− 预调整
接在 z+ 上抬之前，与仅 z+ 的随机对照比较。新 seed173
的十步接触加权物体 z 位移差为 −2.52mm，环境聚类 95% CI
[−7.67,+3.06]，接触比例差 −2.63pp；预注册点值为负，
不加 seed、不从同一 seed 改选 x+、不建新 Cm。分类
`UNPROMISING` for 这一固定序列。简单 x/z 动作组合已不足以
提供下一次大规模 Cm-on/off 的明确物理先验。

---

## 2. Current main unresolved question

当前最重要的问题不是：

> baseline 是否还能再高几个百分点？

而是：

> 如何让 Cm 的一步信息在不同训练 seed 下可靠地改善策略，而不是只在
> 单个训练 seed 或某种权重排序中偶然奏效？

---

## 3. Current active hypothesis

真实随机动作干预已证明当前单轨迹接触分布存在可学习的
多轴、十步物体效应，未见 seed 的 Cm 也能预测其中 x 效应。
但固定在线候选选择低于 base；H5 与 H10 两种训练期辅助
目标均未过跨训练 seed 的策略收益升级门。两步 x 序列
效应能被结构化 Cm 预测，但与抬升方向没有直接对齐；
重复 z 剂量的额外物理效应未过正值置信门。冻结 CmLite 的
PPO reward、actor effect-rank、critic-salience 也没有提供稳定
效用。简单两步 x/z 序列也未形成任务相关的物理增益。
当前怀疑动作族与完整抓取目标不对齐，不能继续在已见
seed 上微调剂量/系数。若继续 Cm，需改变受控动作族或
监督目标；离线预测或辅助 loss 下降不能替代 matched
policy utility 证据。
跨物体新增证据进一步表明：简单扩大同类数据、预测对象平均
wrist-z 效应，也没有提供状态内选择价值。更可能的 blocker 是
self-trained policy 所到达的状态分布与抓握技能本身，而非缺少
另一个局部动作效应回归头。

---

## 4. Current decision boundary

下一阶段应该优先回答：

### Question 1

当前随机干预已给出平均和条件平均处理效应，不是逐状态物理
反事实。多轴 H10 和两步 x Cm 能预测动作效应，但在线
单步选择和训练期辅助学习均未给出稳定抓取收益。下一步
的核心问题是：是否值得改用接触维持的手指动作序列和
接触支持抬升监督目标，还是需要先调整任务/论文 claim？

### Question 2

新方法通过小 Probe 后，再做 matched Cm-on/off 跨训练 seed Validation。
只有正式验证通过才可支持 Cm policy utility claim，之后再提高稳定性并扩展多轨迹。

---

## 5. Things NOT worth prioritizing now

除非它们成为 blocker，否则暂时不优先：

* 继续优化无 Cm baseline 的几个百分点；
* 大规模多轨迹泛化；
* 很多 seed 的早期探索；
* 原始 Cmv2 每一个误差来源的完全解释；
* 对所有历史版本进行重新验证；
* 长期 supervised decoder rollout 完美稳定性；
* 为论文提前补完整 ablation。

这些内容进入 Research Debt。

---

## 6. 已结束的路线与判定经过

用户选择的 H10 多轴 Cm 训练期辅助路线已完成最小 matched
Probe：训练 seed77 on/off 75/128 vs 60/128，seed78 68/128
vs 73/128；合计 143/256 vs 133/256（+3.91pp），低于
预注册 +8pp 且一训练 seed 为负，标 `UNPROMISING`。
停止该 target/系数，不做置乱目标正式验证。随后小规模
两步序列 Probe 已证明 x 效应可预测，但 x 与抬升不对齐，
重复 z 的额外上移在两个 seed 合并后仍不确定。下一步
不应直接接 x-only 规划，也不应在已见 z seed 上调剂量；
随后 x− 预调整→z+ 的随机物理 Probe 也为负。下一步若继续
需改变动作族和监督目标，例如接触维持的手指动作与抬升
结果；这已是较大的路线选择，见
`docs/decisions/D-20260924-after-sequence-gates.md`。
建议先与用户复盘任务和论文 claim，不再试相邻 x/z 剂量。
正式 policy utility 仍需 matched Validation，当前未获证明。
train8 H10 的最后一个 z-only 机会门现也已失败。若继续原 claim，
下一路线应改变策略表示/状态覆盖本身，并先用最小 Probe 证明 Cm
表示在未见对象上提供超出原 observation 的承重状态信息；否则
停止增加局部动作头，复盘 claim。
当前最小前置 Probe 已进一步定位到 train5 正承重状态覆盖不足；
对象均衡短程 continuation 未过门，不能进入 dense Cm representation
Probe，也不继续调采样权重。先做单对象可学习性诊断；若单对象可学，
再设计能够缓解多对象干扰的策略表示，若不可学则先修 curriculum。

随后固定 e320 策略、按对象分层重放解决了 representation 数据支持
blocker：airplane/cubesmall/mug/toothpaste/waterbottle 均达到至少 32 个
H10 承重正例、32 个失败例且正例跨至少 4 个环境（5/5）。waterbottle
在 e320 有 138 个正例但 e340/e360 降为 0，说明 plain PPO continuation
会破坏已有瞬态支持，而非初始化完全没有支持。现在进入冻结原始
ObjectInteractionCm V1.3 的 object-LOO 表征 Probe；必须同时超过 raw、
action-blind 和 action-shuffled control，才值得做 matched PPO Cm-on/off。

该固定门现已失败：640 个对象均衡 H10 样本上，V1.3 高密度
action-aware token 的 object-LOO AUC 为 0.6387，仅 3/5 fold >=0.60；
action-blind 为 0.6694，action-shuffled 为 0.6382，raw 为 0.5640。
10135 点相对 1538 点仅 +0.0025 AUC，且各对象 sample-valid 为
0.922–0.984。结论是 token 有跨对象状态/几何信号，但没有动作相关的
增量证据；点流密度不是该失败的主要原因。按预注册决策停止当前
ObjectInteractionCm V1.3 表征路线，不启动 matched PPO。对象留一只作用于
新训练的线性头：冻结 V1.3 的预训练 train split
已经包含这五个物体，不能把该 Probe 称为整个 Cm 的未见物体泛化测试。
工程上同时
发现通用 extract 入口传 `cfg.model` 会丢失 task-level meta、静默回退
K=8/5cm；已改为传完整 cfg，正式 Probe 明确断言 K=32/2cm。

用户确认回到单对象任务对齐路线后，先去重发现相同 sustained
grip+lift option 已在多对象 e260 actor 上失败（相对 lift-only 仅
+0.15mm；airplane 子组 +1.45mm）。随后补做唯一未覆盖的单对象单网络
e140 actor 物理门：572 个有效接触状态中，grip+lift 相对 lift-only 的
H20 接触加权抬升效应为 -8.20mm，环境聚类 95% CI
[-14.08,-2.49]mm，接触比例也为 -2.05pp。预注册门明确失败且方向
为负；不跑 seed192、不为该动作族训练 Cm。局部 wrist/finger 动作族、
短序列、冻结 V1.3 表征以及多种 PPO 接法现均未提供稳定 policy utility。
用户已在上述 Decision Checkpoint 选择停止当前 campaign，不能再调
相邻系数或将尚未建立的 Cm policy utility 写成正式结论。

---
