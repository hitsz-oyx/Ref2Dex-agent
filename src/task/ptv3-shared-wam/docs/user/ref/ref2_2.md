\## 为什么 PointWAM 的 retargeter 能学会\
\
把今天的统计和 \`paper/pointwam/\` 里的一手审计放在一起看，PointWAM 和我们的差别不在模型、不在数据量（959 条 vs 660 条），也不主要在标签坐标系（ref2_1 的相对 PD/几何表示只换来 7–9%），而在\*\*标签是从哪来的\*\*。\
\
\*\*1. 它的动作标签是"被机器人跟踪的位置目标"，我们的是"RL 策略的力指令"。\*\*\
DexJoCo 示范是遥操作的 EEF 位姿 + 手指目标，机器人用刚性控制去跟踪它；作者还只保留了 1000 条里 \*\*959 条 open-loop replay 成功\*\*的示范（[grasping-gap 审计]\(paper/pointwam/grasping-gap-primary-audit-20261013.md)）。这个过滤等价于：动作序列本身就是一条能独立复现结果的运动学计划，正向映射 动作→运动 近似恒等。于是 \`手轨迹 → 动作 chunk\` 是一个确定性、低噪声的几何映射。\
我们的 RL660 标签是 DExplore actor 在 Kp=200 下输出的 PD 偏移：目标平均离手 128 mm，手每 tick 只走 25 mm，8 步后离目标更远，逐 tick 新息 43 mm。它是一个依赖反馈的力/加速度指令，open-loop 回放 8 步必然发散。同一个 chunk 训练目标，在他们那里是 deployment-aligned，在我们这里是在蒸馏一个闭环控制器的内部变量。\
\
\*\*2. 教师的隐变量不同。\*\* 人的意图全部体现在手轨迹里，除此之外没有决定动作的隐藏状态。DExplore actor 的动作则由 contact forces、t+1/t+16 参考 obs、interaction graph 决定（\`base_dexplore_task.py:383–404\`），R 看不到这些。手不动时 \\|PD−q\\| 250–290 mm 就是在"发力"，运动学上无解。\
\
\*\*3. 评价目标不同。\*\* PointWAM 看闭环成功率；我们在 PD-mm 上比。对力式指令来说 PD-mm 的下界就是标签噪声，而且 43 mm 目标误差 ≈ 8.6 N，并不是 43 mm 的手位移误差。\
\
\*\*4. 旧 PDRetargeter 12.5 mm 的结果不构成反例。\*\* 那张卡（\`P-20261011-competent-pd-retargeter\`）是同一条 airplane motion 的 32 个 env 按 env 划分：同一参考轨迹的近重复 rollout，策略的"高频"其实是参考相位的确定函数，从 τ 可辨识。换成 660 条不同 motion、每条一次 rollout，就不再成立。\
\
\*\*5. 1.15M 视频救的是 forecaster，不是 retargeter。\*\* 这点 ref2_1 说得对。但它正好对应我们另一个真实缺口：dev \`hand_first8\` 从 step 500 起就平在 51 mm，101M 模型在 589 条序列上记忆。PointWAM 的 forecaster 能泛化，我们的不能。\
\
一句话：PointWAM 解的是 \*\*retargeting\*\*，我们现在解的是 \*\*policy inversion\*\*。ref2_1 把坐标系对齐了，但标签来源没变，所以上限没变。\
\
\## 和 RL 结合，我建议怎么做\
\
MISSION 要求的两件事是：不以官方 actor 为最终策略基础自训练出能抓的策略（A），并证明 Cm 有增益（B）。基于上面的诊断，核心原则是：\*\*把"运动学上可确定的部分"交给监督学习，把"运动学上不可观测的部分"交给 RL。\*\* 现在的做法是让 R 用监督去学不可观测的力项，这正是失败的地方。\
\
\*\*第一步，零成本重标注，把 R 变回 retargeting 问题。\*\*\
RL660 里已有 \`q[t+1..t+8]\`。不要以 \`applied\` 为标签，而以\*\*实际实现的未来关节/腕位姿\*\* \`q[t+k] − q[t]\` 为标签（这就是 PointWAM 的数据天然给出的东西），R 学 \`τ, s_t → 未来关节轨迹\`。预期误差应到几 mm 量级，因为这是纯运动学映射。同时模仿作者的 replay 过滤：用刚性位置控制回放实现的 q 轨迹，只保留能复现物体运动的窗口，把"不可 open-loop 复现"的段落剔出监督集。这一步同时解决 chunk 执行（K4/K8）的合法性问题。\
\
\*\*第二步，力项交给 residual RL。\*\*\
策略结构：\`WAM forecaster → R_kin（关节轨迹）→ PD 目标\` 作为 nominal，PPO 学一个零初始化的残差加在 PD 目标上。残差要补的恰好是今天统计出来的东西：−36 mm 的持续下压、接触预紧、起步瞬态。这些有奖励可以学，不需要教师。它满足 Objective A（不依赖官方 actor 权重），而且起点就在示范附近，探索难度远低于从零 PPO。README 里已经把 Residual PPO 列为后续，我认为它应该提前，而不是继续对 raw-native R 调 loss 或表示。\
如果想保留教师，可以让官方 actor 只做 DAgger 式的访问态标注，但标签同样应是实现的 q，而不是它的指令。\
\
\*\*第三步，Cm 的位置。\*\*\
Cm（短期手物交互预测，即 scene head）在这个结构里有三个自然入口，便于做 matched on/off：作为残差策略/critic 的观测特征（预测的物体位移）；作为 dense reward 或 value 的一部分；作为候选手轨迹的排序器（\`agent/cm-action-ranking\` 路线）。只有残差 RL 把执行底座打通后，Cm 的增益才可能被分离出来；现在卡在 R 上，Objective B 根本无法开测。\
\
\*\*在此之前先做一个便宜的决策实验。\*\* 用现有 \`full_stream-step1500.pt\` 在 2–3 条 dev 序列跑 K1 GT-τ 闭环（几分钟 GPU）。如果 43 mm 的力指令误差在闭环里其实能握住，那 R 问题的紧迫性要重新排序；如果握不住，上面的路线就是主线。另外当前 geometry-lift 19k run（ETA 约 5.5 h）不会改变这个判断，是否跑满你可以决定；forecaster 侧 dev 早已平台，继续训练主要是在记忆训练序列。\
\
两点需要注意的不确定性：相对 q 的重标注把"力"从标签里剥离后，nominal 单独执行大概率抓不住（它少了预紧），这是预期行为而不是失败，残差 RL 才是补这块的；以及 DExplore 的 Kp=200 对位置控制偏软，residual 阶段可能需要评估更高的腕部增益或把残差定义在力/增益上，这会改变与官方环境的可比性，需要在卡里单独记录。这个说法怎么样？我们现在怎么做？


这个说法的**核心判断我基本赞同**，而且我会把它升级成当前最值得验证的主假设：

\[
\boxed{\text{PointWAM 做的是低熵的几何 retargeting，我们现在在做高熵的 policy inversion。}}
\]

但有两处需要改得更严谨，然后我会直接调整下一步。

第一，你写“我们的标签是 RL 策略的力指令”不够准确。DExplore 输出的不是直接 force action，而是**反馈依赖的 PD setpoint / normalized command**；它通过 \(K_p\) 产生力。因此更准确的表述是：

> PointWAM 的监督目标更接近可开环复现的几何位置目标；RL660 的监督目标是闭环策略在当前反馈下产生的 PD setpoint，它隐含接触、参考相位和当前误差补偿。

这不影响你的核心结论，反而更严谨。仓库里也确实能看到 DExplore actor observation 含参考未来信息和 contact 相关量，而当前 R 看不到完整的这些输入。

第二，**`q[t+k]-q[t]` 不能直接说“这就是 PointWAM 的 action label”**。它是一个很好的低成本替代，但它是“实际实现的状态轨迹”，不是原始 desired target。PointWAM 的动作本质上是期望机器人去到哪里，而 \(q_{t+k}\) 是系统最终走到了哪里。

所以我们应该先证明：

\[
\boxed{\text{把真实 }q_{t+1:t+8}\text{ 当作位置目标重放，真的能复现原运动。}}
\]

这一步一旦成立，你后面的路线就非常合理。

---

## 我现在建议不要先跑 residual PPO，先做一个更关键、更便宜的 Gate

直接拿现有 RL660，不训练任何网络。

对每个窗口构造：

\[
Q^{GT}_{t}
=
(q_{t+1},q_{t+2},...,q_{t+8})
\]

然后从真实 \(s_t\) reset，用**当前相同的 Inspire controller / Kp=200**，直接把这些真实未来 q 当 target，K4 或 K8 执行。

比较：

\[
q^{replay}_{t+k}
\quad\text{vs}\quad
q^{recorded}_{t+k}
\]

以及最重要的：

\[
O^{replay}_{t+k}
\quad\text{vs}\quad
O^{recorded}_{t+k}.
\]

也就是同时看：

- joint tracking；
- 11 hand point tracking；
- object trajectory；
- held/contact outcome。

这个实验我会叫：

\[
\boxed{\text{GT realized-q replay gate}}
\]

它比现在再测 `full_stream-step1500` 更重要。

---

# 为什么这一步决定整个路线

有三种结果。

### A. GT-q 能重放手，也能重放物体

例如：

\[
EPE_{hand}<5\sim10\text{ mm}
\]

而 object trajectory 也能基本复现。

那就非常强地说明：

\[
\boxed{q_{future}\text{ 是一个有效的 nominal action representation}}
\]

这时我们立刻训练：

\[
R_{\text{kin}}(s_t,\tau)
\rightarrow
q_{t+1:t+8}-q_t.
\]

这才是真正应该做的 R。

而且预测误差应该同时报告：

\[
q\text{ error}
\]

和

\[
FK(R_{\text{kin}})-\tau
\]

的手点误差。

如果后者能做到几毫米，我们基本就把 R 问题解决成了真正的 retargeting。

---

### B. GT-q 能重放手，但物体抓不住

这个结果其实也**很符合你的假设**。

说明：

\[
\boxed{\text{运动学计划是对的，但缺接触预紧/力项}}
\]

也就是说 nominal：

\[
R_{\text{kin}}
\]

负责：

> 手应该走到哪里。

而 residual RL 负责：

> 为了真的抓住，需要多压多少、哪些手指预紧、接触时怎么补偿。

这正是最干净的：

\[
\text{supervised kinematics}
+
\text{RL dynamics correction}.
\]

这种情况下 nominal 单独抓不住**不应该判失败**。

---

### C. 连 GT-q 连手运动都重放不了

那你当前这条方案就要停一下。

因为如果：

\[
q^{GT}_{future}
\]

作为 target 都无法在同控制器下重现：

\[
q^{recorded}_{future},
\]

说明当前 DExplore dynamics / controller 本身高度依赖原策略每 tick 的 feedback compensation。

那单纯：

\[
\tau\rightarrow q\text{ chunk}
\]

也未必足够。

这时才需要考虑：

- K1 而不是 K4/K8；
- velocity/feedforward；
- 更高 Kp；
- trajectory tracking controller；
- 或 policy residual 更靠近低层。

---

# 我不建议第一步就提高 Kp

你最后提到可能提高 wrist gain。

这个可以做，但要作为**独立 ablation**。

先保持：

\[
K_p=200
\]

因为这是现有环境和数据生成时的 controller。

否则如果直接改成：

\[
K_p=400
\]

然后 GT-q replay 成功，我们分不清到底是：

> q-label 合理

还是：

> controller 被换强了。

所以顺序应该是：

\[
K_p=200\quad\text{先测}
\]

如果 hand tracking 明显 lag，再做：

\[
K_p\in\{200,300,400\}
\]

的小规模 controller ablation。

---

# 然后再训练真正的 R_kin

如果 Gate A/B 通过，我建议直接把现有 `action_target` 换成：

\[
\boxed{
y_k=q_{t+k}-q_t,\quad k=1...8
}
\]

但这里我会稍微区别 wrist 和 finger。

更稳的版本：

\[
y_k=
[
\Delta p^{wrist}_k,
\Delta \theta^{wrist}_k,
\Delta q^{finger}_k
].
\]

因为 wrist Euler branch 我们刚已经踩过 \(2\pi\) 的坑，所以可以直接复用现在 geometry-lift 的连续 branch 机制，而不是重新引入问题。

训练：

\[
R_{\rm kin}(s_t,\tau^{GT})
\rightarrow y_{1:8}.
\]

这一次**不需要 `applied`，不需要 PD target**。

---

## 然后做三个 sanity checks

### 1. GT-\(\tau\) offline

\[
\tau^{GT}\rightarrow R_{\rm kin}\rightarrow q
\]

看：

\[
q\text{ RMSE}
\]

和：

\[
hand\text{ EPE}.
\]

我不会预先承诺一定几毫米，但如果这是一个干净 retargeting 问题，确实应该明显低于现在 40–50 mm 的 PD target error。

### 2. GT-\(\tau\) native replay

用预测的：

\[
\hat q_{1:8}
\]

替代 GT q。

看看相对于 **GT-q replay 上界**掉多少。

### 3. predicted-\(\tau\)

最后才接：

\[
W(H,G)\rightarrow\hat\tau
\rightarrow R_{\rm kin}.
\]

这样误差来源终于可以拆成：

\[
\boxed{
\text{forecast error}
+
\text{retarget error}
+
\text{dynamics/contact error}
}
\]

而不是现在全部混在一个 R 里面。

---

# Residual RL 我赞成提前

如果 GT-q replay 结果属于 B——**手能跟，物体不能稳定抓**——我会立刻把 residual PPO 提前。

结构：

\[
a^{nominal}_t
=
Controller(R_{\rm kin}(\hat\tau))
\]

然后：

\[
a_t
=
a^{nominal}_t+\alpha\delta a_t.
\]

这里我建议 residual 最初仍定义在**PD target offset**上，而不是 force。

因为：

- 和当前 Inspire controller 兼容；
- 工程改动最小；
- 零 residual 就严格恢复 nominal；
- 可以直接做 matched on/off。

初始：

\[
\delta a=0
\]

并限制小范围，例如按每组物理单位设 authority，而不是所有 12 维一个统一比例。

---

# RL 的 observation 也应该和“缺失信息”对应

如果我们的诊断是：

> τ 解释运动学，但解释不了预紧和接触反馈，

那 residual actor 就应该读那些 R 故意不读的东西：

\[
o^{RL}=
[
q,dq,
hand-object\ gap,
object\ velocity,
contact/force\ proxy,
WAM/Cm
].
\]

这才有逻辑。

而不是又把：

\[
\tau
\]

重复塞给一个大 PPO 让它重新学所有东西。

---

# Cm 的位置你说得也对

我最赞成的是：

### baseline

\[
\pi_{\rm res}(o)
\]

### Cm-on

\[
\pi_{\rm res}(o,Cm)
\]

其他全部一致。

这样 Objective B 才能真正测：

\[
\boxed{\text{Cm 有没有增益}}
\]

而不是现在 R 根本没过，Cm 的作用完全被执行噪声吞掉。

第一版我会优先让 Cm 进：

\[
\boxed{\text{critic}}
\]

再考虑 actor。

因为 critic-only 更容易得到干净的 on/off：

\[
V(s)
\]

vs

\[
V(s,Cm).
\]

我们以前其实已经有 critic-only Cm 的工程基础。

---

# 关于当前 geometry-lift 19k

我现在的判断是：

**不需要因为这个新假设立刻停，但也不要再把它当主线希望。**

它回答的是：

> 如果继续尝试用监督学习恢复 DExplore 的 closed-loop PD target，长训练是否最终能把它学下来？

这是一个有价值的 negative/positive baseline。

但它不应该阻塞新的：

\[
\boxed{\text{GT-q replay gate}}
\]

因为后者几乎不需要训练。

如果 GPU/环境资源允许，我会**并行**做 GT-q replay，不等 geometry-lift 19k。

---

# 我会修改你原文中的一句话

你现在写：

> “R 学 τ,s →未来关节轨迹，预期误差应到几 mm 量级，因为这是纯运动学映射。”

我建议改成：

> “R 学 \(\tau,s_t\rightarrow\) 未来实现关节轨迹。该映射比 teacher command inversion 更接近纯运动学，但其可辨识性和可执行性必须通过 GT-q replay 与 held-out retargeting 误差验证，不能预设毫米级上限。”

因为 11 个 hand points：

\[
\tau\in\mathbb R^{33\times24}
\]

不一定唯一决定全部关节路径。

例如冗余关节/接触约束仍然可能产生：

\[
\tau_i\approx\tau_j
\]

但：

\[
q_i\ne q_j.
\]

所以这个要实测，不能提前定论。

---

# 我现在建议的实际顺序

1. **GT realized-q replay gate**
   - 3 条 airplane + 3–5 条其他物体；
   - K1/K4，先 K4；
   - same Kp=200；
   - hand/q/object/held 四组指标。

2. 如果 hand replay 通过：
   \[
   \boxed{R_{\rm kin}:\tau^{GT}\rightarrow q_{1:8}}
   \]
   做小规模 R-only。

3. 如果 R_kin offline + native 都通过：
   接 predicted τ。

4. 如果 predicted τ nominal 能接近目标但抓不稳：
   **正式启动 residual PPO**。

5. residual PPO 稳定以后：
   做
   \[
   \boxed{Cm\ on/off}
   \]
   matched experiment。

6. 现有 raw-native / geometry-lift R 保留为 baseline，不再围绕它反复改 loss。

所以你的这套说法，我会给：

\[
\boxed{\text{核心诊断：很有价值；路线：基本赞成；第一步需要改成 GT-q replay gate。}}
\]

如果这个 gate 结果是 B——**手轨迹能复现但物体抓不住**——那几乎就是我们想要的证据：**监督学习负责运动学，RL 负责接触动力学。**这条主线会比现在的 policy inversion 清楚得多。