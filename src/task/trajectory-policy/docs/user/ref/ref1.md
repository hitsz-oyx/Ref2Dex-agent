对。按我们现在讨论收敛后的版本，整套架构我会固定成下面这样。

核心思想只有一句：

> **让策略直接在“未来手轨迹 \(\tau\)”空间里决策；PointWorld 不一定显式输出未来物体轨迹给 actor，而是把“这条轨迹会产生什么 hand-object consequence”编码成 physical token，辅助 RL 优化 trajectory policy。**

---

## 1. 第一层：Trajectory Policy

当前观测/历史：

\[
H_t
\]

输入 trajectory policy：

\[
\boxed{
\pi_\tau(H_t)\rightarrow\tau_t
}
\]

其中：

\[
\tau_t=
\{p^{hand}_{t+1:t+24}\}
\]

是未来 24 步、11 个手部关键点轨迹。

这里我不再限定它一定怎么训练。

现在仓库里的 measured-history predictor：

\[
H\rightarrow\tau
\]

可以先当初始化，因为已经有一定正信号。

后面真正 baseline 可以用 PPO 继续训练：

\[
\boxed{
\pi_\tau:
H\rightarrow\tau
}
\]

直接根据真实 task reward 更新。

所以蒸馏只是：

\[
\pi_\tau^{init}
\]

而不是永远冻结的最终 policy。

---

# 2. 最好不要直接输出裸 792 维

最终我仍建议加一个 trajectory latent：

\[
c_t=\pi_\tau(H_t)
\]

其中：

\[
c_t\in\mathbb R^{16\sim64}.
\]

然后：

\[
\boxed{
\tau_t=D(H_t,c_t)
}
\]

因此严格说高层 policy 是：

\[
H_t\rightarrow c_t\rightarrow\tau_t.
\]

\(c\) 就是：

> **一整段未来手运动的低维控制变量。**

不是 physical token，也不是状态 latent。

以后 PPO / TD3 实际优化的是 \(c\)，避免直接优化 792 维 trajectory。

第一版如果嫌复杂，甚至可以暂时直接输出 \(\tau\)，先验证路线；等动作维度成为问题再做 \(c\)。

---

# 3. 第二层：Trajectory Executor

得到期望未来手轨迹：

\[
\tau_t
\]

以后交给我们现在正在修好的 executor：

\[
\boxed{
R(s_j,\tau_t)\rightarrow A_j
}
\]

这里 \(A_j\) 是 Inspire 的 native command。

注意：

\[
R
\]

**每一个 control step 都读取实时状态**：

\[
q_j,\ dq_j,\ hand_j,\ object_j,\ velocity_j.
\]

所以即便高层 8 步内暂时使用同一条 \(\tau_t\)，低层并不是开环：

\[
A_j=R(s_j,\tau_t).
\]

它一直根据真实机器人状态纠偏。

---

# 4. 24 步规划，先执行 4～8 步

高层：

\[
T_{\text{plan}}=24.
\]

原因是世界模型需要足够长的未来才能判断抓取后果。

但是：

\[
K_{\text{exec}}<24.
\]

比如现在可以先延续历史设计：

\[
K_{\text{exec}}=8.
\]

执行：

\[
\tau_t[1:8]
\]

对应的低层闭环控制。

到了：

\[
t+8
\]

重新读取：

\[
H_{t+8}
\]

然后：

\[
\pi_\tau(H_{t+8})\rightarrow\tau_{t+8}.
\]

所以整个系统实际上是：

\[
\boxed{
\text{24-step imagination}
+
\text{8-step receding horizon}
}
\]

如果后面发现 contact 阶段 8 步太长，可以直接改成 2/4，不影响整体结构。

---

# 5. 到这里先形成一个纯 trajectory baseline

最先要得到：

\[
\boxed{
H
\rightarrow
\pi_\tau
\rightarrow
\tau
\rightarrow
R
\rightarrow
A
\rightarrow
Z
}
\]

这就是新的 **trajectory-policy baseline**。

它暂时：

- 不用 PointWorld；
- 不用 evaluator；
- 不用 \(Y\)；
- 不用 RLToken。

先证明：

> 把 policy action space 从 native command 换成未来 hand trajectory，是可以完成抓取任务的。

而 \(\pi_\tau\) 后面完全可以 PPO 更新。

---

# 6. 第三层：World Model

有了这个 baseline 后，再接 PointWorld。

但不是一定：

\[
(H,\tau)\rightarrow \hat E
\]

然后把 \(\hat E\) 拼给 policy。

而是：

\[
\boxed{
z^{phys}
=
F_{\rm WM}(H,\tau)
}
\]

其中 \(F_{\rm WM}\) 是 PointWorld backbone。

这个 \(z^{phys}\) 应该包含：

- hand 会怎么运动；
- object 会怎么响应；
- hand-object relative motion；
- 是否逐渐分离；
- 是否形成稳定 holding；
- object velocity / pose tendency；
- contact-related geometry。

也就是：

\[
\boxed{
z^{phys}
=
\text{action-conditioned consequence representation}
}
\]

---

# 7. PointWorld 怎么训练

预训练时仍然显式预测未来：

\[
F_{\rm WM}(H,\tau)=z
\]

然后：

\[
D_E(z)\rightarrow E
\]

\[
D_I(z)\rightarrow I.
\]

甚至可以预测完整未来 hand-object geometry。

所以显式 future prediction 依然重要，但它是：

> **representation learning supervision**

而不一定是最终 policy interface。

训练完之后：

\[
D_E,D_I
\]

部署时可以不用。

留下：

\[
(H,\tau)\rightarrow z^{phys}.
\]

---

# 8. 第四层：RL / RLToken 风格接入

这时候 trajectory policy 已经有 baseline：

\[
\pi_\tau^{base}(H)\rightarrow\tau^{ref}.
\]

世界模型给：

\[
z^{ref}
=
F_{\rm WM}(H,\tau^{ref}).
\]

然后可以有一个小的 RL refinement actor：

\[
\boxed{
\Delta c
=
\pi_{\rm refine}
(H,c^{ref},z^{ref})
}
\]

得到：

\[
c^{RL}=c^{ref}+\Delta c
\]

\[
\tau^{RL}=D(H,c^{RL}).
\]

这就是我们自己的 RLToken-style 结构。

对应关系：

\[
\text{RLT reference action}
\leftrightarrow
\tau^{ref}
\]

\[
\text{RL token}
\leftrightarrow
z^{phys}
\]

\[
\text{RL action}
\leftrightarrow
\Delta c.
\]

---

# 9. Critic 看修改后的 physical consequence

修改以后重新经过 world model：

\[
z^{RL}
=
F_{\rm WM}(H,\tau^{RL}).
\]

critic：

\[
\boxed{
Q(H,c^{RL},z^{RL})
}
\]

预测真正 task return。

于是 actor 优化的是：

\[
\max Q(H,c,z^{phys})
\]

但同时不能离 baseline 太远：

\[
L_{\pi}
=
-Q
+
\lambda\|\Delta c\|^2.
\]

也就是：

> RL 可以改抓取轨迹，但默认相信已有 trajectory policy，只有确实能提高价值时才偏离。

这对我们的 contact manipulation 非常重要。

---

# 10. PPO 和 off-policy 在哪里选

这里其实有两个 RL 层次，不要混。

### 第一层：训练 trajectory baseline

完全可以：

\[
\boxed{
H\xrightarrow{\rm PPO}c\rightarrow\tau
}
\]

真实 Gym reward 训练。

这是你刚才强调的部分，我觉得合理。

### 第二层：WM-informed refinement

等 baseline 稳定以后，可以继续 PPO，也可以切：

- TD3；
- SAC；
- TD3+BC。

我更偏向后者，因为可以重复利用 replay buffer。

但这是后面的选择，不需要现在决定死。

---

# 11. 整个架构最后画成一条线

### Baseline

\[
\boxed{
H_t
\rightarrow
\pi_\tau
\rightarrow
c_t
\rightarrow
D
\rightarrow
\tau_t
\rightarrow
R
\rightarrow
A_t
\rightarrow
\text{Gym}
}
\]

### 加世界模型后

\[
(H_t,\tau_t)
\rightarrow
F_{\rm WM}
\rightarrow
z_t^{phys}.
\]

### 加 RLToken-style refinement 后

\[
\boxed{
\begin{aligned}
c_t^{ref} &= \pi_\tau(H_t)\\
\tau_t^{ref} &= D(H_t,c_t^{ref})\\
z_t^{ref} &= F_{\rm WM}(H_t,\tau_t^{ref})\\
\Delta c_t &= \pi_{\rm refine}(H_t,c_t^{ref},z_t^{ref})\\
c_t^{RL} &= c_t^{ref}+\Delta c_t\\
\tau_t^{RL} &= D(H_t,c_t^{RL})\\
z_t^{RL} &= F_{\rm WM}(H_t,\tau_t^{RL})\\
Q_t &= Q(H_t,c_t^{RL},z_t^{RL})
\end{aligned}
}
\]

然后：

\[
\tau_t^{RL}
\rightarrow
R
\rightarrow
A.
\]

执行若干步后重新观测，重新规划。

---

## 12. 这样原来的 E/I/Y 分别去哪了？

这个也可以彻底理清。

**E/I：**

变成 world-model representation 的监督：

\[
z^{phys}\rightarrow E/I.
\]

不是必须显式输入 actor。

**Y：**

降级成 auxiliary / diagnostic / oracle evidence。

我们已经知道 GT-Y rolling 能改善 Z，所以它证明“未来 consequence 有决策价值”即可。

真正 RL policy 可以直接优化真实 reward / return。

**PointWorld：**

从：

> “预测一堆未来数字给 evaluator”

变成：

> **“给 trajectory-conditioned RL 提供 physical representation”。**

我认为这才是现在最统一的一版架构。

而目前仓库正在做的 generated-\(\tau\) execution，实际上就是在验证最前面的：

\[
\boxed{
H\rightarrow\tau\rightarrow R\rightarrow A
}
\]

能不能成为一个基本可工作的 trajectory baseline。只要这一步站住，后面的 PPO trajectory training、WM token 和 RL refinement 才真正有统一接口。

这个担心是成立的。**如果长期写成**

\[
\tau=\tau^{ref}+\Delta\tau
\]

那确实把 base policy 变成了不可摆脱的“母策略”。后面无论世界模型还是 RL，都只能在它附近修补，最终上限会被 reference 的行为流形限制。

我更推荐把 reference **降级成训练先验，而不是部署时必须存在的动作基座**。

## 更合适的结构：absolute trajectory policy + behavior prior

最终 policy 直接输出绝对的 trajectory latent：

\[
\boxed{
c_t \sim \pi_\theta(c|H_t)
}
\]

然后：

\[
\tau_t=D(H_t,c_t)
\]

再：

\[
\tau_t\rightarrow R\rightarrow A.
\]

所以最终部署链就是：

\[
\boxed{
H
\rightarrow
\pi_\theta
\rightarrow
c
\rightarrow
\tau
\rightarrow
R
\rightarrow
A
}
\]

**这里已经没有 \(c^{ref}\) 或 \(\tau^{ref}\)。**

base policy 只在训练早期出现。

---

## 那原 base policy 怎么利用？

把现有 baseline rollout 学成一个 behavior prior：

\[
p_\beta(c|H)
\]

它描述：

> 在这个状态附近，已有成功/可执行行为通常落在哪些 trajectory latent 区域。

RL actor 自己是：

\[
\pi_\theta(c|H).
\]

训练目标不是 residual，而是：

\[
\boxed{
J(\pi)
=
\mathbb E[Q(H,c,z^{phys})]
-
\lambda
D_{\mathrm{KL}}
\left(
\pi_\theta(c|H)
\|
p_\beta(c|H)
\right)
}
\]

早期：

\[
\lambda\ \text{较大}
\]

避免策略跑到完全不可执行的轨迹空间。

随着 RL 真正学出东西：

\[
\lambda\downarrow0.
\]

最后：

\[
\boxed{
\pi_\theta
\text{ 可以完全离开旧 base policy}
}
\]

这和永久 residual 有本质区别。

---

# 这样 base policy 的地位就合理很多

它只负责三件事：

1. 提供初始可执行数据；
2. 初始化 trajectory policy；
3. 训练初期提供安全的行为先验。

它**不规定最终策略必须**

\[
\tau^{ref}+\Delta\tau.
\]

可以把过程理解成：

\[
\text{Base policy}
\rightarrow
\text{initial prior}
\rightarrow
\text{RL policy improvement}
\rightarrow
\text{standalone trajectory policy}.
\]

---

# PointWorld 仍然怎么接？

这一部分不变，而且反而更自然。

对 actor 给出的**任意绝对 trajectory**：

\[
\tau=D(H,c)
\]

PointWorld backbone：

\[
z^{phys}=F_{\rm WM}(H,\tau).
\]

critic：

\[
\boxed{
Q_\phi(H,c,z^{phys})
}
\]

于是 world model 在告诉 critic：

> “当前状态下，如果执行这个 actor 自己提出的 trajectory，它对应怎样的未来物理后果表示？”

这里完全不依赖 reference policy。

最终 actor 优化：

\[
\max_c Q(H,c,F_{\rm WM}(H,D(H,c))).
\]

这才是我认为长期应该达到的形态。

---

# 甚至 RLToken 也不需要绑定 reference action

我们可以借 RLToken 的 **compact physical representation** 思想，但不必继承它“永远围绕 reference chunk”的结构。

我们的 token 可以直接是：

\[
\boxed{
z^{phys}=F_{\rm WM}(H,\tau)
}
\]

于是：

\[
H
\rightarrow
\pi(c|H)
\rightarrow
\tau
\rightarrow
z^{phys}
\rightarrow
Q.
\]

这其实更像：

> **world-model-conditioned actor-critic**

而不是严格意义上的 residual RLT。

我觉得这更适合我们的研究目标。

---

# 还有一个我更喜欢的后续方案：planning → distillation

如果 PointWorld 最后足够好，还可以进一步把 base policy 完全拿掉。

当前 actor 给一个初始 trajectory：

\[
\tau^{(0)}.
\]

在 latent \(c\) 空间里，用 critic/world model 做少量优化：

\[
c^*
=
\arg\max_c
Q(H,c,F_{\rm WM}(H,D(H,c))).
\]

可以是：

- CEM；
- gradient ascent；
- sampling；
- MPPI。

得到：

\[
\tau^*=D(H,c^*).
\]

然后把这个 improved trajectory 再蒸馏回 actor：

\[
\pi_\theta(H)\rightarrow c^*.
\]

不断循环：

\[
\boxed{
\text{Actor proposal}
\rightarrow
\text{WM/Q improvement}
\rightarrow
\text{distill back into Actor}
}
\]

这非常漂亮，因为最终 actor 会逐渐吸收 planner 的改进。

最后 inference 时甚至可以只跑 actor：

\[
H\rightarrow c\rightarrow\tau.
\]

也可以保留少量在线 planning。

---

# 我现在会把整个路线分成三个阶段

### 阶段 1：建立 trajectory action space

先有：

\[
H\rightarrow c\rightarrow\tau\rightarrow R\rightarrow A.
\]

base policy rollout 只用于初始化 \(D\) 和 \(\pi\)。

---

### 阶段 2：prior-regularized RL

训练：

\[
\pi_\theta(c|H)
\]

critic 使用：

\[
Q(H,c,F_{\rm WM}(H,\tau)).
\]

早期用：

\[
KL(\pi||p_{\rm behavior})
\]

防止乱跑。

然后逐渐减弱这个约束。

---

### 阶段 3：摆脱 base policy

当 RL policy 已经有稳定性能：

\[
\lambda_{\rm prior}\rightarrow0.
\]

最终变成：

\[
\boxed{
H
\rightarrow
\pi_\theta
\rightarrow
c
\rightarrow
\tau
\rightarrow
R
\rightarrow
A
}
\]

PointWorld 只是提供 physical representation / critic support。

旧 base policy 完全退出在线系统。

---

## 我认为这比永久 residual 更合理

永久 residual：

\[
\tau^{ref}+\Delta\tau
\]

适合**快速工程增益**，但作为最终研究路线确实容易受 baseline 上限限制。

更完整的科研路线应该是：

\[
\boxed{
\text{behavior prior}
\neq
\text{behavior constraint}
}
\]

也就是：

> **借 base policy 起步，但不让它定义最终策略空间。**

如果让我现在重新定架构，我会把这个作为主版本，而把 `reference + residual` 只保留成一个训练初期稳定化 baseline。


可以。真要给这套架构泼冷水，我觉得它现在最大的危险不是“理论上不成立”，而是**模块太多、每一层都有合理解释，最后很容易做成一个什么都能解释但很难证明哪一部分真的有用的系统**。

我会重点质疑下面这些地方：

1. **\(c\) 这个 trajectory latent 可能本身就是人为制造的新瓶颈。**  
   我们为了避免直接优化 792 维 \(\tau\)，引入
   \[
   H\rightarrow c\rightarrow\tau
   \]
   但问题是：什么样的 \(c\) 才是“对抓取有意义”的？如果只是 PCA，它可能只保留方差最大的运动，而不是最关键的接触调整；如果是 learned latent，又会引入新的 representation learning 问题。最后可能出现：
   \[
   c\text{ reconstruction 很好}
   \]
   但真正需要的 thumb preload / contact correction 根本不在容易优化的 latent 方向里。那我们只是把“高维难优化”换成“低维但错误的动作空间”。

2. **PointWorld token 很可能对 critic 是冗余的。**  
   critic 已经输入：
   \[
   H,\ c
   \]
   而 \(z^{phys}=F(H,\tau)\) 又是 \(H,c\) 的确定函数。理论上一个足够强的 critic 完全可以自己学：
   \[
   Q(H,c)
   \]
   根本不需要 world-model token。也就是说：
   \[
   Q(H,c,z^{phys})
   \]
   提升了，不一定是因为“未来物理理解”有价值，可能只是多了一个大网络做 feature engineering。  
   这其实是整条路线最危险的 scientific risk：
   > **world model 可能不是必要组件。**

3. **critic 极容易绕过 \(z^{phys}\)。**  
   这是我们以前 Cm 接法反复遇到的问题。即使给它：
   \[
   z^{phys}
   \]
   它也可能只靠：
   \[
   H,c
   \]
   把 return 拟合掉。然后表面上架构里有 world model，实际上策略完全没用它。  
   所以必须有严格的：
   \[
   z\text{-shuffle},\quad z=0,\quad \text{stop-grad/control}
   \]
   不然最后很容易自欺。

4. **PointWorld 学到的 representation 未必是 RL 真正需要的 representation。**  
   现在 PointWorld 主要靠：
   \[
   E/I\text{ future prediction}
   \]
   训练。  
   但“未来物体位置预测得准”不等于“能区分哪个动作的长期 return 更高”。这就是典型的 representation-objective mismatch。  
   我们已经见过：
   \[
   E/I_{\rm GT}\rightarrow Y
   \]
   有用，但
   \[
   \hat E/\hat I\rightarrow Y
   \]
   信息保留不好。  
   所以 backbone latent 也可能只是“动力学好用”，而不是“控制价值好用”。

5. **\(H\) 本身可能不够 Markov。**  
   我们现在倾向用纯历史测量：
   \[
   H_{t-3:t}
   \]
   但灵巧手接触状态里隐藏变量很多：
   - 接触法向；
   - 摩擦状态；
   - solver warm state；
   - 实际 preload；
   - 微小 slip；
   - 未观测接触对。  
   如果：
   \[
   H_t
   \]
   对未来动力学不是充分状态，那再漂亮的 actor/critic/world-model 都可能被 partial observability 限死。  
   我们仓库里 parallel env 同样输入 tick1 就出现 state divergence，这已经是很现实的警告。

6. **trajectory action horizon 可能仍然太长。**  
   即便低层 executor 每步闭环，高层：
   \[
   \tau_{t:t+24}
   \]
   还是在一次决策里规定了 0.8 秒意图。对于 contact manipulation，这可能太刚性。  
   你可以 2～4 步 replanning，但 critic 评价的仍是一整条 24-step trajectory。于是：
   \[
   \text{计划后半段}
   \]
   很可能从来不会真正执行，却影响当前 Q。  
   这会造成典型的 receding-horizon mismatch。

7. **executor \(R\) 可能成为真正的瓶颈，而我们却把注意力转移到 RL。**  
   即使 actor 找到了理想的：
   \[
   \tau^*
   \]
   如果：
   \[
   R(s,\tau^*)\rightarrow A
   \]
   对 OOD trajectory 不稳定，那么 RL 会被迫学习“哪些轨迹 executor 能执行”，而不是“哪些轨迹任务上最好”。  
   最终 Q 可能学到的是：
   > 哪种 \(\tau\) 不会把 executor 搞崩。
   
   而不是：
   > 哪种 \(\tau\) 具有更好的 hand-object consequence。

8. **behavior prior 即使逐渐衰减，也可能产生隐性上限。**  
   理论上：
   \[
   \lambda_{\rm prior}\rightarrow0
   \]
   后可以摆脱 base policy。  
   但实际上 replay buffer、critic 初始化、数据覆盖、actor initialization 全部来自 base behavior。即使 KL 已经关掉，policy 也可能从没见过真正不同的好行为区域。  
   所以：
   \[
   \text{soft prior}
   \]
   仍然可能变成一个事实上的 support constraint。

9. **off-policy RL 在接触任务里并不天然比 PPO 容易。**  
   我刚才说 TD3/SAC 可以复用数据，这是对的；但代价是：
   \[
   Q\text{ extrapolation error}
   \]
   很可能非常严重。  
   尤其我们的 action 是 trajectory latent，actor 一旦走出数据分布：
   \[
   Q(H,c)
   \]
   很容易高估从没执行过的 \(c\)。  
   然后 actor 就会钻 critic 的漏洞。  
   这种问题甚至比 PPO 更难排查。

10. **整套架构可能过重，论文故事容易失焦。**  
    现在已经有：
    \[
    H\rightarrow c\rightarrow\tau\rightarrow R\rightarrow A
    \]
    再加：
    \[
    (H,\tau)\rightarrow z^{phys}
    \]
    再加 twin critic、behavior prior、trajectory decoder、receding horizon。  
    任何一个模块失败，都可以再解释“是另一模块的问题”。  
    这在科研上很危险，因为很容易变成一个工程大杂烩，而不是一个单一、可证伪的方法。

---

### 我觉得最致命的两个风险

如果只让我选两个：

**第一是：world-model token 可能根本没必要。**

最终最简单的：

\[
Q(H,c)
\]

可能就已经足够。

如果这样，那我们整个 PointWorld 主线会变成旁枝。

**第二是：trajectory latent / executor 组合可能限制 policy 比原 native-action policy 更严重。**

也就是说，我们以为：

\[
\tau
\]

是更好的 planning interface，

但实际上可能：

\[
H\rightarrow a
\]

才是更灵活的控制接口。

如果 trajectory parameterization 丢掉接触所需的细粒度控制，那整条路线从根上就错了。

---

## 所以我不会直接把整套架构全做出来

我会先做三个非常残酷的 falsification。

先比较：

\[
\boxed{
\text{native PPO}
\quad vs\quad
\text{trajectory policy}
}
\]

如果 trajectory baseline 自己都明显更差，先别谈 WM。

然后比较：

\[
\boxed{
Q(H,c)
\quad vs\quad
Q(H,c,z^{phys})
}
\]

如果 physical token 不提高 held-out return prediction / policy learning，就停 PointWorld-RL 这条线。

最后比较：

\[
\boxed{
\text{trajectory RL without WM}
\quad vs\quad
\text{trajectory RL + WM}
}
\]

只有第三步真的有收益，整个故事才站得住。

所以现在这套架构我会把它当成**一个待证伪的假设**，不是既定答案：

\[
\boxed{
\text{trajectory action space}
+
\text{world-model representation}
\stackrel{?}{\Longrightarrow}
\text{better RL}
}
\]

如果最简单的 \(Q(H,c)\) 就能做到一样好，那应该果断删掉 world model；如果 native-action RL 明显更好，那也应该果断删掉 trajectory policy。这样的路线才不会再陷入“每个模块都能讲故事，但整体没有净收益”的循环。