对，这个判断是对的。**如果我们直接要求模型学**

\[
(H,G_{t+24})\rightarrow \tau_{1:24},
\]

它实际上在解一个“给定终点，反推出中间手动作”的 inverse planning 问题。这个问题比普通的未来预测难，数据需求也更高，因为同一个 \(G\) 往往对应很多条可行 \(\tau\)。

但我觉得现在要区分两件事：

**第一，当前实验远没有到“数据太少所以不行”的程度。** 现在三源 PointWorld corpus 有约 **437.8 万个 train anchors**，而最新 GoalPointWAM 的正常训练只有 600 updates、microbatch=2，也就是只经历了约 1200 个窗口的更新。它现在几乎不响应 goal，不能直接解释成“这个方向需要的数据多到不可行”；更像是我们还处在极短的 probe 阶段。

不过更重要的是第二点：

\[
\boxed{\text{光增加数据量，不一定能解决 goal 被忽略。}}
\]

因为在普通 observational video 数据里，通常是：

\[
H_i \rightarrow G_i,\tau_i
\]

每个历史状态 \(H_i\) 只对应它真实发生过的那一个 future。于是模型完全可以学：

\[
H\rightarrow\tau
\]

而忽略 \(G\)，照样把训练 loss 做得不错。

这就是现在最值得警惕的地方。

---

## 所以我觉得不应该全靠 \(G\to\tau\) 的监督学习

更合理的是让 **world model 真正参与“从 goal 反推 action”**，而不是只当 auxiliary loss。

我们已经有前向模型：

\[
\boxed{
W(H,\tau)\rightarrow \hat E
}
\]

它负责回答：

> 给定当前状态和这条手轨迹，物体实际上会怎么运动？

那 action model：

\[
A(H,G)\rightarrow\hat\tau
\]

之后，可以直接接：

\[
\hat E=W(H,\hat\tau).
\]

然后增加一个非常自然的目标：

\[
\boxed{
L_{\rm goal}
=
d(\hat E_{24},G)
}
\]

也就是：

> 你生成的手轨迹经过物理模型以后，最终物体到底有没有到我给的目标？

于是整个链变成：

\[
\boxed{
(H,G)
\rightarrow
\hat\tau
\rightarrow
W(H,\hat\tau)
\rightarrow
\hat E_{24}
\rightarrow
G
}
\]

这时候不是让 actor 凭数据“背出来”：

\[
G\rightarrow\tau.
\]

而是给它一个**物理约束**：

\[
\boxed{
\text{只要 }\tau\text{ 能通过世界模型实现 }G\text{ 就可以。}
}
\]

---

# 这样训练会比纯 imitation 合理很多

大规模视频阶段我们有真实：

\[
(H,\tau^{human},E^{actual}).
\]

先训练 world model：

\[
L_E=
d(W(H,\tau^{human}),E^{actual}).
\]

同时 action branch 学：

\[
L_\tau=
d(A(H,G),\tau^{human}),
\]

其中：

\[
G=E^{actual}_{24}.
\]

然后再加：

\[
\boxed{
L_{\rm goal}
=
d(
W(H,A(H,G))_{24},
G
)
}
\]

总的：

\[
\boxed{
L=
L_\tau
+
\lambda_E L_E
+
\lambda_G L_{\rm goal}.
}
\]

这三项的功能完全不同：

- \(L_\tau\)：给动作模型一个人类动作先验；
- \(L_E\)：让 world model 学真实物理；
- \(L_{\rm goal}\)：强迫 action model **真的使用 G**。

这最后一项正好针对当前仓库出现的：

\[
G+10\text{cm}
\]

但预测几乎一点不动的问题。

---

## 而且这会降低对“一模一样动作标签”的依赖

假设对于同一个目标：

\[
G
\]

人类 A 用左边绕过去，人类 B 从右边过去。

纯：

\[
L_\tau
\]

会觉得其中一个是“错误答案”，因为它要拟合唯一的 GT trajectory。

但：

\[
L_{\rm goal}
\]

只在乎：

\[
W(H,\tau)\rightarrow G.
\]

所以两条都可以是正确的。

这对后面大规模视频非常重要。

---

# 但这里有一个条件

前提是：

\[
W(H,\tau)
\]

本身得先学得比较可信。

否则 actor 会利用 world model 的漏洞，找到一些：

\[
\tau_{\rm weird}
\]

让预测模型认为能到 goal，但真实物理里完全不行。

所以训练顺序我会改成：

### Stage A：先把 forward dynamics 做扎实

大量 human video：

\[
(H,\tau_{\rm human})
\rightarrow E_{\rm actual}.
\]

这是最适合大数据训练的部分，因为它不涉及 inverse ambiguity。

这也是我们现有 PointWorld 最成熟的任务。

---

### Stage B：再学 goal-conditioned action

初始化：

\[
(H,G)\rightarrow\tau
\]

用人类 hand trajectory 做 BC。

同时先冻结或慢更新 \(W\)：

\[
A(H,G)
\rightarrow\tau
\rightarrow W(H,\tau)
\rightarrow E
\]

加：

\[
L_{\rm goal}.
\]

这样就真正是：

> 利用一个已经学好的世界模型，把目标反演成动作。

---

### Stage C：机器人适配

加入：

\[
L_{PD}
=
\|R(s,\hat\tau)-q^{des}_{teacher}\|.
\]

再做现在已经成功的 PD-DAgger。

---

# 这也解释了为什么 PointWAM 要那么大规模数据

PointWAM 的优势不是“网络结构神奇，所以少量数据就会从目标想出动作”。

它本质上就是依赖大规模：

\[
\text{scene motion}+\text{hand motion}
\]

共演化数据学先验。

我们如果也想做到：

\[
G\rightarrow\text{合理的中间动作}
\]

确实需要大量多样数据。

但我们的好处是：

\[
\boxed{\text{最需要大数据的 forward world learning 可以直接用人类视频。}}
\]

只有最后 robot execution：

\[
\tau\rightarrow q^{des}
\]

才依赖相对少量机器人数据。

这其实正是我们想要的分工。

---

# 我现在对当前仓库的判断也会因此调整

当前最新结果：

- fixed 6 windows 可以拟合；
- mixed-source 600 updates 不行；
- goal response 基本为 0；

我不会立刻去改 GoalEncoder 架构。

因为现在甚至还没真正测试：

\[
\boxed{
L_{\rm goal}=d(W(H,A(H,G)),G)
}
\]

这一条闭环一致性。

当前只是：

\[
(H,G)\rightarrow\tau
\]

和：

\[
(H,\tau_{\rm executed})\rightarrow E
\]

两个 branch 共享 backbone，**世界 branch 并没有直接约束 actor 必须满足 G**。

所以目前的“WAM”其实还差了非常关键的一环。

---

## 我更推荐最终把架构定成

\[
\boxed{
\begin{aligned}
\hat\tau &= A_\theta(H,G)\\
\hat E^{teacher} &= W_\phi(H,\tau_{\rm executed})\\
\hat E^{policy} &= W_\phi(H,\hat\tau)
\end{aligned}
}
\]

训练：

\[
\boxed{
L=
L_\tau
+\lambda_E L_E
+\lambda_G d(\hat E^{policy}_{24},G)
}
\]

robot 阶段再：

\[
+\lambda_{PD}L_{PD}.
\]

这样 goal 不再只是一个“输入提示”，而是变成**对 action 的物理约束**。

我觉得这比继续单纯扩大 \(H,G\to\tau\) 的监督训练更合理，也更能解释为什么我们需要 PointWorld，而不是随便一个大 Transformer。


有，而且你们现在遇到的其实很典型。只是论文里通常不叫“WAM 不收敛”，而是两个更具体的问题：

\[
\boxed{\text{condition collapse}}
\]

模型发现不用 goal 也能把 loss 降下来，于是忽略 goal；

以及

\[
\boxed{\text{action-agnostic world prediction}}
\]

世界模型学成“这个任务通常接下来会发生什么”，而不是“**这个动作**会导致什么”。

近几个月的 WAM 工作已经专门在处理这两个问题。一个关键结论是：**基本没人靠“端到端丢进去，然后等它自己学会”来保证。**

### 1. SelfWAM 遇到的几乎就是我们的 world-branch 问题

SelfWAM 明确指出：如果 future prediction 只看当前 observation 和 task prompt，模型很容易学成 generic task progression，而不是 action-specific consequence。

它的解决办法不是加数据这么简单，而是直接改信息流：

\[
(H,a_{\text{executed}})
\rightarrow E
\]

world branch 明确看到一份 **clean demonstrated action**；同时 action-prediction branch 又不能偷看到这份 GT action 或未来帧。

也就是：

\[
\boxed{\text{teacher-force 真实执行动作给 world model}}
\]

并用 attention mask 防止标签泄漏。它还加入 future robot self-mask，让预测更集中在动作真正造成的变化，而不是背景纹理。[arXiv](https://arxiv.org/abs/2608.00725?utm_source=chatgpt.com)

这和我们现在坚持：

\[
W(H,\tau_{\rm executed})\rightarrow E
\]

完全一致。

所以这一块我们的方向没有问题。

---

## 2. ViGAR 遇到的更像我们现在的 **G 被忽略**

这个我觉得尤其值得你注意。

ViGAR 做 visual-subgoal-conditioned WAM，他们真的做了 **goal injection route ablation**。

结果是：

- Goal 注入 generator 的 VAE latent：平均 **62.17**
- Goal 同时进 generator + reasoner：61.72
- Goal 只进 reasoner ViT tokens：50.97
- **完全不给 goal：49.18**

也就是说：

\[
\boxed{
\text{“把 goal 作为几个 token 给 Transformer”可能几乎等于没给。}
}
\]

这跟我们现在：

\[
G\rightarrow GoalEncoder\rightarrow tokens\rightarrow TrajectoryDecoder
\]

然后 causal response 几乎为零，症状非常像。

ViGAR发现目标的**注入位置和表示方式**非常重要。goal 真正进入 generative pathway，比只是作为 attention memory 强得多。[大组网站](https://dagroup-pku.github.io/ViGAR/?utm_source=chatgpt.com)

这一点对我们当前架构是一个警告：

> 现在的问题不一定是数据量，而可能是 \(G\) 作为 side-memory token 太容易被绕过。

---

## 3. ViGAR 还发现“监督本身不一致”会让 goal learning 崩掉

它还有一个很相关的问题。

在 subtask 边界：

两个视觉上几乎一样的 observation，

却可能一个被标：

\[
G_i
\]

另一个被标：

\[
G_{i+1}.
\]

这相当于：

\[
H\approx H'
\]

但：

\[
G\neq G'.
\]

网络会学得很混乱。

所以他们专门做 **subgoal look-ahead relabeling**，让相近状态对应一致目标。

同时，普通 image loss 被大量静态背景支配，于是又对 end-effector ROI 加了 4:1 权重。

去掉这两个处理后，RoboTwin Random 成功率从 67.02% 掉到 52.40%。[大组网站](https://dagroup-pku.github.io/ViGAR/?utm_source=chatgpt.com)

这个对我们也很有意义：

我们以后不能让大量：

\[
P_t\approx P_{t+24}
\]

的静止/弱运动窗口占训练主体。

否则最优解就是：

\[
\boxed{\text{预测“不怎么动”}}
\]

然后自然忽略 \(G\)。

---

# 4. Act2Goal 不直接从零 joint train

Act2Goal 是目前和我们“给 goal，自己想象中间过程”最接近的路线之一。

它不是简单：

\[
(o_t,G)\rightarrow a
\]

硬学。

而是：

\[
(o_t,G)
\rightarrow
\text{intermediate visual trajectory}
\]

再让 action expert 通过 cross-attention 使用这些 intermediate states。

而且训练是明显分阶段的：

\[
\boxed{\text{先 world-model / large-scale imitation，再 joint training}}
\]

之后 rollout 还使用 **hindsight goal relabeling**：

实际 rollout 最后到了状态 \(s'\)，就把 \(s'\) 重新作为 goal，构造新的：

\[
(s,G=s')
\]

训练样本。

这非常关键，因为它打破了原始数据里：

\[
H_i\leftrightarrow G_i
\]

几乎一一绑定的问题，增加同一个状态附近的 goal diversity。[arXiv](https://arxiv.org/abs/2512.23541?utm_source=chatgpt.com)

所以 Act2Goal 的解决方案其实就是：

\[
\boxed{
\text{大量预训练}
+
\text{分阶段}
+
\text{goal relabeling}
}
\]

而不是期待 endpoint goal 自己被学会。

---

# 5. PointWAM为什么看起来更容易训？

因为它实际上没解决我们现在这个更难的问题。

PointWAM主要学：

\[
(H,\text{language})
\rightarrow
(\tau_{\rm hand},E_{\rm scene})
\]

而训练数据直接有：

\[
\tau_{\rm hand}^{GT},
\qquad
E_{\rm scene}^{GT}.
\]

它不是：

\[
(H,G_{\rm endpoint})
\rightarrow
\text{自己反推出唯一的 }\tau.
\]

而且 PointWAM先用 **115万个人类 episode** 做 human pretraining，再在 robot demonstrations 上 fine-tune；scene trajectory supervision只监督 shared trunk，论文报告这部分带来额外 10.9 个百分点的收益。[arXiv](https://arxiv.org/abs/2610.02840?utm_source=chatgpt.com)

所以它的 recipe 本质上是：

\[
\boxed{
\text{dense future supervision}
\rightarrow
\text{大规模 human pretrain}
\rightarrow
\text{robot fine-tune}
}
\]

这比我们现在仅靠 endpoint \(G\) 反推 24 步动作容易很多。

---

# 6. FastWAM 给了另一个非常重要的结论

FastWAM发现：

> **WAM 的主要价值可能不是测试时真的生成未来，而是训练时 future prediction 改善 representation。**

它保留 video/world co-training，但部署时直接 action-only，性能依然很强；去掉 video co-training 的损失反而更明显。[arXiv](https://arxiv.org/abs/2603.16666?utm_source=chatgpt.com)

而且它使用 pretrained video DiT，不是从随机初始化训练一个 world-action joint model。

这说明：

\[
\boxed{\text{世界模型不一定要成为 actor 的可微 planner。}}
\]

它可以首先只是：

\[
L_E
\]

对 shared representation 的强辅助监督。

这其实对我们很重要。

---

# 7. DreamTrajectory让我会修正一下我刚才给你的建议

我上一条说可以直接：

\[
A(H,G)\rightarrow\tau
\rightarrow W(H,\tau)
\rightarrow E
\]

然后用：

\[
L_{\rm goal}=d(E,G)
\]

反传进 actor。

**这个可以做，但我现在不建议作为第一步。**

为什么？

因为 world model 目前还不够准时，actor 很容易学会 exploit：

\[
W
\]

的误差。

DreamTrajectory采取了更保守的路线。

它分别学：

\[
\text{planned trajectory}
\]

和：

\[
W(H,a)\rightarrow \text{induced trajectory}.
\]

然后测试时生成多个 candidate action：

\[
a_1,\ldots,a_N
\]

分别预测：

\[
W(H,a_i)=\hat\tau_i
\]

最后选：

\[
a^*
=
\arg\min_i
d(\hat\tau_i,\tau_{\rm plan}).
\]

也就是说，**不需要通过不完美 world model 直接反向训练 actor**，而是用 world model 做 candidate ranking。[arXiv](https://arxiv.org/abs/2608.01381?utm_source=chatgpt.com)

这个其实和我们过去 oracle/candidate ranking 的经验非常契合。

---

# 8. InternW0-Δ 更能说明现在大型 WAM 是怎么“训稳”的

它不是一个网络随机初始化后直接 joint train。

而是：

- pretrained video expert；
- 单独 action expert；
- masked information flow；
- training-only future supervision；
- Causal Imprint tokens；
- frozen 4D foundation model 做 distillation；
- 超过 **20K hours** 的 heterogeneous data；
- 再针对下游任务 post-train。[arXiv](https://arxiv.org/abs/2609.31394?utm_source=chatgpt.com)

换句话说，现在比较成功的 WAM 基本都是：

\[
\boxed{
\text{强预训练}
+
\text{结构化信息流}
+
\text{密集辅助监督}
+
\text{分阶段 fine-tune}
}
\]

而不是一个普通 supervised learning problem。

---

# 所以“别人怎么保证收敛”可以总结成 6 条

1. **World branch teacher forcing**

\[
W(H,a_{\rm executed})\rightarrow E_{\rm actual}
\]

绝不用早期 actor 的烂 action 当 world supervision。

2. **防止 shortcut / leakage**

action head不能偷看未来，world head必须看 executed action；SelfWAM/FastWAM 都通过 asymmetric attention mask 做这个。[SelfWAM](https://selfwam.github.io/?utm_source=chatgpt.com)

3. **目标必须真的进入主生成路径**

ViGAR已经直接证明，只把 goal 当普通 side token 可能作用很弱。[大组网站](https://dagroup-pku.github.io/ViGAR/?utm_source=chatgpt.com)

4. **增强 goal diversity**

用 goal relabeling / hindsight，而不是永远：

\[
H_i\rightarrow G_i
\]

一对一绑定。Act2Goal就是这么干的。[arXiv](https://arxiv.org/abs/2512.23541?utm_source=chatgpt.com)

5. **先预训练 world representation，再 joint train**

PointWAM、Act2Goal、FastWAM、InternW0 基本都属于这个范式。[Chunghyun Park](https://chrockey.github.io/PointWAM/?utm_source=chatgpt.com)

6. **world loss强调 action-relevant changes**

不能被静态点/背景淹没。ViGAR做 ROI weighting，SelfWAM做 robot self-mask，4D-WAM加入几何一致性监督。[大组网站](https://dagroup-pku.github.io/ViGAR/?utm_source=chatgpt.com)

---

## 对我们现在最重要的判断

我现在不认为应该直接：

> “把 437 万 windows 全部训练一遍，看看会不会自己学会 G。”

风险太大。

当前结果已经告诉我们：

\[
\boxed{G\text{ 有 attention，但几乎没有 causal influence}}
\]

这更像**conditioning contract 问题**，不是单纯 epoch 不够。

我建议我们的路线变成：

\[
\boxed{\textbf{Stage 1}}
\]

先把：

\[
W(H,\tau_{\rm executed})\rightarrow E
\]

作为独立 forward dynamics 用现有 PointWorld 数据真正训好。

然后：

\[
\boxed{\textbf{Stage 2}}
\]

做：

\[
A(H,G)\rightarrow\tau
\]

但重点解决 **goal injection**，而不是马上扩大数据。

最值得模仿 ViGAR 做一个小型三臂实验：

\[
\begin{array}{ll}
A:& G\text{作为现在的memory tokens}\\
B:& G\text{直接注入trajectory query/generative latent}\\
C:& \text{no-goal}
\end{array}
\]

然后不用看最终任务成功率，先看我们已经有的 counterfactual test：

\[
G\rightarrow G+\Delta x
\]

是否产生：

\[
\hat\tau\rightarrow\hat\tau'.
\]

如果：

\[
B\gg A\approx C,
\]

那我们就基本定位到了和 ViGAR 很类似的问题。

之后再决定是否扩大到百万级数据。

**所以目前最值得参考的不是“PointWAM用了多少数据”，而是 ViGAR 的 goal injection ablation + SelfWAM 的 asymmetric conditioning。** 这两篇和我们现在具体遇到的 failure mode 最接近。