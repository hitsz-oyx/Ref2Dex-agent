我把当前 `main` 上的 `consequence-evaluator` 看了一遍。结论很明确：

**你们现在的标签方向并没有错，但把“严格状态匹配的局部 preference”当成主要训练标签，不合适。它应该降级成验证标签。真正训练 evaluator 的主标签应该是 Gym rollout 自动产生的“后续任务价值”。**

当前代码里，`supervision.py` 实际只生成两种排序：

\[
\text{maintained\_hold} > \text{unrecovered\_drop}
\]

\[
\text{lift\_achieved} > \text{grasp\_lost}
\]

而且还要求 same expert / motion / phase，并且当前 object、hand、完整 history \(H\) 都足够接近。这个定义很干净，但代价已经在你们自己的结果里暴露出来了：216 条 episode 做完之后，当前 H-matched 规则下实际只有大约 **1/1/1 个独立 train/val/test episode pair**；聚焦 hold 阶段又采了 48 条 episode，最终还是只有 **1 个 train pair**。这已经不是模型问题，而是监督密度的问题。

### 我建议最后把标签定义成下面这一套

核心不是一个手工 reward，而是一个**分层价值标签**：

\[
\boxed{
Y_t =
\left(
S_t,\;
P_t,\;
M_t
\right)
}
\]

其中最重要的是 \(S_t\)。

- **\(S_t\)：最终任务成功，主标签。**  
  从时刻 \(t\) 执行已经确定的 24 步 residual plan \(\delta_{t:t+23}\)，然后继续执行固定 baseline/expert，最终是否完成任务。对于现在的 airplane 抓取，可以定义为“物体被稳定抓持、抬升并保持”，而不是直接拿 native PPO reward 当成功。它实际上是在监督
  \[
  Q(H_t,\delta_t)
  \approx
  P(\text{最终成功}\mid H_t,\delta_t).
  \]
  这是 evaluator 最应该预测的东西。

- **\(P_t\)：过程进度，辅助标签。**  
  不要所有失败轨迹都标成完全一样的 0。用物理状态自动得到阶段，例如
  \[
  \text{未形成有效交互}
  \rightarrow
  \text{抓持}
  \rightarrow
  \text{抬升}
  \rightarrow
  \text{稳定保持}.
  \]
  这个思想和 Robometer 很接近：Robometer 本身就是用 expert trajectory 的 progress supervision，再用成功/失败、优劣 trajectory 做 preference supervision，而不是只靠最终 binary success。公开实现现在甚至明确保留了 progress、success、preference 三种 head。[GitHub](https://github.com/robometer/robometer?utm_source=chatgpt.com)

- **\(M_t\)：物理 margin，辅助区分同阶段样本。**  
  比如稳定抬升高度、保持持续时间、hand-object 几何距离等。这里我反而不建议把当前 net contact force proxy 当核心，因为你们代码自己已经验证了它只是 proxy，不是精确的 hand-object collision pair。可以保留 force 做诊断，但主监督优先用 object height + hand-object geometry + persistence。

- **Preference 不再手工标，而是从 \(Y\) 自动派生。**  
  比较两个 rollout 时采用类似字典序：
  \[
  S_A>S_B
  \]
  优先；如果 \(S_A=S_B\)，再比较 \(P\)；仍然相同再比较 \(M\)。差别很小时直接 tie / abstain。这样不用人为选择“0.37 lift + 0.28 contact + 0.35 reward”这种很难解释的权重。

最关键的改动其实是：

> **训练 preference 不需要再要求 H 完全匹配。**

你们现在把“训练 evaluator”和“证明 Z 是否提供额外信息”混在一起了。

如果我要证明

\[
E_{\rm oracle}(H,\delta,Z)
>
E_0(H,\delta),
\]

那么测试集上做严格 H/object/hand-matched comparisons 很合理，因为这样能隔离 \(Z\) 的作用。

但如果是在**训练一个 \(Q(H,\delta,Z)\)**，H 本来就是输入。完全没有必要找到两个近乎相同的 \(H\) 才能训练。你们的 residual 本来就是在未来 outcome 发生前随机分配的，这已经给了 action variation。强行 H-match 才把几百条 trajectory 压缩成了 1 个 pair。

所以我会把你们当前管线拆成：

\[
\text{所有 Gym rollout}
\rightarrow
(S,P,M)
\rightarrow
\text{训练 evaluator}
\]

而：

\[
\text{严格 H-matched pairs}
\rightarrow
\text{只用于 held-out oracle headroom evaluation}.
\]

这一个改动，我认为比继续调整当前 `local_preferences()` 阈值重要得多。

---

### 为什么不直接用 Gym reward / return-to-go？

我不建议把它作为主标签。

你们的 native reward 本质上还带着 reference tracking、控制 shaping 等设计。如果直接：

\[
Y_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
\]

那 evaluator 最后很可能学成“原 PPO reward 的复刻器”。它是否真的认为一个动作有利于**最终抓取成功**，并不确定。

这也是你之前一直担心“标签是否和 success 正相关”的根本问题。

如果用我上面的 \(S_t\)，则至少监督目标在定义上就是和 success 对齐的：

\[
S=1 \Rightarrow \text{成功},\qquad
S=0 \Rightarrow \text{失败}.
\]

**模型在新数据上的预测和成功率是否仍然相关不能数学保证，但标签本身不再绕一道 shaping reward。**你只需要在 held-out rollout 上测：

\[
\operatorname{AUC}(\hat S,S),
\quad
\operatorname{Spearman}(\hat V,S/P),
\quad
\text{candidate top-1 success gain}.
\]

最后一项才是真正需要的结果。

---

### 还有一个很重要的细节：不要只看未来 24 步是否成功

Evaluator 输入可以仍然是：

\[
(H_t,\delta_{1:24},Z_{1:24}),
\]

但是它的价值标签最好看**更长的后续结果甚至 episode end**：

\[
(H_t,\delta_{1:24},Z_{1:24})
\rightarrow
Y_{\rm long}.
\]

否则一个动作可能 24 步内把物体抬得很高，但第 35 步直接掉了，你却把它标成好动作。

这正是 evaluator 的意义：

> PointWorld 负责告诉 evaluator “这 24 步会发生什么”；  
> evaluator 负责判断“这种 24 步后果是不是会导向最终成功”。

Motus2 的总体逻辑也是 policy 提 candidate、world model 预测 consequence、evaluator 对 predicted outcome 做 value assessment；而且论文明确强调失败和次优 interaction 是 value learning 的重要数据。[arXiv](https://arxiv.org/abs/2608.30237?utm_source=chatgpt.com) DenseReward 同样专门在仿真里生成 collision、missed grasp、drop、recovery 等真实物理 failure，再训练 dense reward，而不是要求人工逐帧打标签。[arXiv](https://arxiv.org/abs/2607.13033?utm_source=chatgpt.com)

所以你们有 Isaac Gym，**完全没有必要手工打价值标签。**

---

### 我会怎样改你们当前 evaluator

现在 `model.py` 是：

\[
L=L_{\rm BT-preference}+L_{\rm progress}
\]

我会保留这两个东西，但加一个真正的 success/value head：

\[
\boxed{
L=
L_{\rm success}
+
L_{\rm preference}
+
L_{\rm progress}
}
\]

其中：

\[
L_{\rm success}
=
\operatorname{BCE}(\hat S,S)
\]

是主监督；

`preference` 从 `(S,P,M)` 自动构造；

`progress` 继续做过程监督。

这样也正好和 Robometer 的三种监督范式对应，但输入接口换成你们自己的

\[
H+\delta+Z,
\]

根本不需要把 Robometer 那套视频输入硬搬过来。[GitHub](https://github.com/huggingface/lerobot/blob/main/docs/source/robometer.mdx?utm_source=chatgpt.com)

---

**因此，如果你现在问我“最终选什么标签”，我的答案不是当前的 `maintained_hold > drop`。**

我会定成：

\[
\boxed{
\textbf{主标签：后续最终任务成功 }S
}
\]

\[
\boxed{
\textbf{辅助标签：物理任务进度 }P
}
\]

\[
\boxed{
\textbf{排序标签：由 }(S,P,M)\textbf{ 自动派生}
}
\]

而你们现在那套 **H-matched `maintained_hold > drop` / `lift > grasp_lost`**，留下来专门作为一个很干净的 **oracle headroom 测试集**，不要再拿它卡住训练数据。

这会把现在“216 个 episode → 1 个训练 pair”的局面，变成基本上“每条合法 Gym rollout / intervention 都能贡献 value supervision”。这才是我认为当前仓库最应该改的标签路线。