可以。现在最好把整条链固定下来，不再反复改 chunk 定义。**第一版只保留两个时间尺度：**

\[
\boxed{K=24\text{：规划/评价窗口}}
\]

\[
\boxed{K_{\rm exec}=8\text{：实际执行后重新规划}}
\]

24 步就是一个完整 action chunk，也是 PointWorld 要解释的未来窗口；8 步直接沿用我们 Ref14 已经验证过的 rolling cadence。除此之外不再引入 4/16/32 等额外 chunk。\(H\) 直接用现有策略的当前观测/历史合同，也不重新定义一个 history length。

完整链路我建议冻结成下面这样。

1. **先采连续 rollout 数据，不 fork。** 六专家正常控制，每条 episode 选一个合适阶段，在接下来 24 步 action 上加入平滑随机扰动，然后重新让六专家继续运行到 episode 结束。不是每一步 iid 大噪声，而是产生一个合理的 24-step residual chunk。数据要覆盖 clean rollout 和扰动后的 miss、slip、drop、recovery 等。不要只在 reset 后马上扰动；参考 DenseReward 的做法，按 approach/contact/grasp/lift/hold 等阶段分层采扰动，否则大量数据只是“手一开始就没碰到物体”的简单失败。DenseReward 本身就是通过 phase-aware targeted perturbation 在仿真中制造 missed grasp、drop、collision、recovery 等失败。[DenseReward](https://dense-reward.github.io/?utm_source=chatgpt.com)

2. **从每条真实 rollout 切出 evaluator 样本。** 对任意合法时刻 \(t\)，保存
   \[
   H_t,
   \quad
   A^{GT}_{t:t+23},
   \quad
   Z^{GT}_{t:t+23}.
   \]
   这里 \(A^{GT}\) 是这24步**真的执行过的动作**，也可以转成我们统一的 action point-flow \(F^{action}_{1:24}\)；\(Z^{GT}\) 是执行这些动作后真实发生的24步未来物理过程。关键是：**\(Z^{GT}\) 的字段必须和以后 PointWorld 能预测的字段完全一致**。比如 PointWorld 最终输出24步物体运动/点流表示，就让 oracle evaluator 也只读这些 GT 物理量，不能偷偷加 future success、reward、drop flag 等 PointWorld 永远预测不了的字段。

3. **标签按 Robometer 思路生成，不再构造旧 \(Y\)。** 成功/可靠轨迹可以提供逐帧 progress 和 success 作为绝对尺度锚点；失败和次优轨迹的精确 progress 本来就含糊，因此主要作为 preference：
   \[
   \tau_A\succ\tau_B.
   \]
   Robometer 的正式做法就是“expert progress + trajectory preference”；它的失败/次优轨迹很多情况下直接 mask 掉 dense progress loss，而用同任务下 success/suboptimal/failure 的相对关系训练 preference。[Robometer](https://robometer.github.io/assets/robometer.pdf?utm_source=chatgpt.com)
   所以对于24步窗口，Evaluator 可以输出一条
   \[
   \hat p_{1:24}
   \]
   的 progress trajectory，同时输出一个整个 branch 的 preference/value score \(s\)。**真正用于动作排序的主指标看 preference/ranking，不再看旧 \(Y\) 的 MSE。**

4. **先做最关键的两个 matched evaluator。**
   \[
   E_0:\quad(H,A)\rightarrow V
   \]
   和
   \[
   E_{\rm oracle}:\quad(H,A,Z^{GT})\rightarrow V.
   \]
   两者完全相同的数据、label、split、容量，只差有没有真实未来。这个实验回答我们最核心的问题：
   \[
   \boxed{\text{未来物理后果本身有没有额外动作价值信息？}}
   \]
   如果
   \[
   E_{\rm oracle}\approx E_0,
   \]
   那么 PointWorld 对 action ranking 没有必要，我们应该接受这个结果。反过来如果
   \[
   E_{\rm oracle}\gg E_0,
   \]
   就证明了世界模型这条链存在明确 oracle headroom。这里虽然 label 和 \(Z^{GT}\) 都来自真实 rollout，但不构成泄漏，因为一个是**原始未来物理过程**，一个是**这段过程的任务质量监督**；只要 \(Z^{GT}\) 不含 label 本身即可。Motus2 训练 evaluator 的逻辑也是让 value query 读取 action 和 future，再判断这个预测后果的任务价值。[Motus Robotics](https://motus-robotics.github.io/motus2/?utm_source=chatgpt.com)

5. **然后才把 PointWorld 接进来。** OakInk2 预训练完成后，用这批机器人 rollout 做域内适配：
   \[
   (H,A)\xrightarrow{\text{PointWorld}}\hat Z_{1:24}.
   \]
   然后得到第三个对照：
   \[
   E_{\rm WM}:\quad(H,A,\hat Z)\rightarrow V.
   \]
   最终就形成非常干净的三级链：
   \[
   \boxed{
   E_0(H,A)
   \rightarrow
   E_{\rm WM}(H,A,\hat Z)
   \rightarrow
   E_{\rm oracle}(H,A,Z^{GT})
   }.
   \]
   我们关心的不是 PointWorld 的 EPE 单独有多漂亮，而是 \(E_{\rm WM}\) 能恢复多少 \(E_{\rm oracle}-E_0\) 的排序增益。这和 Motus2 的“policy proposes action chunks → simulator predicts visual consequences → evaluator ranks predicted branches”是同一个逻辑。[Motus Robotics](https://motus-robotics.github.io/motus2/?utm_source=chatgpt.com)

6. **最后才进入真正在线规划。** 这里还有一个必须正视的接口：现在六专家是 reactive policy，并不会在一个 \(H_t\) 下一次性输出24步 future action chunk；训练数据里的 \(A^{GT}_{1:24}\) 是 rollout 完成后才知道的。因此如果前面 oracle 成立，我们还需要一个 Motus-style **24步 chunk proposal policy**：
   \[
   H_t\rightarrow
   A_1^{1:24},\ldots,A_N^{1:24}.
   \]
   这可以从六专家 rollout 数据蒸馏出来。然后在线时：
   \[
   A_i
   \xrightarrow{\text{PointWorld}}
   \hat Z_i
   \xrightarrow{\text{Evaluator}}
   s_i,
   \]
   选择
   \[
   i^*=\arg\max_i s_i,
   \]
   但只执行前 **8步**，拿真实新状态重新规划。Motus2 本身也是 policy 提 executable action chunks、simulator 想象未来、evaluator 排序，执行后从下一真实观测重新规划。[Motus Robotics](https://motus-robotics.github.io/motus2/?utm_source=chatgpt.com)

这里有一个特别重要的顺序关系：

\[
\boxed{
\text{现在}
=
\text{采 rollout}
\rightarrow
\text{训练 }E_0/E_{\rm oracle}
}
\]

暂时**不需要**解决在线24步候选生成，也不需要等 PointWorld。

只要先回答：

\[
\boxed{
(H,A,Z^{GT})
\text{ 是否真的比 }
(H,A)
\text{ 更会判断 rollout 好坏？}
}
\]

如果答案是否定的，我们可以直接停止这套世界模型→Evaluator 接法，省掉后面大量工程。如果答案明显为正，再把正在训练的 PointWorld 接进来。

所以整条主线可以压成：

\[
\boxed{
\begin{aligned}
&\text{六专家+扰动连续rollout}\\
&\Downarrow\\
&(H,A,Z^{GT})+\text{Robometer labels}\\
&\Downarrow\\
&E_0(H,A)\;\text{vs.}\;E_{\rm oracle}(H,A,Z^{GT})\\
&\Downarrow\\
&\text{PointWorld}:(H,A)\rightarrow\hat Z\\
&\Downarrow\\
&E_{\rm WM}(H,A,\hat Z)\\
&\Downarrow\\
&\text{24步候选}\rightarrow\text{评分}\rightarrow\text{执行8步}\rightarrow\text{重新规划}
\end{aligned}
}
\]

我认为现在就应该按这个版本冻结，不再改 \(Y\)、horizon 和 chunk 语义。
