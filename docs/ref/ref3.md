我建议先做一个**很小的 action-chunk Gate 2**，不再扩实验树。

只固定一个长度：

\[
\boxed{K=8}
\]

大约四分之一秒，既比单步有实际物理意义，又不会像 H32 那么难。

1. **数据直接用现有离线轨迹**  
   每条样本变成：
   \[
   (H_t,\ A_{t:t+7},\ C^{GT}_{t+1:t+8},\ G_t)
   \]
   其中 \(C=(E,I)\)。不重新跑仿真。

2. **只训练两个 consequence predictor**
   \[
   Cm_1(H,a_t)\rightarrow \hat C_{1:8}
   \]
   和
   \[
   Cm_8(H,A_{t:t+7})\rightarrow \hat C_{1:8}
   \]
   网络尽量相同，只改变 action 输入。这样直接回答：
   > 给完整 action chunk，是否比只给第一步动作更容易预测真实 interaction consequence？

3. **训练一个 GT bridge，然后冻结**
   \[
   g(H,C^{GT}_{1:8})\rightarrow G
   \]
   测试时分别代入：
   \[
   g(H,\hat C^{Cm_1})
   \]
   和
   \[
   g(H,\hat C^{Cm_8})
   \]

4. **只看三个结果**
   \[
   V_H,\qquad g(H,\hat C^{Cm_1}),\qquad g(H,\hat C^{Cm_8}),\qquad g(H,C^{GT})
   \]
   主要看：
   \[
   Cm_8
   \]
   是否明显比 \(Cm_1\) 保留更多 GT consequence 带来的 \(G\) 增益。

判断也很简单：

如果：

\[
Cm_8 \gg Cm_1
\]

并且明显优于 \(V_H\)，就说明**action chunk 是正确方向**，后面正式把 Cm 定义成：

\[
\boxed{(H,A_{\text{chunk}})\rightarrow(E,I)}
\]

如果 \(Cm_8\) 仍然几乎预测不了 consequence，或者接入 \(g\) 后不比 \(V_H\) 好，就先停，不继续做 PPO、蒸馏或更大的 world model。

**这轮先不要扫 K=3/5/8/10，也不要改复杂 decoder。只测 K=8，确认“chunk 是否解决了单动作无法解释多步未来”这个核心问题。**