我看完了。**我认为现在最可疑的不是 ACT 模型本身学不会，而是我们 `receding8` 的部署方式和 ACT 的标准执行方式不一致。先修这个，比重新训练更合理。**

仓库里的事实很关键：

\[
\text{open\_loop24}: 0.7939m,\ held478
\]

而 reactive teacher 是：

\[
0.8134m,\ held483
\]

说明 **ACT 输出的动作语义、时间对齐、native action 接口基本是对的**。而同一个 checkpoint 换成：

\[
\text{receding8}
\]

就直接 **held=0、完全抓不起来**。甚至把 21 条数据全部拿来拟合，离线 MSE 降到 \(4.53\times10^{-5}\)，receding8 仍然失败。因此“训练误差不够低”不是主要解释。

### 我发现最值得怀疑的地方

当前代码的 receding8 实际是：

\[
H_t \rightarrow [a_t,\ldots,a_{t+23}]
\]

执行前 8 步，然后在 \(t+8\)：

\[
H_{t+8}\rightarrow[a'_{t+8},\ldots,a'_{t+31}]
\]

**直接丢掉旧 chunk 剩余的 16 步，硬切换到新 chunk 的第 0 步。**

代码就是每 8 步重新生成，然后：

```python
chunk_offset = tick % proposal_period
```

所以 receding8 永远只消费每个预测 chunk 的 **0～7 位**。

这其实不是标准 ACT 的 temporal aggregation。官方 ACT 开启 temporal aggregation 后，会**每一步重新预测一个 chunk，然后把所有覆盖当前时刻的历史 chunk 预测按权重融合**，而不是硬切换。官方代码明确把所有 overlapping chunk 存下来，对当前 action 做指数加权平均。[GitHub](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)

所以我们现在所谓的 “receding8 ACT” 更准确地说是：

\[
\boxed{\text{hard-switch chunk MPC}}
\]

而不是真正的 ACT temporal ensemble。

---

更重要的是，**open_loop24 已经证明 ACT 在自己造成的状态上并非完全不能工作**。它整个 542 步会不断每 24 步重新生成 chunk，却还能 held478。

所以仓库之前写：

> “replanning every eight steps moves proposal out of distribution”

这个解释我认为**不完整**。

更准确的怀疑应该是：

\[
\boxed{\text{8步频繁硬切 chunk 导致动作边界不连续}}
\]

尤其抓取接触阶段对很小的 wrist/finger control jump 都很敏感。

---

### 我建议先做三个非常便宜的检查，不重新训练

1. **chunk overlap consistency**  
   对现有数据直接算：

\[
\hat A_t[8:24]
\quad vs\quad
\hat A_{t+8}[0:16]
\]

特别看：

\[
\|\hat A_t[8]-\hat A_{t+8}[0]\|.
\]

如果这里跳得明显，就基本找到原因了。

2. **按 horizon 统计误差**  
   现在只报整体 MSE 和 first-action MAE。应该画：

\[
e_k =
\|\hat a_{t+k}-a_{t+k}\|,
\quad k=0,\ldots,23.
\]

因为 receding8 **只执行 k=0..7**。如果恰好前 8 个 token 误差或 bias 更大，整体 MSE 会把问题掩盖掉。

3. **先改 inference，不改训练**  
   用现有 checkpoint 实现 overlap blending：

在 \(t=8\) 时不要直接：

\[
a=\hat A_{t+8}[0]
\]

而是融合：

\[
a=
w_1\hat A_t[8]
+
w_2\hat A_{t+8}[0].
\]

到了 \(t=16\)，可以融合三个 chunk：

\[
\hat A_t[16],
\hat A_{t+8}[8],
\hat A_{t+16}[0].
\]

最好直接做官方 ACT 那种 temporal ensemble。官方实现甚至在开启 aggregation 时把 query frequency 设为 1，每一步产生 chunk 并融合历史预测。[GitHub](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)

---

如果这样一改，抓取恢复了，那就说明我们根本不用先重训 ACT。

如果 temporal ensemble 后仍然抓不起来，第二优先级才是**训练分布**。这方面仓库已经有正信号：deployment-conditioned 数据上，新 fit 的 MSE 大约：

\[
1.83\sim1.90\times10^{-4}
\]

而原 clean checkpoint 在相同 deployment history 上是：

\[
1.93\sim2.02\times10^{-3}.
\]

差了接近一个数量级。

说明 clean 21 episode 确实和 native deployment state 有分布差异。但我不会先拿这个当主因，因为 **open_loop24 已经能完成任务**。

---

### 还有一点：我们这个其实只是 “ACT-like”，不是完整 ACT

现在 `NativeActionChunkProposal` 是一个：

\[
H_t\rightarrow24\times18
\]

的 deterministic Transformer chunk regressor。

它没有：

- ACT 的 CVAE latent；
- temporal ensemble；
- overlapping prediction aggregation。

对我们这种几乎确定性的 baseline，CVAE 未必重要；**temporal ensemble 很重要**。

所以我现在的判断排序是：

\[
\boxed{
\text{① receding8 hard switch}
>
\text{② deployment distribution gap}
>
\text{③ 模型容量/训练不足}
}
\]

而且 `H_t → action[t:t+24]` 的 off-by-one 我检查了，是对齐的；target 也是 native `pre_physics_step` 的真实 18D control，RMS/clip 后来也修正了。**暂时没看到“action 标签取错一帧”这种明显代码 bug。**

因此我建议现在让代理先**不要重训**，直接做 `chunk overlap / horizon-error audit + temporal ensemble behavior screen`。这是目前最小、信息量最高的一步。