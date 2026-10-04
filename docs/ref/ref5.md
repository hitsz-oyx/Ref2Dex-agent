我看了最新 `agent/cm-interaction-oracle`，HEAD 已到 `a84436a`。从 `c0983da` 之后，真正影响研究判断的主要是三组实验。

### 1. 新 surface-token I 做了，但目前没过

现在定义了一个 hand-agnostic 的 K=1 token：

\[
I=[m_{\rm contact},d,v_n,v_t,\Delta contact]
\]

冻结已有 Cmv2 E encoder，只训练新的 I head。

结果：

| | Direct state/action | Cmv2 frozen-E |
|---|---:|---:|
| token RMSE | **0.06899** | 0.07525 |
| normalized RMSE | **0.8161** | 0.8709 |
| contact-mass RMSE | 0.08094 | **0.05281** |

所以整体还是 Cmv2 I 较差约 7%，但 **contact mass 明显好 34.8%**。

这里有两个限制要注意：

- direct baseline 大概 100k 参数，新 I head 只有约 6.5k，比较并不完全 capacity-matched；
- 更重要的是，现在其实只有 **一个 surface token**，把 256 个 object anchors 又 pool 到一起了。

所以它还没有真正实现我们说的：

\[
\{I_1,\ldots,I_8\}
\]

这种保留 spatial topology 的 surface-token field。

因此这个负结果我不会解释成“surface I 不行”，更准确是：

\[
\boxed{\text{单个 global surface token 暂时没优于 direct baseline}}
\]

---

### 2. E 对长期 G 的独立价值出来了，而且是正的

新 `Ref4 G bridge`：

\[
H\rightarrow G
\]

\[
H+E^{GT}\rightarrow G
\]

\[
H+E^{GT}+I^{GT}\rightarrow G
\]

结果：

\[
H:24.485
\]

\[
H+E:21.376
\]

所以 GT E 带来：

\[
\boxed{12.7\%}
\]

MAE 改善。

这是目前比较关键的正结果：

\[
\boxed{E\text{ 本身确实包含 H 之外的长期价值信息}}
\]

但是加 I：

\[
H+E+I:22.075
\]

反而比 E-only 差：

\[
-3.27\%
\]

因此当前证据不支持把 I 作为 G 主输入。

**不过有个非常重要的限定：这里用的 I 还是旧 Gate1 的 80D interaction，不是刚刚的新 surface-token I。**

所以它证明的是：

\[
\boxed{\text{旧 80D I 在已有 E 后没有独立价值}}
\]

并没有检验我们新设计的 surface I。

---

### 3. predicted E → G 又失败了，但这个实验不是 point-flow Cm

新的 predicted-E bridge：

\[
(H,a)\rightarrow\hat E_{1:8}
\]

结果：

\[
H:22.716
\]

\[
H+E^{GT}:21.376
\]

\[
H+\hat E:22.681
\]

GT E 提升约 5.9%，predicted E 只提升：

\[
\boxed{0.15\%}
\]

而且 E prediction 自身甚至不如 train mean baseline。

但这里千万不要得出：

> “我们的 point-flow Cm 预测 E 对 G 没用。”

因为我看了代码，这个 predictor 还是：

\[
\text{history GRU}
+
\text{raw action MLP}
\rightarrow
\text{一次性 flatten 输出 }E_{1:8}
\]

它**没有使用**最近已经验证为正向的：

\[
\text{object geometry}
+
\text{hand geometry}
+
\text{hand flow}
\rightarrow E
\]

也不是那个 K4 的 point-flow + GRU 模型。

所以这个实验基本再次证明：

\[
\boxed{\text{generic raw }(H,a)\to E_{1:8}\text{ 不好}}
\]

而不是否定 OI-Cm。

---

# 所以现在路线其实更清楚了

我认为当前最值得做的**下一步只有一个**：

把已经有效的真正 point-flow Cmv2 接到 G bridge：

\[
\boxed{
\text{geometry}+F^{hand}
\rightarrow
\hat E
\rightarrow
G
}
\]

先 K=1，或者直接用已经存在的 K4-GRU checkpoint 做一个匹配实验。

比较：

\[
H
\]

\[
H+E^{GT}
\]

\[
H+\hat E^{\text{point-flow Cm}}
\]

这才是现在真正缺的证据。

如果：

\[
H+\hat E^{Cm}<H
\]

而且能够保留一部分 GT-E 的增益，那么我们终于有完整链：

\[
\boxed{
a
\rightarrow
\text{hand flow}
\rightarrow
\hat E
\rightarrow
G
}
\]

这时才值得进入 action improvement / teacher / policy。

---

## I 现在怎么处理？

我的判断现在比上一轮更明确：

\[
\boxed{\text{暂时不要让 I 阻塞主线}}
\]

原因是 E 已经同时满足两个条件：

1. point-flow 可以预测；
2. GT E 对 G 有价值。

而 I 目前两个条件都没有同时满足。

所以现阶段：

- **E = 主路线**
- **I = auxiliary / diagnostic**
- 不继续调当前 single-token I head；
- 不做 I 的 K4；
- 不因为 I 没成熟而停止 Cm。

如果以后再回来看 I，我只做一个真正的 **8/16 object-surface tokens 的 GT value ablation**：

\[
H+E
\quad vs\quad
H+E+I^{surface}_{GT}
\]

如果连 GT surface-I 都不能增加 G 信息，就直接结束 I 主路线。

### 我认为仓库现在最大的进展

不是又多了几个 probe，而是已经基本把问题缩成：

\[
\boxed{
\text{point-flow Cm 能预测 E}
}
\]

以及

\[
\boxed{
E^{GT}\text{ 对长期 G 有价值}
}
\]

现在就差把这两块真正接起来。

**这应该是下一步，不需要再开新的 representation 分支。**