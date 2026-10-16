可以，而且这个“平均化”是能比较系统地量出来的，不应该只靠看轨迹觉得“变平了”。

对我们现在的 WAM，最核心要检测的是：

\[
\boxed{p(\tau_{future}\mid H,G)\text{ 是否本来是多模态的，而 deterministic MSE 模型却落在几个 mode 中间}}
\]

我建议分三层统计。

### 1. 最直接：看预测是否更接近“邻域均值”而不是真实 mode

对每个验证样本 \(i\)，不要只看它自己的 GT。先在 TRAIN 中找与它条件相近的 \(K\) 个样本：

\[
\mathcal N_i=
\mathrm{KNN}(H_i,G_i)
\]

最好用当前 WAM 的 frozen \(z\) 做 retrieval，而不是直接在几千维原输入上算距离。

这些邻居有未来手轨迹：

\[
\tau_i^{(1)},...,\tau_i^{(K)}
\]

求经验条件均值：

\[
\bar\tau_i=
\frac1K\sum_j\tau_i^{(j)}
\]

然后比较 WAM 预测 \(\hat\tau_i\) 到：

\[
D_{\rm mean}
=
d(\hat\tau_i,\bar\tau_i)
\]

以及到最近真实 future：

\[
D_{\rm real}
=
\min_jd(\hat\tau_i,\tau_i^{(j)})
\]

如果经常出现：

\[
\boxed{
D_{\rm mean}\ll D_{\rm real}
}
\]

说明模型非常像是在预测条件均值，而不是落在真实轨迹流形上。

但这还不够，因为如果数据本来就是单峰，均值本身就是合理轨迹。

---

## 2. 所以一定要先找“真正多模态”的条件

把这些邻居的未来轨迹聚类，比如：

\[
\tau^{(1:K)}
\rightarrow
K\text{-means / spectral clustering}
\]

不要直接 flatten 全24步；可以先提几个比较有物理意义的 descriptor：

\[
[
\Delta p_{\rm wrist}^{8},
\Delta p_{\rm wrist}^{24},
R_{\rm wrist}^{24},
\text{finger endpoints},
\text{approach direction}
]
\]

例如一个物体存在：

- 左侧 approach；
- 右侧 approach；

那么就应该明显出现两个 cluster。

定义：

\[
C_1,C_2,\ldots,C_M
\]

对应 prototype：

\[
\mu_1,\mu_2,\ldots,\mu_M.
\]

然后算：

\[
D_{\rm mode}
=
\min_m d(\hat\tau,\mu_m)
\]

再算预测到 conditional mean：

\[
D_{\rm avg}
=
d(\hat\tau,\bar\tau).
\]

如果在**明显多模态样本**里：

\[
D_{\rm avg}<D_{\rm mode}
\]

大量发生，那就是非常直接的 averaging evidence。

甚至可以定义一个简单分数：

\[
\boxed{
S_{\rm avg}
=
\frac{D_{\rm mode}}
{D_{\rm avg}+D_{\rm mode}}
}
\]

越接近 1，说明越倾向于 conditional mean 而不是某个 mode。

不过论文里我更建议直接报告两个距离，不一定需要自己发明一个 fancy metric。

---

# 3. 对机器人来说，还有几个非常直观的“平均化症状”

这个反而可能比上面的统计更有说服力。

### 运动幅度衰减

MSE 平均不同方向的运动后，经常出现：

\[
\|\Delta \hat p\|
<
\|\Delta p^{GT}\|.
\]

统计：

\[
r_{\rm motion}
=
\frac{\|\hat\tau_{24}-\tau_0\|}
{\|\tau^{GT}_{24}-\tau_0\|}.
\]

或者 first8：

\[
r_8
=
\frac{\|\Delta \hat p_{1:8}\|}
{\|\Delta p^{GT}_{1:8}\|}.
\]

如果高多模态区域系统性：

\[
r<1,
\]

尤其显著小于单模态区域，这很像 average collapse。

---

### 速度被抹平

算：

\[
v_t=\tau_{t+1}-\tau_t.
\]

比较：

\[
E\|\hat v_t\|
\]

和：

\[
E\|v_t^{GT}\|.
\]

如果真实未来存在“左/右快速移动”，均值很容易变成“中间慢慢走”。

---

### 曲率 / 转腕幅度衰减

灵巧操作很重要。

统计：

\[
\sum_t
\|\hat v_{t+1}-\hat v_t\|
\]

和 GT。

平均化通常会让轨迹变得：

\[
\boxed{\text{更直、更慢、更平滑}}
\]

但不一定物理正确。

---

## 4. 我觉得最适合我们的：预测是不是落在“低密度区域”

这是判断 multimodal averaging 最有力的统计之一。

假设真实未来分成：

```text
      mode A
       ●●●
      ●●●

            × predicted mean

                     ●●●
                    ●●●
                   mode B
```

虽然 × 对两个 mode 的平均 L2 都不算特别大，但：

\[
\boxed{\text{数据里实际上根本没有这种 trajectory}}
\]

所以可以计算 prediction 到 TRAIN future manifold 的 kNN 距离：

\[
D_{\rm manifold}
=
\frac1k
\sum_{j\in kNN(\hat\tau)}
d(\hat\tau,\tau_j).
\]

然后拿 GT trajectory 也算：

\[
D_{\rm GT-manifold}.
\]

如果：

\[
\boxed{
D_{\rm pred-manifold}
\gg
D_{\rm GT-manifold}
}
\]

尤其只发生在高 conditional-diversity 样本上，那就是非常强的 averaging 证据。

这个我很推荐。

---

# 5. 对我们的抓取最重要的是 contact averaging

这个可以单独统计。

例如两个有效 future：

- mode A：食指从左侧接触；
- mode B：食指从右侧接触。

MSE 平均后可能变成：

> 指尖停在物体外面。

所以对每个 fingertip、每个 horizon 算：

\[
d_{surf}(p_{tip})
\]

即到物体表面的 signed/unsigned distance。

然后比较 GT 与 prediction：

\[
P(d^{GT}_{surf}<\epsilon)
\]

vs

\[
P(d^{pred}_{surf}<\epsilon).
\]

尤其只在：

\[
\text{GT至少有一种mode进入contact}
\]

的 ambiguous samples 上看。

如果：

\[
\boxed{
\text{GT modes有contact，但mean prediction经常悬在表面之外}
}
\]

这是对我们特别有意义的“平均化”。

比单纯 ADE 更能说明为什么：

\[
\text{EPE不算太大，但抓不起来}.
\]

---

# 我建议我们真正做一个“multimodality audit”

流程可以非常简单：

1. 从 DEV 抽 1–2k windows。
2. 用 frozen WAM 的 pre-head shared \(z\) 在 TRAIN 中检索每个样本的 32 或 64 个近邻。
3. 只保留 \(H,G\) 足够相似的邻居。
4. 对未来 hand trajectory 做 PCA/descriptor 后聚类。
5. 按 conditional future spread 分：
   \[
   \text{low / medium / high ambiguity}
   \]
6. 每层统计：
   - hand ADE / first8；
   - prediction→conditional mean；
   - prediction→nearest mode；
   - prediction→future manifold；
   - motion magnitude ratio；
   - fingertip surface/contact error。

这里最重要的横轴就是：

\[
\boxed{
\sigma_{\rm future}
=
E_j\|\tau_j-\bar\tau\|
}
\]

也就是“相似当前条件下，未来本身有多分散”。

然后画：

\[
\sigma_{\rm future}
\quad\text{vs}\quad
D_{\rm manifold}
\]

如果明显正相关，而且 high-diversity 区域 prediction 越来越靠近 \(\bar\tau\)，就基本坐实了。

---

## 然后 Flow Matching 怎么验证？

这时候才值得上 FM。

同一个输入：

\[
(H,G)
\]

Flow Matching 输出 \(M\) 个 future：

\[
\hat\tau^{(1)},...,\hat\tau^{(M)}.
\]

不要再只看 mean ADE。

应该看：

\[
\boxed{\min ADE@M}
\]

\[
\boxed{\min FDE@M}
\]

以及：

\[
\boxed{\text{mode coverage}}
\]

例如：

\[
\min_{m=1:M}
d(\hat\tau^{(m)},\tau^{GT})
\]

如果 deterministic WAM：

\[
ADE=25mm
\]

而 FM：

\[
\min ADE@8=10mm,
\]

同时 samples 分别覆盖真实的左抓/右抓 mode，而不是靠噪声乱散，那 Flow Matching 才真正解决了问题。

还可以加：

\[
\text{diversity}
=
\frac{2}{M(M-1)}
\sum_{i<j}d(\hat\tau_i,\hat\tau_j)
\]

但 diversity 不能越大越好，要同时看：

\[
\boxed{\text{precision + coverage}}
\]

否则模型只是乱采样。

---

### 我最建议先算的三个指标

如果不想一下搞复杂，我认为先做这三个就够：

\[
\boxed{1.\ \text{conditional future variance}}
\]

\[
\boxed{2.\ D(\hat\tau,\text{conditional mean})\ vs\ D(\hat\tau,\text{nearest real mode})}
\]

\[
\boxed{3.\ prediction-to-real-future-manifold kNN distance}
\]

如果这三个都显示：

> 条件越多模态，deterministic WAM 越趋向真实数据不存在的中间轨迹，

那我们就有很充分的理由从现在的 deterministic MSE head 升级成：

\[
\boxed{\text{conditional Flow Matching future head}}
\]

而不是因为“Flow Matching 比较新”就换架构。