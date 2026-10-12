对，那我前面把两个算力池混在一起了。既然**视频/EgoDex 在另一台服务器跑**，那本地这台完全可以围绕机器人 WAM 主线来安排，不需要给视频留卡。

我会把今晚本地算力这样用：

\[
\boxed{\text{本地所有可用 GPU 优先跑 C：predicted-}\tau\rightarrow E\text{ 的两阶段训练}}
\]

而且我不建议多 seed 并行，还是按你之前的习惯：**一个模型、较大 batch、长时间连续训练**，避免把算力拆碎。

具体我会设成：

1. **Stage C0：短 gate，500–1000 updates**
   
   保持当前稳定 old-WAM + 已合格 Chunk-\(R\)。

   先验证：
   \[
   \hat\tau\rightarrow W_{\rm frozen}\rightarrow\hat E
   \]
   的梯度和数值合同完全正确，尤其检查：
   - World 参数梯度 = 0；
   - Chunk-\(R\) 参数梯度 = 0；
   - world scene/object context detach；
   - \(L_E\) 对 \(\hat\tau\) 和 trajectory decoder 梯度非零；
   - GT-\(\tau\) world 指标不变。

2. **C0 通过后直接进入长跑 C1**
   
   我倾向：
   \[
   \boxed{20k\text{ updates 起步}}
   \]
   如果吞吐足够、曲线到 20k 还在持续下降，就继续到：
   \[
   50k.
   \]

   这次不要再像前面 B/B2 那样只跑 200 steps，因为 C 是一个**全新的训练信号**，长训练本身有价值。

3. **用本地多卡 DDP 跑同一个模型**
   
   比如有 4 张空闲卡，就直接：
   \[
   \text{global batch}=4\times \text{per-rank batch}
   \]
   不做四个不同配方。

   我反而希望今晚把：
   \[
   \text{sample exposure}
   \]
   真正做大，而不是又跑四个小实验。

---

### 数据配比

今晚不要只喂 robot recovery 那几百条，会严重过采样。

我建议训练 batch 还是以原三域主数据为主，再混 robot：

\[
\boxed{
60\%\ \text{OakInk2/GRAB/ARCTIC}
+
25\%\ \text{nominal robot}
+
15\%\ \text{actual recovery}
}
\]

不过有个关键区别：

对于没有 robot PD 标签的 human 数据：

\[
L_a=0
\]

但仍可训练：

\[
L_\tau + L_E.
\]

对于 robot 数据：

\[
L_\tau+L_a+L_E.
\]

这样 C 的大训练不至于被我们现在很小的 robot dataset 锁死。

---

### Loss 我建议先保守

第一晚不要搞复杂权重搜索。

直接：

\[
L
=
L_\tau
+
\lambda_a L_a
+
\lambda_E L_E.
\]

初始可以用：

\[
\boxed{\lambda_a=1,\qquad \lambda_E=0.1}
\]

我不建议一开始 \(L_E=1\)。

原因是 frozen World 目前也不是完美 oracle；effect loss 太大容易让 WAM 为了迎合 World 改坏 hand trajectory。

如果 1k gate 显示：

\[
\|\nabla_{\hat\tau}L_E\|
\]

明显太小，再提高；不要先猜。

---

### 今晚最重要的 checkpoint 不是 final loss

每 1k 固定评估同一组 panel：

\[
\begin{aligned}
&E_\tau:\quad h1,\ first8,\ all24\\
&E_a:\quad first\text{-}PD\ XYZ/rotation/finger\\
&E_E:\quad \hat\tau\rightarrow W\rightarrow E\\
&E_{oracle}:\quad \tau^{GT}\rightarrow W\rightarrow E
\end{aligned}
\]

尤其看这个 gap：

\[
\boxed{
\Delta_E
=
E(W(H,\hat\tau),E^{GT})
-
E(W(H,\tau^{GT}),E^{GT})
}
\]

它非常关键。

因为它直接表示：

> **World 本身的不可约误差之外，WAM 预测 \(\tau\) 又额外造成了多少 effect error。**

如果今晚能把：

\[
\Delta_E
\]

明显压下去，同时 first-PD 不退化，那这次训练就是有意义的。

---

## 我现在不建议本地今晚跑的东西

先不要：

- 再调 B2 的 \(\alpha\)；
- 再扩大 PTv3 hand readout；
- 再做 recovery DAgger；
- 再跑旧 10k WAM；
- 加 RL；
- 加 3PoinTr；
- 多 seed。

这些都不如 C 值得烧一晚上。

所以修正后的安排非常简单：

\[
\boxed{
\textbf{视频服务器继续视频；本地服务器整晚专门跑 C。}
}
\]

如果本地目前是 4/8 张可用卡，我会直接把 C 做成**单模型多卡长训练**，而不是再拆成若干小 Probe。