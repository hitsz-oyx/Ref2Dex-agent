我重新看了当前仓库。现在 `main` 最新整体 HEAD 已到 `14f87e2`，但这个最新提交主要是视频/整段几何那条并行线；**机器人 WAM 主线最新真正改变判断的是 `10a5315` / `4cf822a` 的 WAM+\(R\) 联合训练结果**。

当前结论已经比前一轮更清楚了：

\[
\boxed{\text{现在不应该继续扩 WAM 容量，也不应该马上上 RL。}}
\]

真正需要先解决的是 **WAM→\(R\) 的执行接口，以及 student visited-state coverage**。

### 先看现在已经证明了什么

geometry-only 路线基本已经到头了。startup anchor 之后，frame0 和 t8 已经恢复得不错，但 fresh 128-step rollout 依然是 **pure actor 0 held**，并且 t9–12 很快开始漂。也就是说：

\[
\text{起步问题}\quad \checkmark
\]

但：

\[
\text{closed-loop distribution shift}\quad \times
\]

这不是再多训练 500 次 geometry loss 能解决的问题。

更关键的是最新的 WAM+\(R\) joint fit。它确实证明了 **control-aware joint training 是能学的**：

\[
\text{startup PD p95}:20.5\rightarrow9.9\text{ mm}
\]

\[
\text{nominal near8}:17.2\rightarrow14.8\text{ mm}
\]

pure early train 也从约 \(588\) mm 降到了 \(61\) mm。

但它同时暴露出一个非常严重的问题：

\[
R(\tau_{\rm GT})
\]

的物理语义被 joint training 破坏了。

例如 nominal：

\[
10.59\rightarrow43.96\text{ mm}
\]

recovery：

\[
20.44\rightarrow62.76\text{ mm}.
\]

所以现在发生的是某种 **WAM 和 \(R\) 共适应**：

\[
\hat\tau
\rightarrow R
\rightarrow u
\]

两边一起调整以后，可以把训练动作拟合得更好，但 \(\tau\) 不再保持原本的“真实手轨迹 → 控制”的稳定意义。

更明显的是：

\[
\text{late pure diagnostic}\approx955\text{ mm}
\]

基本完全没改善。

所以现在不能说“联合训练成功了”，也不能说“联合训练失败了”。准确说是：

\[
\boxed{
\text{joint optimization 可行，但当前 formulation 有明显的 representation drift。}
}
\]

---

## 我认为下一步应该按这个顺序

1. **先暂停 WAM 结构修改，单独把 \(R\) 查清楚。** 当前 `PDRetargeter` 的输入是

\[
87D\ state + 24\times33D\ \tau
\]

然后直接 flatten：

\[
[87+792]
\rightarrow \text{MLP}
\rightarrow12D\ PD_t.
\]

这实际上很值得怀疑：我们用**未来完整 24 步手轨迹来决定当前一步控制**。

而我们已经知道 WAM 的误差是：

\[
1\!:\!8\ll9\!:\!16\ll17\!:\!24.
\]

所以我现在最建议先做一个极便宜、信息量很大的实验：

\[
R_K(s,\tau_{1:K})\rightarrow u_t,
\qquad
K=1,4,8,16,24.
\]

先完全不动 WAM。

用 **GT \(\tau\)**，在同样的 nominal / t8 / t16 / t32 recovery 数据上比较：

\[
PD_{XYZ},\ rotation,\ finger,\ p95.
\]

现有 takeover 数据都已经被多次观察过，所以这里只能叫 development diagnostic，不能叫独立泛化验证；但用来选执行接口完全足够。

如果出现：

\[
R_8 \gg R_{24},
\]

那答案就很明确：

\[
\boxed{
24\text{步用于 world reasoning，}
4\sim8\text{步用于即时 execution。}
}
\]

我现在认为这个概率不低。

2. **然后再做 joint training，但必须把 WAM 和 \(R\) 的梯度职责拆开。** 仓库现在建议的“增加 true-\(\tau\)\(\rightarrow R\) supervision”方向是对的，但我建议比现在再严格一点。

不要继续：

\[
L_{\rm action}
\big(R_\phi(s,\hat\tau_\theta),u^*\big)
\]

同时随便更新 \(\theta,\phi\)。

改成两条明确的监督：

\[
\boxed{
L_R=
D(R_\phi(s,\tau^*),u^*)
}
\]

只用**真实 expert trajectory**训练 \(R\)。

而 WAM：

\[
\boxed{
L_{\rm WAM}
=
L_\tau(\hat\tau,\tau^*)
+
\lambda
D(R_{\rm stopgrad\,param}(s,\hat\tau),u^*)
}
\]

这里的含义很重要：

- 梯度可以穿过 \(R\) 回到 \(\hat\tau\)；
- **但是 action loss 不更新 \(R\)**；
- \(R\) 自己只通过 GT \(\tau^*\rightarrow PD^*\) 更新。

这样就不会让：

\[
\hat\tau
\quad\text{和}\quad
R
\]

私下重新约定一套“假的 latent code”。

对于那些 pure-WAM visited states，当前只有：

\[
s_t^{WAM},u_t^*
\]

而没有可信的 \(\tau^*\)。

这些数据应该：

\[
\boxed{\text{只通过 frozen/anchored }R\text{ 更新 WAM，不更新 }R。}
\]

我认为这是当前 joint formulation 最关键的修正。

3. **这个版本通过 offline retention 后，只跑一次 fresh actor rollout。** 不需要马上做完整 542 step。

我们真正要观察的是：

\[
t=0,4,8,12,16,24,32
\]

的：

\[
wrist\ error,\quad PD\ error,\quad clipping,\quad held.
\]

尤其要看以前的故障点：

\[
t\approx9\sim16.
\]

如果以前：

\[
t12:\ PD\approx100\text{ mm}
\]

而新模型能明显延后/降低这个漂移，那才说明这条路值得继续。

如果依旧：

\[
0\ held
\]

那么就不要再优化 loss 了，直接进入真正的数据问题。

4. **届时开始多 takeover 的 DAgger 式 aggregation。** 当前最大的证据之一就是 late pure state 仍然严重 OOD。

所以需要：

```text
new WAM rollout
      ↓
student visited state
      ↓
expert takeover at t=8 / 16 / 32
      ↓
true expert future τ*
+
expert PD*
      ↓
aggregate
```

关键还是：

> expert future 必须从 student 真正走到的状态开始。

不能拿 nominal teacher future 配过去。

这样逐渐把：

\[
p_{\rm train}(s)
\]

推向：

\[
p_{\rm WAM}(s).
\]

这才是在真正解决现在的 covariate shift。

---

### 有一件事我现在反而不建议急着做

就是马上换 Flow Matching、Diffusion、更大的 Transformer、继续解冻 PTv3。

因为现在已有实验已经说明：

\[
\text{geometry 能学}
\]

甚至：

\[
\text{recovery geometry train EPE 能降得非常漂亮}.
\]

真正失败的是：

\[
\text{geometry}
\rightarrow
\text{control}
\rightarrow
\text{new state}
\rightarrow
\text{next prediction}
\]

这条闭环。

所以再换一个 trajectory generator，大概率只是把问题藏起来。

---

### Video / World / RL 怎么处理

视频那条线可以并行。最新 `14f87e2` 已经在做 whole-record geometry / short-window restart，之前跨 clip 的 scale inconsistency 也已经明显改善。它现在应该继续做：

\[
\text{one recording}
\rightarrow
\text{one consistent 3D geometry}
\]

然后再扩大视频训练，不要影响机器人主线。

PointWorld / world branch 先冻结。RL 更应该继续往后放。

我会把当前路线冻结成：

\[
\boxed{
\begin{array}{c}
\text{① R horizon audit}\\
\downarrow\\
\text{② GT-}\tau\text{-anchored R + control-aware WAM}\\
\downarrow\\
\text{③ fresh closed-loop}\\
\downarrow\\
\text{④ multi-takeover student-state aggregation}\\
\downarrow\\
\text{⑤ world consistency / evaluator}\\
\downarrow\\
\text{⑥ RL}
\end{array}}
\]

其中**现在最该立即做的是第①步，而不是直接再开一个 1000-update joint run**。

因为仓库现在已经给了我们一个非常具体的嫌疑：**\(R\) 把完整 24 步未来 flatten 后去预测当前 PD，这个执行合同本身可能就不合理。** 先把这个变量排除，后面所有 WAM 微调才有意义。