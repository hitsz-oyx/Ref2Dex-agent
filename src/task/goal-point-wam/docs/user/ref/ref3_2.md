因为 PointWAM 真正成功的地方，和我们现在这条链其实有一个很大的结构差别：

\[
\boxed{\text{PointWAM 并不是靠“粗 object goal → 手轨迹 → 固定 R”直接抓起来的。}}
\]

它更像：

\[
\boxed{\text{人类视频预训练几何先验}
\rightarrow
\text{机器人示范上的端到端 BC 微调}}
\]

这正好解释了为什么它没有我们现在这么严重的 actor–R 语义问题。

### 1. PointWAM 的机器人阶段是 actor 和 retargeter 一起训的

PointWAM 定义：

\[
(\hat U_t,\hat V_t)
=
f(P_t,Q_t,\ell)
\]

其中 \(\hat V_t\) 是预测的 hand trajectory。

然后：

\[
\hat A_t
=
g(\hat V_t,s_t)
\]

其中 \(s_t\) 明确包含机器人当前末端位姿和 finger joint angles。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

关键是机器人阶段的 loss：

\[
L_{\rm robot}
=
L_{\rm traj}
+
\lambda
\|\hat A_t-A_t^*\|_1,
\qquad \lambda=1.
\]

而论文明确写的是：**机器人微调时继续 fine-tune trajectory forecaster，同时从头训练 retargeter**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

所以 PointWAM 训练的是一个整体：

\[
\boxed{
f_{\theta}(s)
\rightarrow
\hat\tau
\rightarrow
g_{\phi}(\hat\tau,s)
\rightarrow
a
}
\]

而不是我们现在这种反复：

```text
先固定 R
↓
训 actor 去适配 R
↓
再单独修 R
↓
actor 又不适配新 R
```

我们刚刚的实验已经直接证明了这个问题：

\[
R_{\rm repaired}(\tau^{GT})
\]

非常好，

但：

\[
R_{\rm repaired}(\hat\tau_{\rm actor})
\]

反而可能更差。

PointWAM 从训练机制上就不容易出现这么严重的 pair drift。

---

## 2. 更关键：它的 \(\tau\) 一直被“真实物理手轨迹”锚定

PointWAM 的机器人 trajectory GT 不是人手轨迹硬搬过来。

机器人 demonstration 中，它直接从**机器人自身状态通过 FK**得到 hand keypoint trajectories。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

因此同一个 robot demo 同时提供：

\[
\tau^*_{\rm robot}
\]

和：

\[
A^*_{\rm robot}.
\]

于是：

\[
\boxed{
\tau^*
\leftrightarrow
A^*
}
\]

来自同一个 embodiment、同一个状态、同一次执行。

这非常重要。

PointWAM 同时要求：

\[
\hat\tau\approx\tau^*
\]

以及：

\[
g(\hat\tau,s)\approx A^*.
\]

所以 \(\hat\tau\) 很难变成我们现在这种“为了补偿 R 而产生的奇怪 latent trajectory”。

因为一旦它偏离真实 robot hand motion：

\[
L_{\rm traj}
\]

会直接拉回来。

---

# 3. PointWAM 没有让 frozen World 来“指导” \(\tau\)

这个差别我现在觉得尤其重要。

PointWAM 的 scene world modeling 是：

\[
\hat U_t\approx U_t^*
\]

也就是**直接监督 scene trajectory**。

它和 hand trajectory 共用 backbone，因此 scene supervision 帮助模型理解：

\[
\text{hand motion}
\leftrightarrow
\text{object/scene motion}.
\]

但论文明确说：

> scene forecast 不进入 action retargeter。

也就是说实际 action 是：

\[
\hat V_t+s_t
\rightarrow
A_t,
\]

不是：

\[
W(H,\hat\tau)
\rightarrow E
\rightarrow
\text{反过来优化}\hat\tau.
\] :chatgpt-content-reference{index="3"}


这和我们的 C 很不一样。

我们现在有：

\[
L_E
=
D(W_{\rm frozen}(H,\hat\tau),E^{GT}).
\]

这就允许一种危险现象：

\[
\boxed{
\hat\tau
\text{不一定物理正确，只要 frozen World 觉得它会产生正确 effect。}
}
\]

我们已经看到 negative world-gap，这正是这种风险。

PointWAM 基本没有这个 model exploitation 通道。

---

# 4. PointWAM 的“world model”其实更像强辅助监督，不是规划器

这一点很容易被名字误导。

PointWAM 确实证明了 world modeling 很有用：

去掉 scene-trajectory supervision：

\[
64.4\%\rightarrow53.5\%
\]

成功率掉了：

\[
\boxed{10.9\text{ percentage points}}
\]

说明 world supervision 很重要。[arXiv](https://arxiv.org/abs/2610.02840)

但它的作用路径其实是：

\[
\boxed{
\text{scene prediction}
\rightarrow
\text{改善 shared representation}
\rightarrow
\text{更好的 hand trajectory}
\rightarrow
\text{更好的 action}
}
\]

而不是：

\[
\text{World model online评价动作}
\rightarrow
\text{选动作}.
\]

所以 PointWAM 的成功并不能证明：

> frozen World effect consistency 一定可以指导我们的 actor。

反而它说明一个更朴素的方案就已经很强：

\[
\boxed{
\text{joint scene+hand prediction}
+
\text{robot BC action supervision}
}
\]

---

# 5. PointWAM 也没有我们现在这么“欠约束”的 goal

我们现在是：

\[
(H,\;P^{obj}_{t+24})
\rightarrow\tau.
\]

一个 object endpoint 必须反推出：

- approach；
- grasp side；
- finger closure；
- hold；
- lift。

这个 inverse problem 非常难。

PointWAM 输入的是：

\[
\text{scene point cloud}
+
\text{current hand}
+
\text{language instruction}.
\]

比如任务固定为类似“pick up bucket”。

DexJoCo multi-task 中每个任务有一个固定 instruction。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

语言本身当然也没有告诉它“拇指转多少度”，但它不需要解析出唯一 grasp：

> 机器人 demonstration 已经告诉模型，在这个 scene + instruction 下通常应该怎么抓。

所以 PointWAM 更像：

\[
\boxed{
\text{从 demonstration distribution 中模仿一个可执行 grasp mode}
}
\]

而我们是在做：

\[
\boxed{
\text{从 desired object effect 反求 grasp trajectory}
}
\]

后者其实难得多。

这也是为什么我一直觉得我们已经在不知不觉间“比 PointWAM 做了更难的问题”。

---

# 6. PointWAM 的机器人 fine-tuning 数据规模也远大于我们当前 own-state coverage

DexJoCo 中，剩余十个任务共有：

\[
959
\]

条能成功 replay 的 robot demonstrations，大致每任务约 100 条。

PointWAM 在这上面继续：

\[
60k\ iterations
\]

global batch：

\[
192.
\] :chatgpt-content-reference{index="6"}


而我们当前真正的 autonomous C exact-state：

\[
\boxed{3}
\]

个：

\[
t32,\ t40,\ t48.
\]

虽然有 192 个 expert continuation state，但真正 C 自己产生的 takeover state 就 3 个。

这个数量级差别非常大。

所以 PointWAM 并不是：

> “只靠 1.15M human videos，零 robot data 就抓起来了。”

恰恰相反：

\[
\boxed{
1.15M\ human
\rightarrow
959\ robot demos
\rightarrow
60k\ robot fine-tuning
}
\]

才得到最终结果。人类预训练贡献很大，但 robot fine-tuning 是完整的一阶段。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

---

# 7. 它还直接预测一个完整 action chunk

PointWAM retargeter：

\[
g(\hat V,s)
\rightarrow
A_{t:t+T}
\]

直接预测整段：

- end-effector pose；
- finger joint target。

而不是我们现在“预测 8 步但每 tick 只执行第一步”。

DexJoCo 中 chunk 长度是 30，约：

\[
0.6s.
\]

评估器不是每一个 20 ms 都重新把整段计划扔掉，而是 action buffer 执行若干步后再异步补新 chunk。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

这可能对 contact phase 很重要。

因为抓取最怕：

```text
tick54：准备闭合
tick55：重新规划另一条trajectory
tick56：又换一个接触模式
```

我们现在 first-only MPC 风格的执行可能会放大这种 jitter。

我把这个看作次要但值得检查的区别。

---

# 所以 PointWAM 为什么能抓，而我们还不能？

我会浓缩成一句：

\[
\boxed{
\text{PointWAM 把 }\tau\text{ 当作“受 GT 约束的物理中间表示”，}
}
\]

而我们现在逐渐把 \(\tau\) 训练成了：

\[
\boxed{
\text{一个同时要满足 World、R、object goal 的内部控制 latent。}
}
\]

于是它的物理语义被慢慢破坏了。

PointWAM 是：

\[
\text{GT hand trajectory}
\longleftrightarrow
\boxed{\tau}
\longleftrightarrow
\text{GT robot action}.
\]

我们现在变成：

\[
\text{object goal}
\rightarrow
\boxed{\tilde\tau}
\rightarrow
R_{\rm old}
\rightarrow
a
\]

同时：

\[
W_{\rm frozen}(\tilde\tau)
\rightarrow E.
\]

约束多了，但中间变量反而不再干净。

---

## 这给我们一个很明确的启发

如果我们想真正借 PointWAM，而不是只借外形，我现在反而建议**把机器人阶段简化**：

\[
\boxed{
L
=
L_{\tau}^{robot}
+
\lambda_a L_{action}
+
\lambda_s L_{scene}
}
\]

先暂时拿掉：

\[
L_E(W_{\rm frozen}(\hat\tau)).
\]

然后固定一个新的 paired training：

```text
robot state + scene + goal
        ↓
      actor
        ↓
  physical τ_hat
        ↓
   repaired/new R
        ↓
      action
```

**actor 和 R 一起训练。**

同时强制：

\[
\hat\tau\approx\tau^{GT}_{robot}
\]

保证 \(\tau\) 还是物理 hand trajectory。

也就是说下一步不应该继续：

> “actor 修一次 → R 修一次 → actor 再修一次”。

而应该像 PointWAM：

\[
\boxed{
(\text{actor},R)
\text{作为一个 pair 联合 robot fine-tune。}
}
\]

World 先回到 auxiliary scene prediction 的角色。

等这个 pair 能：

\[
\text{稳定 grasp}
\]

之后，我们再把 World/RL 加回来研究：

\[
\text{world model 是否能进一步提高动作选择。}
\]

这样反而更适合你的毕设主线——先复现一个真正能工作的 PointWAM-style supervised base，再证明 **world-model/RL 在这个 base 上带来额外收益**。

我现在认为，这是 PointWAM 和我们当前实现之间最关键、也最可能解释“为什么它能抓而我们抓不住”的差别。 [Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)