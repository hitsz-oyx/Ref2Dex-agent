是，**现在确实遇到了一个关键问题，但不是 PointWorld，而是我们刚定义的 GT Value 标签本身。**

最新主分支到 `1b24c5b`。目前情况是：

\[
Y_t=P_{t+24}-P_t
\]

这条 reference-progress 标签，**全局看起来已经能工作，但局部还不可信**。

最明显的证据是 TCC 之后，4 条 clean 成功轨迹最终都能正确对齐到 reference 尾部：

\[
P_{\rm final}=0.932\sim0.994
\]

self-reference、static、因果 prefix 检查也都通过。这说明 TCC 确实把之前“整条轨迹卡在前面”的工程问题解决了。

但局部出现了严重反例：4 条 clean trajectory 在 `tick≈176` 附近全部得到

\[
Y_{24}\approx -0.023\sim-0.025
\]

也就是模型认为这 24 步在**倒退**，可是实际物体高度却大约：

\[
6\text{ cm}\rightarrow30\text{ cm}
\]

明显正在完成一次正常抬升。每条 clean episode 还有大约 5–8 个这样的负 \(Y\) 窗口，累计回退量约：

\[
0.249\sim0.368
\]

超过我们原先定的 0.2 gate。

所以目前 `training_allowed=false` 是对的。**现在绝对不应该拿这批 Y 去训练 evaluator。**

---

更重要的是，我觉得现在暴露出来的不是简单调 TCC 参数能解决的问题，而是一个更底层的问题：

\[
\boxed{\text{reference phase}\neq\text{task value}}
\]

我们的原始 reference 本身包含重复的 lift / return / place 段，而且它是一条**运动参考轨迹**，并不是一次真实机器人成功执行。

实际机器人成功 rollout 和它存在明显差异：

```text
原始reference:
指定的手/物轨迹
→ lift
→ return
→ 再发生类似状态
→ 最终place

真实机器人成功:
实际接触
→ 实际lift
→ 稳定
→ 实际place
```

于是同一个“物体正在抬升”的物理状态，在 reference 里可能对应多个类似阶段。

TCC 的目标只是：

> 找一个时序上一致的对应关系。

它并没有被训练成：

> “越接近任务成功，value 必须越大”。

所以即便：

\[
P_{\rm final}\approx1
\]

也完全不能保证：

\[
P_{t+24}-P_t
\]

每一个局部窗口都是正确的动作价值。

---

这也意味着，**我会修改之前“只用原始 reference 作为 canonical R”的建议。**

现在仓库已经给了我们足够证据：

> 原始运动 reference 可以定义任务，但不适合直接作为唯一的 value 时间轴。

我们已经有现成的 `source230` nominal robot rollout，62/64 是完整成功，而且它们是：

- Inspire 同一 embodiment；
- 实际 PhysX 执行；
- 测量的 11 个 hand points；
- 实际 object trajectory；
- 完整正常放回。

所以我认为下一步最便宜、也最有信息量的是：

\[
\boxed{\text{先用成功的真实机器人 rollout 做 reference bank}}
\]

不用重新采数据。

例如冻结 8 条 source230 成功轨迹：

\[
R_1,\ldots,R_8
\]

TCC/phase matcher 学这些**真实成功物理轨迹之间**的对应关系，再去给 source261 那 8 条独立轨迹打标签。

然后专门看刚才那个 `tick176`：

> 正常 lift 还会不会被打成负值？

如果这个问题消失，说明问题主要来自：

\[
\text{kinematic reference}\leftrightarrow\text{physical execution}
\]

的 gap。

---

但如果换成成功 physical reference bank 后，局部 \(Y\) 仍然频繁错，我就建议**停止使用**

\[
Y=\Delta P
\]

作为主 GT。

这时 TCC 仍然有用，只是不拿 phase index 当 value。

改成：

\[
\boxed{
\text{先用TCC定位当前phase}
\rightarrow
\text{比较候选GT future和该phase之后的成功future是否一致}
}
\]

也就是：

\[
Y(A)
=
-\sum_kq_t(k)
D\left(
\Delta Z_A^{GT},
\Delta R_{k:k+24}
\right)
\]

这就是我们前面说的 **transition alignment**。

它问的是：

> “候选动作产生的这 24 步物理变化，像不像成功轨迹在当前阶段应该发生的变化？”

而不是：

> “我在 reference index 上向前走了多少？”

对于：

```text
抓起 → 放下
```

这种存在重复状态的任务，这个定义要可靠得多。

---

所以当前完整链路没有崩，只是 **Gate0——GT value 定义还没有过关**：

\[
\boxed{
\text{成功物理reference}
\rightarrow
\text{GT candidate value}
\rightarrow
\text{Gate1滚动选择}
\rightarrow
\text{Oracle Evaluator}
\rightarrow
\text{PointWorld}
\rightarrow
\text{Execution Model}
}
\]

我现在建议**不要再调 temperature、max_step、epsilon 来救当前 \(\Delta P\)**。先用现有成功 robot rollout reference bank 做一次非常便宜的对照。这一步能直接判断到底是“原始 reference 不合适”，还是“phase-delta 本身就不适合作为 value”。