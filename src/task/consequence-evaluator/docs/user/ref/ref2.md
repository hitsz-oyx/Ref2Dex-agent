我看完 `consequence-evaluator` 最新分支了。当前 head 是 `3666bbe`，最后一个提交其实是 EPIC-Contact 审计；Evaluator 主线最新状态仍是：**新的 s1 parent 在重建/资格检查链上，真正的 E0 / Eoracle 还没有用真实 rollout 训练。**

所以现在不是“evaluator 实验又失败了”，而是**它还没真正跑到 evaluator；同时我认为当前 Probe 定义本身确实还有几个需要先修的地方。**其中第一个最严重。

1. **现在的 \(A_{1:24}\) 不是规划时能知道的 action chunk。** `prepare_windows.py` 直接取 `action[t:t+24]`；但 `collect_continuous.py` 每一步都是重新执行 `base = self.get_action(obs, True)`，下一步 action 会依赖上一步执行后的新 observation。也就是说：

\[
a_{t+1}=\pi(H_{t+1}),\quad
a_{t+2}=\pi(H_{t+2}),\ldots
\]

所以在时刻 \(t\)，你其实根本不知道完整的

\[
A_{t:t+23}.
\]

它是**未来真实物理反馈发生之后才形成的闭环 action trace**。

这会导致：

\[
E_0(H_t,A_{t:t+23})
\]

已经偷偷获得很多未来信息。例如发生 slip 后，六专家后续 action 自然会改变；于是 E0 可以从“未来策略怎么反应”推断之前发生了什么。

因此当前：

\[
E_{\rm oracle}(H,A,Z_{GT})-E_0(H,A)
\]

**不能作为我们世界模型的干净上限。**

分支 README 其实已经意识到了这一点，里面明确写了：

> `Recorded reactive future A 已受未来反馈影响`

但目前只是作为 limitation 记录，没有真正修掉。

最自然的修法其实就在你们现有代码里：`collection.py` 在 episode 开始时已经预先生成了完整的 **24步 smooth residual chunk**：

\[
\delta_{1:24}.
\]

这个 \(\delta\) 是在执行之前就确定的，因此是真正可以在决策时知道的计划。控制过程则是：

\[
u_{t+k}
=
\pi(H_{t+k})+\delta_k.
\]

所以第一版 evaluator 更合理的是：

\[
\boxed{E_0(H_t,\delta_{1:24})}
\]

对比：

\[
\boxed{E_{\rm oracle}(H_t,\delta_{1:24},Z_{GT})}.
\]

这还有一个好处：**完全不需要 fork。** 六专家仍然是反馈控制器，只是在未来24步叠加一个事先确定的 residual plan。

现在 `actual_residual` 已经保存在 diagnostics 里，只需要把“预先确定的 residual chunk”正式提升成 evaluator 的 action input，而不是拿事后形成的闭环 native-control trace 当 \(A\)。

---

2. **当前 \(Z_{GT}\) 太窄，不太像我们刚讨论的 Motus \(Z\)。**

现在 `data.py` 里的 oracle future 只有：

\[
Z_{GT}
=
T_t^{-1}T_{t+1:t+24}^{object}
\]

也就是**单个 anchor object 的24步 SE(3)**。

但标签却在判断：

- maintained hold；
- unrecovered drop；
- lift achieved；
- grasp lost。

这些本质上很多是：

\[
\text{hand-object interaction}
\]

而不只是：

\[
\text{object pose}.
\]

尤其 `grasp_lost`，两个未来可能物体位姿都差不多，但一个手已经松了，一个仍稳定包络物体。

Motus 的 \(Z\) 是 future observation/video sequence，手、物体、中间过程都在里面。我们现在把它缩成了单物体 SE(3)，其实是在测试：

> “GT object effect 是否能帮助 evaluator”

而不是：

> “GT physical future 是否能帮助 evaluator”。

我会至少做成两个 oracle：

\[
E_{\rm oracle-E}:
(H,\delta,Z_{\rm object})
\]

和

\[
E_{\rm oracle-EI}:
(H,\delta,Z_{\rm object},Z_{\rm interaction}).
\]

这里不用搞复杂，\(I\) 最开始完全可以是：

\[
\text{未来手关键点相对物体的位置}
\]

或者未来 hand-object relative geometry。

这样如果：

\[
E_0 < E_{\rm oracle-E} < E_{\rm oracle-EI},
\]

我们才能真正知道 interaction future 有没有额外价值。

---

3. **现在的 contact 标签不能真的叫 GT。**

当前 `collect_continuous.py` 是：

\[
handContact =
\text{任意手部刚体净接触力}>0.1
\]

\[
objectContact =
\text{物体净接触力}>0.1
\]

最后：

\[
contact = handContact \land objectContact.
\]

问题是这并不能证明：

\[
\text{hand}\leftrightarrow\text{object}
\]

真的彼此接触。

例如：

- 手碰桌子；
- 物体也碰桌子；

两个条件都可能成立。

分支文档自己也承认这是：

> `native hand_and_object net_force proxy`

而不是 hand-object collision pair。

这会直接污染 Robometer-style preference，因为 `grasp_lost / maintained_hold` 都依赖这个 contact。

在真正大量采数据前，我会先把这个 label source 核准。至少需要拿一批轨迹，与真实 hand-object pairwise contact 或几何距离做对照。如果没办法可靠拿 pairwise contact，就不要把这个 proxy 当唯一 GT，而应该组合：

\[
\text{物体抬升}
+
\text{手物距离}
+
\text{持续性}
\]

来判断事件。

否则我们又会回到以前的问题：

> evaluator 学不起来，到底是未来信息没用，还是 label 本身错？

---

4. **当前 preference pairing 仍然有明显 confound。**

现在只要求：

\[
\text{same task}
+
\text{same phase}
+
|\Delta h|<1cm
+
\text{different episode}.
\]

但没有要求：

- 同一个 expert；
- 同一个 motion/reference；
- 当前 object pose 接近；
- 当前 hand pose 接近；
- \(H\) 本身接近。

因此完全可能出现：

\[
(H_A,\delta_A,Z_A) \succ (H_B,\delta_B,Z_B)
\]

其实 A 本来就处于一个容易成功的状态，而 B 本来就很差。

这时：

\[
E_0(H,\delta)
\]

光看 \(H\) 就能排对。

那么即使 \(Z\) 非常重要：

\[
E_{\rm oracle}-E_0
\]

也会被压小。

如果坚持不 fork，这个问题不能完全消失，但至少可以严格匹配：

\[
\boxed{
task + expert + motion + phase
+
\text{current object pose}
+
\text{current hand pose}
}
\]

再在里面找 preference pair。

甚至可以用当前 \(H\) 的距离做最近邻配对。

这仍然是 observational Probe，但干净很多。

---

5. **当前 evaluator 的 action 接口跟 PointWorld 还是没真正接上。**

现在 evaluator 输入的是：

\[
18D\ native\ control.
\]

而你们正在训练的 PointWorld action 是：

\[
24\text{步 hand point-flow}.
\]

所以即使当前 oracle Probe 得到：

\[
E_{\rm oracle}\gg E_0,
\]

也还不能直接说：

> 现在这个 PointWorld 能补这个 oracle gap。

分支 README 也明确写了：

> `当前 A 仍是 native control，PointWorld 的手部 point-flow action 适配尚未实现`

这个问题不是必须现在马上解决，但需要从一开始把实验含义说清楚。

---

### 还有当前工程状态本身

之前那个：

\[
0/64
\]

的 parent qualification **不能再当真实失败结果**。

后来已经发现 r4 reset 有严重错误：

\[
\text{first-step object jump}\approx1.48m.
\]

修复 batched GPU setter 后变成：

\[
1.43mm.
\]

而且原始 s1 reference tensor 与 r4 使用的 canonical tensor根本不等价：

- object 初始位置差很多；
- wrist translation 最大差约 1.316m；
- wrist rotation 最大差约 1.779rad；
- finger 最大差约 \(\pi\)。

现在 `567c909` 已经换回 recovered original s1 + 正确 reset，smoke 里已经出现真实 contact/lift signal，所以这条重建比 r4 合理得多。

因此当前真正状态应该理解成：

\[
\boxed{
\text{六专家 substrate 尚未恢复完成}
}
\]

而不是：

\[
\text{Evaluator 又失败了}.
\]

---

## 我现在会怎么改这条分支

我不会继续原封不动地跑当前 E0/Eoracle。

应该先把核心合同改成：

\[
\boxed{
A=\delta_{1:24}
}
\]

也就是**事先生成、决策时已知的 residual plan**。

然后：

\[
E_0(H,\delta)
\]

对比：

\[
E_{\rm oracle-E}(H,\delta,Z^{GT}_{object})
\]

再对比：

\[
E_{\rm oracle-EI}(H,\delta,Z^{GT}_{object+interaction}).
\]

Preference 仍然用连续 rollout 生成，不 fork，但把 pairing 收紧到同 expert / motion / phase / 相近当前状态。

这样三条结果才真正有解释力：

\[
E_0
\rightarrow
E_{\rm oracle-E}
\rightarrow
E_{\rm oracle-EI}.
\]

如果第一跳明显：

> object future 有价值。

如果第二跳还明显：

> interaction future 也有价值。

最后 PointWorld 再去恢复：

\[
Z^{GT}\rightarrow\hat Z.
\]

**我认为这才是当前分支最应该先改的地方。现在最大的 conceptual bug 不是网络结构，而是把“事后形成的闭环24步动作”当成了决策时可用的 action chunk。**这个不修，后面的 oracle headroom 数字即使出来，也很难解释。