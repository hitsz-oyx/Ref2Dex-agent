不用重新做一遍大规模预训练。现在更合理的是：

\[
\boxed{\text{保留现有几何预训练/WAM，新增 robot-specific control adapter，再做机器人阶段微调。}}
\]

你前面的理解是对的：**预训练阶段为了吃大规模视频/人手数据，可以只依赖几何和较粗 goal；真正部署到 Inspire 上时，没有理由故意把 \(q,\dot q\)、参考轨迹这些已知信息丢掉。**

### 我建议这样加

保留现有：

\[
\text{PTv3 scene}+ \text{raw hand}+32\text{-point endpoint goal}
\]

完全不动，再加两条只在 robot 数据上存在的条件。

第一条是 proprioception：

\[
x_{\rm prop}=
[q_t,\dot q_t,a_{t-1}]
\]

也就是：

\[
18+18+12=48D.
\]

做一个很小的 MLP：

\[
48\rightarrow128\rightarrow128
\]

得到：

\[
z_{\rm prop}.
\]

第二条是更明确的 robot goal。不要先上很复杂的 3PoinTr，直接给：

\[
T^{ref}_{t+8},T^{ref}_{t+16},T^{ref}_{t+24}.
\]

最好全部表示成相对当前物体的 SE(3)：

\[
\Delta T_k=(T_t^{obj})^{-1}T^{ref}_{t+k}.
\]

每个位姿用：

\[
3D\ translation+6D\ rotation
\]

所以：

\[
3\times9=27D.
\]

再编码为：

\[
z_{\rm goal}^{robot}.
\]

然后不要重新设计 decoder，直接做 residual conditioning：

\[
\boxed{
q^{new}_{i,t}
=
q^{old}_{i,t}
+
\alpha_p z_{\rm prop}
+
\alpha_g z_{\rm goal}(t)
}
\]

其中：

\[
\alpha_p=\alpha_g=0
\]

初始化。

这点非常重要，因为这样加完模型后：

\[
\boxed{\hat\tau_{\rm new}=\hat\tau_{\rm C-final}}
\]

初始输出应该严格不变，和我们 B2 的思路一样。

---

## 不需要让 human 数据拥有这些输入

OakInk / GRAB / ARCTIC / EgoDex 仍然：

\[
\text{geometry + coarse goal}\rightarrow\tau
\]

robot conditioner 对它们直接 mask 掉：

\[
z_{\rm prop}=0,\qquad z_{\rm robot-goal}=0.
\]

所以整体结构变成：

\[
\boxed{
\underbrace{\text{大规模 geometry prior}}_{\text{跨 embodiment}}
+
\underbrace{\text{robot control context}}_{\text{Inspire-specific}}
}
\]

这比重新训练整个 PTv3 合理很多。

---

# 真正需要重新训练的是 robot finetune，而且数据比模型改动更重要

我不建议现在马上拿原 t8/t16/t32 recovery 再跑 50k。

因为我们已经知道 C-final 真正崩的是：

\[
t32\rightarrow t56.
\]

所以在加新输入之前/同时，应该重新采当前 **C-final 自己访问的状态**：

\[
t32,\ t40,\ t48
\]

必要时 t56。

流程还是我们之前已经验证过的实际 takeover：

```text
C-final 自己执行
      ↓
跑到 t32 / t40 / t48
      ↓
expert 从这个真实 state 接管
      ↓
记录 future hand τ*
      ↓
记录 expert action
      ↓
记录真实 object effect
```

这样我们才能得到真正需要的：

\[
\boxed{
s_t^{C}
\rightarrow
\tau_t^{expert}
}
\]

而不是“旧 student 状态 + expert 已经救回来之后的状态”。

这批数据的价值比再增加几百万次重复 sampling 大得多。

---

# 训练我会分两层，不需要一上来大跑

### 第一阶段：只训练新 adapter

从 **C-final** 载入：

- PTv3 冻结；
-原 goal encoder 冻结；
- raw-hand 冻结；
- World 冻结；
- Chunk-\(R\) 冻结；
- 原 trajectory decoder 先冻结。

只训练：

\[
z_{\rm prop},\quad z_{\rm robot-goal},\quad\alpha_p,\alpha_g.
\]

先跑大约：

\[
1k\sim3k\ updates.
\]

这个实验很便宜，而且初始严格等价 C-final。

如果这时 contact-state 的：

- first8；
- first-PD；
- t32/t40/t48 recovery；

明显改善，再进入第二阶段。

---

### 第二阶段：小学习率解冻 decoder

然后：

\[
lr_{\rm adapter}\sim10^{-4}
\]

\[
lr_{\rm decoder}\sim10^{-5}.
\]

仍然不要解冻 PTv3。

数据我反而会让 robot 占主要部分，比如：

\[
20\%\ human\ replay
+
30\%\ nominal
+
50\%\ current\ C\text{-state recovery}.
\]

human 这时候主要是防 decoder 忘掉原几何能力，不再是训练主力。

这一步可以再跑：

\[
5k\sim20k.
\]

是否继续不是看 training loss，而是看 **新 C-state contact panel**。

---

# 暂时不要直接复制 DExplore 的 1442D

虽然我同意“控制阶段信息应该充足”，但我不建议第一版直接：

\[
1442D\rightarrow WAM.
\]

原因是 1442D 里有大量 DExplore 特定的 reference/task encoding，会一下子把：

> 到底是缺 proprioception，还是缺 reference，还是缺 contact 信息

全部混在一起。

我们现在其实可以用一个很小但关键的子集：

\[
\boxed{
q,\dot q,a_{t-1},
T^{ref}_{t+8},T^{ref}_{t+16},T^{ref}_{t+24}
}
\]

大约几十维，就已经补上当前最明显的信息缺口。

如果这个版本仍然在 contact transition 崩，再逐步加：

- fingertip contact flags；
- net contact force；
- target/current joint error；
- 更完整 DExplore observation。

这样实验逻辑更干净。

---

## 所以我的建议顺序是

\[
\boxed{
\text{C-final}
\rightarrow
\text{采 t32/40/48 own-state recovery}
\rightarrow
\text{加 zero-init proprio + temporal-goal adapter}
\rightarrow
\text{robot-only 小规模适配}
\rightarrow
\text{closed-loop}
}
\]

而不是：

\[
\boxed{\text{重新训练 PTv3 / 重新跑整个大规模预训练}}
\]

大规模预训练现在应该被当作一个**先验底座**；我们当前缺的是最后这一层“从通用几何理解到具体 Inspire 控制”的信息补全。

如果这个 robot adapter 最后有效，再考虑下一版从预训练阶段就加入“可选状态 token / 可选 temporal goal token”，让整个架构统一。现在没必要为这个把已经完成的大训练推倒重来。