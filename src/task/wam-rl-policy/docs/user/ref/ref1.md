我看了今天最新仓库。现在其实已经有答案雏形了：**我不建议“BC 把 shared 更新完，然后直接把整个 shared+actor 丢进 PPO 继续训”**。更适合我们当前状态的是：

\[
\boxed{\text{BC base/reference policy}+\text{冻结 WAM}+\text{小 RL actor}}
\]

但不要做成永远只能 `base + residual` 的死结构，而是更接近 **RLToken 的 reference-conditioned refinement**。

仓库里目前正好已经走到这一步：`trajectory-policy` 新增了 `v2-base-residual-adapter`，已经验证 **zero residual 时 RL 接口输出和 v2 BC 原生执行完全一致**；但还没有真正开始 PPO。同时，今天最新的 matched 实验又说明：把当前 mean-pooled shared 直接塞进 BC 控制器，TRAIN 更好、DEV 反而更差，因此现在更不应该让 RL 直接更新整个 shared。

---

## 先把三种方案分清楚

### 方案 A：BC 训练完，直接把 BC actor 拿去 PPO

即：

\[
\pi_{\rm BC}(o;\theta)
\]

BC 结束后直接：

\[
\theta_{\rm BC}\rightarrow\theta_{\rm PPO}
\]

然后 PPO 更新整个：

\[
\text{WAM/shared}+\text{actor}.
\]

这个最简单，但**我现在不推荐作为主路线**。

因为我们 WAM 有约 1 亿参数，而真正 robot control head 很小。PPO 的 reward 信号远比九源预测监督弱。如果整个 backbone 都放开，容易发生：

\[
\text{RL reward gradient}
\rightarrow
\text{破坏 pretrained physical representation}.
\]

而且今天的 matched 结果已经告诉我们，目前 `shared → control` 的 readout 都还没有证明能泛化。现在直接让 PPO 改 shared，最后即使性能变化，也很难知道到底是在学控制还是把 WAM 改坏了。

所以：

\[
\boxed{\text{第一阶段不要用 RL 更新 WAM backbone}}
\]

---

# 方案 B：BC 永久作为 base policy，RL 学 residual

仓库目前已经在为这个做接口：

\[
a_{\rm BC}=\pi_{\rm BC}(o)
\]

\[
\Delta a=\pi_{\rm RL}(o)
\]

\[
a=a_{\rm BC}+\alpha\Delta a.
\]

这个有一个非常大的好处：

### 初始 RL policy 自动就是 BC policy

令：

\[
\Delta a=0
\]

那么：

\[
a=a_{\rm BC}.
\]

这正是今天刚验证的 zero-residual contract。

所以它对我们现在做第一个 RL 实验非常合适。

但我不希望**最终架构永远停在这里**。

原因很简单：

\[
a=a_{\rm BC}+0.15\Delta a
\]

实际上限定了策略只能在 BC 附近移动。

如果 BC 本身某些地方就错了，RL 很难完全摆脱它。

而且我们之前旧 WAM 的 residual PPO 已经做过类似实验，最终 residual 越来越大但还是抓不稳。那个实验不能否定现在的 v2 BC residual，因为 base 完全不同，但至少说明：

> permanent residual 不应该被当成最终方法。

---

# 方案 C：我现在最推荐——**RLToken-style**

这个比纯 residual 多一步，也正好符合我们最终论文逻辑。

RLToken 本身的思路是：

1. 大模型冻结；
2. 从大模型内部提一个 compact token；
3. base model 给 reference action；
4. 小 actor/critic 做在线 RL；
5. actor 同时看到 token 和 reference action；
6. 用 BC/reference regularization 限制早期探索；
7. 但 actor 有独立产生动作的能力，而不只是死 residual。

Physical Intelligence 的 RLT 确实就是这种结构：VLA 冻结，RL token 提供紧凑状态表示，小 actor/critic 在线训练；actor 接受 VLA 的参考 action chunk，并用 reference regularization 保持初期行为接近 base，同时还使用 reference-action dropout，避免 RL actor 只能复制 base。[pi.website](https://www.pi.website/research/rlt?utm_source=chatgpt.com)

这个和我们非常匹配。

---

# 我建议我们最终这样接

现在先定义：

\[
o_t=
[\text{state87},\text{current721},G]
\]

冻结 WAM：

\[
\boxed{
z_t=f_{\rm WAM}(H_t,G_t)
}
\]

BC 给：

\[
\boxed{
a_t^{BC}=\pi_{BC}(o_t,z_t)
}
\]

然后真正的 RL actor 是：

\[
\boxed{
a_t^{RL}
=
\pi_\phi(o_t,z_t,a_t^{BC})
}
\]

critic：

\[
\boxed{
V_\psi(o_t,z_t)
}
\]

或者以后 off-policy 时：

\[
Q_\psi(o_t,z_t,a_t).
\]

这里最重要的是：

> **RL actor 不需要就是 BC actor 本身。**

它是一个新的、小网络。

---

## 第一阶段可以初始化成 residual

为了保证刚开始不把机器人搞崩，可以写：

\[
\boxed{
a_t^{RL}
=
a_t^{BC}
+
\alpha\Delta a_\phi
}
\]

并且：

\[
\Delta a_\phi\approx0.
\]

这正好利用仓库刚完成的 zero-residual adapter。

但是与此同时，我会让 RL actor 有：

\[
(o,z)
\rightarrow a_{\rm absolute}
\]

这条独立路径。

比如：

\[
a^{RL}
=
m\cdot(a^{BC}+\Delta a)
+
(1-m)\cdot a^{absolute}.
\]

训练时偶尔把：

\[
a^{BC}
\]

drop 掉。

这就非常像 RLToken 的 reference-action dropout。

这样早期：

\[
a^{RL}\approx a^{BC}
\]

后期 RL 可以逐渐摆脱 BC。

---

# BC 正则怎么加

如果还是 PPO，我建议：

\[
L=
L_{\rm PPO}
+
\lambda_{\rm BC}
L_{\rm BC}.
\]

其中：

\[
L_{\rm BC}
=
\|a^{RL}-a^{BC}\|^2.
\]

早期：

\[
\lambda_{\rm BC}\text{ 大}
\]

然后逐渐：

\[
\lambda_{\rm BC}\downarrow.
\]

所以整个过程是：

\[
\boxed{
\text{BC imitation}
\rightarrow
\text{BC-anchored RL}
\rightarrow
\text{independent RL policy}
}
\]

而不是：

\[
\text{BC}\rightarrow\text{永远 residual}.
\]

---

# 那 shared 到底什么时候更新？

这个问题是关键。

我的答案是：

## **现在不要让 RL 更新 shared backbone。**

先：

\[
\boxed{\text{freeze WAM}}
\]

只训练：

- RL actor；
- critic；
- 一个很小的 WAM→RL-token adapter。

甚至 BC 阶段我现在也不建议继续大幅更新 WAM backbone。

今天的新实验已经很明确：

相同腕部 MLP：

| 输入 | TRAIN XYZ | DEV XYZ |
|---|---:|---:|
| no shared | 16.46 mm | **37.84 mm** |
| aligned shared | **14.37 mm** | 42.54 mm |
| shuffled shared | 15.13 mm | 40.52 mm |

也就是说当前的 mean-pooled shared：

> 更容易拟合 TRAIN，却没有带来泛化收益。

所以现在如果做：

\[
\text{BC}\rightarrow\text{update shared}
\rightarrow\text{PPO继续update shared}
\]

科学上会特别混乱。

---

# shared 应该变成真正的 RL token

这里我反而认为可以更接近 RLT 一点。

不是现在简单：

\[
2304D\ \text{mean pool}
\]

而是增加一个：

\[
\boxed{
z_{RL}\in\mathbb R^{64\sim256}
}
\]

learned query。

例如 WAM 有一堆：

\[
Z_{\rm scene},Z_{\rm hand},Z_{\rm goal}.
\]

通过一个小 cross-attention：

\[
q_{\rm RL}
\xrightarrow{\text{cross-attn}}
Z
\rightarrow z_{RL}.
\]

第一阶段离线训练这个 token，使其能够 reconstruct / distill WAM representation：

\[
D(z_{RL})
\rightarrow Z_{\rm WAM}.
\]

WAM 本身 frozen。

这与 RLToken 的核心思想就很接近：它同样用一个 bottleneck representation 去保留大模型内部的信息，然后冻结大模型，让小 actor/critic 使用 token。[pi.website](https://www.pi.website/research/rlt?utm_source=chatgpt.com)

所以：

\[
\boxed{
\text{WAM shared}
\neq
\text{直接作为 PPO 网络}
}
\]

而是：

\[
\boxed{
\text{WAM}
\rightarrow z_{RL}
\rightarrow
\text{small actor/critic}.
}
\]

---

# 这样我们的结构会非常清楚

### Stage 1：WAM pretraining

现在已经完成的九源：

\[
H,\tau,G
\rightarrow
F_{\rm WAM}
\rightarrow
Z.
\]

80k 已经跑完。

---

### Stage 2：Robot BC

冻结或极低 LR WAM。

训练：

\[
o,Z
\rightarrow
\pi_{BC}
\rightarrow a^{BC}.
\]

目的只有一个：

> 得到一个能工作的 bootstrap policy。

---

### Stage 3：RL token

冻结：

\[
F_{\rm WAM}.
\]

得到：

\[
z_{RL}=T(Z).
\]

---

### Stage 4：RL

每一步：

\[
a^{BC}=\pi_{BC}(o,z)
\]

\[
\boxed{
a^{RL}
=
\pi_\phi(o,z_{RL},a^{BC})
}
\]

然后进 Gym：

\[
a^{RL}\rightarrow env.
\]

reward 是真实任务 reward：

\[
r=
r_{\rm approach}
+r_{\rm lift}
+r_{\rm hold}
+r_{\rm goal}
-r_{\rm drop}
-r_{\rm clip}.
\]

critic：

\[
V(o,z_{RL})
\]

或者：

\[
Q(o,z_{RL},a).
\]

---

# 这里还有一个很重要的实验设计

其实这会非常自然地解决我们一直头疼的：

> **怎么证明 WAM 有用？**

RL 阶段直接做三组：

\[
\boxed{A:\quad \pi_{\rm RL}(o,a^{BC})}
\]

没有 WAM。

---

\[
\boxed{B:\quad \pi_{\rm RL}(o,z_{\rm shuffle},a^{BC})}
\]

placebo。

---

\[
\boxed{C:\quad \pi_{\rm RL}(o,z_{\rm WAM},a^{BC})}
\]

真正 WAM token。

三组：

- 同一个 BC；
- 同样 actor；
- 同样 critic；
- 同样 reward；
- 同样 PPO updates；
- 同样 reset；
- 同样 seed。

如果：

\[
C>A,B,
\]

那这个结果比我们以前“把 Cm 拼到 actor 里”干净得多。

它直接回答 MISSION B：

\[
\boxed{
\text{预训练物理表征能否提高RL policy improvement}
}
\]

---

# 所以你刚才提出的三个选项，我的排序是

| 方案 | 我现在的判断 |
|---|---|
| BC 后直接 PPO 整个 shared+actor | ❌ 不推荐 |
| BC 永久作为 frozen base + residual PPO | 🟡 很适合第一个工程 Probe |
| **RLT-style：冻结 WAM + BC reference + 小 actor/critic** | **✅ 主路线** |

也就是说：

> **“BC 当 base policy”是第一步，但“RLToken-style”才应该是我们最终 BC→RL 的接口。**

两者其实不是互斥的。

我们的 RLT-style 初期完全可以：

\[
a^{RL}=a^{BC}+\alpha\Delta a.
\]

只是以后不要把这个 residual 约束写死。

---

## 一个现实问题：现在是不是马上能开始？

代码层面已经非常接近。

今天仓库已经完成：

\[
\boxed{
a^{BC}
+
0\cdot\Delta a
=
a^{BC}
}
\]

的 v2 adapter 审计。

所以**最便宜的下一步不是再设计一个庞大 RL 架构**，而是先完成：

\[
\boxed{
\text{v2 BC}
+
\text{small residual PPO}
}
\]

的小实验，确认现在这个 BC substrate 上 PPO 至少能学。

但我会明确把它定义成：

> **RLT-style 的 Stage-0 engineering baseline**

而不是最终方法。

而且当前 Inspire BC 还不够稳：有效 current/8-step 权重目前只有杯子出现 held≈60，其他对象没有一致成功。因此如果马上做 RL，RL 很可能是在“从半失败 policy 学抓取”，而不是在“refine 一个 competent BC”。

反而 Shadow 当前 BC 已经有 **11/16 official、6/16 held** 的信号，做 BC→RL bootstrap 会更干净。

所以如果要并行推进，我会：

\[
\boxed{
\text{Shadow：先验证 BC→RL 机制}
}
\]

同时：

\[
\boxed{
\text{Inspire：继续把 BC substrate 做到稳定}
}
\]

然后把相同 RLT-style RL 接口搬过去。

这比现在直接对 Inspire 的 100M shared WAM 开 PPO 风险低很多。