可以上，但我建议现在上的不是“端到端把 WAM、R、World 全部交给 RL”，而是先上 **Residual PPO**。以目前仓库状态看，这个时点已经合适了。

原因是现在 base policy 已经不是随机策略了。当前较好的 `robot-context actor + old Chunk-R` 已经能做到：

\[
\text{approach}\rightarrow\text{contact}\rightarrow\text{物体离桌}
\]

只是：

\[
\text{held}=0
\]

也就是说 RL 不需要从零探索“手怎么走到飞机旁边”，它只需要解决当前最缺的：

\[
\boxed{\text{接触建立、闭合、保持、抗滑移、lift 后持续跟随}}
\]

这正是 residual RL 比继续 BC 修几个 takeover state 更适合的地方。

---

## 我建议第一版就这样接

冻结当前 WAM 和一个固定 R：

\[
\hat\tau_t=A_{\rm WAM}(o_t,G_t)
\]

\[
a_t^{base}=R(s_t,\hat\tau_t)
\]

然后 PPO 只输出一个 12D native residual：

\[
\Delta a_t^{RL}\in\mathbb R^{12}
\]

实际执行：

\[
\boxed{
a_t=
\operatorname{clip}
\left(
a_t^{base}
+
\alpha\Delta a_t^{RL}
\right)
}
\]

第一版不要改 WAM，不改 R，不让 RL 反传进入 World。

这样实验非常干净：

\[
\boxed{
\text{WAM+R held0}
\quad\overset{RL}{\longrightarrow}\quad
\text{能否稳定held}
}
\]

---

## RL 的 observation 可以给得比 WAM 丰富

完全没必要再受 human-pretraining 输入合同限制。

我会至少给：

\[
q_t,\dot q_t,a_{t-1}
\]

当前 object：

\[
T_t^{obj},v_t^{obj}
\]

hand-object relative geometry，

WAM 给出的短期计划：

\[
\hat\tau_{t:t+8}
\]

以及当前 planned object reference：

\[
T^{ref}_{t+8},T^{ref}_{t+16},T^{ref}_{t+24}.
\]

即：

\[
\boxed{
\pi_{\rm RL}
(
s_t,
\hat\tau_{1:8},
G_t
)
\rightarrow
\Delta a_t
}
\]

这里 RL 的职责很明确：

> WAM 告诉我“大致怎么动”，RL 根据真实物理反馈修正“具体怎么抓稳”。

---

# Reward 不要再模仿 expert

这是最重要的。

不要主要用：

\[
-\|a-a^{expert}\|
\]

否则只是换一种方式继续 BC。

应该直接优化任务。

例如可以由几项组成：

\[
r=
w_p r_{\rm progress}
+
w_h r_{\rm hold}
+
w_g r_{\rm goal}
+
w_c r_{\rm contact}
-
w_a\|\Delta a\|^2
-
w_s r_{\rm slip}.
\]

其中最核心的是三个。

### 物体轨迹 progress

设 reference error：

\[
d_t=
d(T_t^{obj},T_t^{ref})
\]

奖励：

\[
r_{\rm progress}=d_{t-1}-d_t.
\]

不是奖励“当前误差小”而已，而是奖励：

> 我这个动作有没有把物体往正确方向推进。

---

### 稳定抓持

物体脱离支撑以后，如果 hand-object proximity 保持：

\[
r_{\rm hold}>0.
\]

现在我们的最大症状就是：

\[
\text{物体升高276mm}
\]

但：

\[
d_{\rm hand-object}\rightarrow364mm.
\]

RL 必须明确知道：

\[
\boxed{\text{把物体撞飞不算成功。}}
\]

---

### residual penalty

\[
r_{\rm residual}=-\|\Delta a_t\|^2.
\]

让 PPO 优先保留 WAM 先验，只在必要的时候纠偏。

这样后面换任务时，不容易退化成一个完全绕开 WAM 的独立 PPO。

---

# 训练初始化也要改变

这一点比继续采 t44/t47 特别重要。

不要全部 episode 都从 frame0 开始。

第一阶段可以从 successful reference 的不同 phase 随机初始化，例如：

\[
s_0\sim
\{
\text{approach},
\text{pre-contact},
\text{contact},
\text{early lift}
\}.
\]

并且随机扰动：

\[
q+\epsilon_q
\]

\[
T^{obj}+\epsilon_o
\]

\[
\text{finger state}+\epsilon_f.
\]

所以 RL 学的是：

\[
\boxed{
\text{一片状态区域里的恢复策略}
}
\]

而不是：

> tick47 应该输出哪一个固定动作。

这正好解决你上一条担心的“持续性”。

---

# 我会先用 old R 还是 repaired R？

**第一版 Residual PPO，我会用目前物理闭环表现更好的 old R + robot-context actor。**

因为这个 pair 至少出现过：

\[
\min gap=0.913mm
\]

和：

\[
max\ height=276.6mm.
\]

虽然 held0，但已经进入了 RL 能得到任务信号的区域。

而 repaired R 虽然：

\[
R(s,\tau^{GT})
\]

更正确，但和当前 actor 不兼容，直接替换后：

\[
maxheight\approx2.7mm
\]

而且后期 clipping 很高。

所以第一版不要同时换 R 和加 RL，否则归因又混了。

---

# 但我不会现在就让 World 参与 PPO actor

我们真正想做的是“强化学习 + 世界模型”，但最好分两步。

### Phase A：先建立 RL baseline

\[
\boxed{
\text{WAM+R+Residual PPO}
}
\]

先证明：

\[
0\ held\rightarrow稳定抓持.
\]

如果这一步都做不出来，先调 RL/controller，不碰 World。

---

### Phase B：World 先进入 Critic

等 PPO baseline 有了，再做：

\[
\hat E=W(H,\hat\tau)
\]

或者 World latent：

\[
z_W
\]

只给：

\[
V_\phi
\]

即：

\[
V_\phi(s,\hat\tau,z_W).
\]

比较：

\[
\text{PPO}
\]

和：

\[
\text{PPO + World-conditioned critic}.
\]

看：

- sample efficiency；
- success；
- hold duration；
- perturbation robustness。

这个实验会非常干净。

如果有效，再考虑：

\[
\text{World-conditioned actor}
\]

或者 candidate planning。

---

# 现在唯一需要注意的是：不要把 RL 当成掩盖 supervised baseline 问题的工具

你刚才说：

> airplane 单任务理论上应该可以过拟合。

这个判断我仍然认可。

所以我会**并行保留一个 sanity track**：

\[
\text{full robot BC overfit}
\]

去回答：

> PointWAM-style supervised pair 本身到底能不能抓？

但这不妨碍 RL 主线现在启动。

两条线承担不同问题：

\[
\boxed{
\text{BC-overfit}
\rightarrow
\text{检查 supervised pipeline 有没有结构性 bug}
}
\]

\[
\boxed{
\text{Residual PPO}
\rightarrow
\text{建立真正具备闭环纠错能力的方法}
}
\]

---

## 如果现在让我定实验优先级

我会排：

1. **冻结当前最佳 WAM+old R，建立 12D Residual PPO。**
2. 先做 airplane 单任务，目标只要求 stable grasp/lift。
3. phase-randomized reset，不做 tick-specific DAgger。
4. reward 只用真实 task outcome/contact/progress，不模仿 expert。
5. PPO 能稳定 held 后，再加 World 到 critic。
6. 同时用一个独立的小实验做 PointWAM-style full-BC overfit sanity check。

所以结论是：

\[
\boxed{\text{现在可以上 RL，而且从研究路线看已经到了该上的时候。}}
\]

但第一步应该是一个**干净、冻结 base 的 residual RL baseline**，而不是再把 WAM、R、World 全部一起改。