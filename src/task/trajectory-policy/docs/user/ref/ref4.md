可以。把前面讨论收敛以后，我认为我们最终可以做成一套比较完整、而且逻辑闭合的 **Goal-Conditioned Point WAM**。它既能吃大规模人类视频，也能接我们现在已经跑通的 DExplore→PD-DAgger 机器人链。

## 1. 最终任务定义

我们不把任务写成 language：

\[
\text{“pick up the airplane”}
\]

而定义成一个**短程目标状态变化**。

当前时刻 \(t\)，观察过去 4 帧：

\[
H_t=P_{t-3:t}
\]

预测 horizon 固定：

\[
K=24\approx0.8s.
\]

目标 \(G\) 不给完整未来轨迹，只描述：

> 这个物体/交互区域在 0.8 秒后希望变成什么状态。

最通用定义不是 SE(3)，而是一对无序点集：

\[
\boxed{
G_t=(P_{\text{start}}^G,P_{\text{goal}}^G)
}
\]

训练视频里：

\[
P_{\text{start}}^G=P_t,\qquad
P_{\text{goal}}^G=P_{t+24}.
\]

如果机器人规划器直接给目标点云，则：

\[
P_{\text{start}}^G=P_t^{robot},
\qquad
P_{\text{goal}}^G=P_{target}.
\]

以后如果输入是一段人类示范：

\[
P_{\text{start}}^G=P_{\text{demo,start}},
\qquad
P_{\text{goal}}^G=P_{\text{demo,goal}}.
\]

所以整个系统永远认一种 goal：

\[
\boxed{\text{point-set transition goal}}
\]

而不关心它来自规划器、仿真还是人类视频。

---

# 2. 整体网络

我建议最终结构就是：

\[
\boxed{
\begin{aligned}
H_{\rm scene}
&\xrightarrow{F_{\rm scene}}
Z_H\\
G=(P^G_s,P^G_g)
&\xrightarrow{F_{\rm goal}}
Z_G\\
Z_H,Z_G,H_{\rm robot}
&\xrightarrow{A}
\hat\tau_{\rm hand}\\
Z_H,\hat\tau_{\rm hand}
&\xrightarrow{W}
\hat E_{\rm scene}\\
\hat\tau_{\rm hand},s_t
&\xrightarrow{R}
\hat q^{des}_{PD}
\end{aligned}
}
\]

这里五个模块职责非常明确。

---

## 3. Scene Encoder：回答“现在是什么状态”

这一部分直接继承我们现有 PointWorld 最有价值的东西。

输入不是 H328 这种 flatten vector，而是过去 4 帧真实 point history：

\[
P_{t-3:t}.
\]

里面可以包含：

- 目标物体/interaction region 点；
- hand points；
- 周边相关 scene points；
- point velocity；
- acceleration；
- history displacement；
- normal；
- semantic/object mask。

经过 PointWorld/PTv3 backbone：

\[
F_{\rm scene}(H)\rightarrow Z_H.
\]

这里可以直接用我们已经预训练的 PointWorld scene-side 权重初始化：

- `scene_proj`
- PTv3 backbone
- point history representation

但**不沿用原来未来 action token 作为输入的那一部分**。

\(Z_H\) 表示：

\[
\boxed{\text{当前世界及其短期动态状态}}
\]

---

# 4. Goal Encoder：回答“我想让世界变成什么样”

这里不需要 VAE。

输入两组无序点：

\[
P_s^G,\quad P_g^G.
\]

分别经过共享 point encoder：

\[
Z_s=F_G(P_s^G),
\qquad
Z_g=F_G(P_g^G).
\]

再通过少量 cross-attention：

\[
\boxed{
Z_G=
\operatorname{GoalFuse}(Z_s,Z_g)
}
\]

它学习的是：

> 这一团点从 start 变成 goal，意味着什么短程运动意图。

不是：

\[
p_i^{start}\leftrightarrow p_i^{goal}
\]

的显式匹配。

点：

- 数量可以变化；
- 采样可以变化；
- index 可以变化；

都没关系。

第一版甚至可以控制在：

\[
N=128
\]

点左右，完全够用。

---

# 5. Action / Hand Trajectory Head：真正的 policy

这是最终真正输出动作意图的部分：

\[
A(Z_H,Z_G,H_{\rm robot})
\rightarrow
\hat\tau.
\]

其中：

\[
\boxed{
\hat\tau
\in
\mathbb R^{24\times11\times3}
}
\]

仍然使用我们现在已经验证过的 24 步、11 hand points 表示。

这里我不会马上引入 latent \(c\)、VAE、diffusion。

先直接 deterministic：

\[
(Z_H,Z_G)\rightarrow24\times11\times3.
\]

可以用 24×11 trajectory query tokens，对：

\[
[Z_H,Z_G]
\]

做 Transformer cross-attention。

这样 network 学的是：

> 当前物体现在这样运动，短程目标想变成那样，那么手应该怎么运动？

这就是我们的 action model。

---

# 6. World Head：真正体现 WAM 的部分

这一部分和 action head 有一个非常重要的区别：

\[
\boxed{W\text{ 不允许直接读取 }G}
\]

它只能看：

\[
Z_H
\]

和：

\[
\tau.
\]

即：

\[
\boxed{
W(Z_H,\tau)\rightarrow
\hat E_{1:24}.
}
\]

它回答：

> 当前世界是这样，如果手按照这条轨迹运动，物体实际上会怎么动？

这实际上就是我们现有：

\[
(H,\tau)\rightarrow E
\]

PointWorld 思想的延续。

---

## 7. \(E\) 不再要求刚体 SE(3)

这是这套框架一个很重要的变化。

我们最终统一预测：

\[
\boxed{
\hat P_{t+1:t+24}^{obj}
}
\]

也就是未来 point sets / point motion。

最简单可以让当前 \(N\) 个点各预测：

\[
\Delta\hat p_i^k.
\]

于是：

\[
\hat P_{t+k}
=
\{p_i^t+\Delta\hat p_i^k\}.
\]

如果数据有可靠 point tracks，就直接监督：

\[
L_{\rm flow}
=
\|\Delta\hat p-\Delta p\|.
\]

如果大规模视频只能得到每帧物体点集，没有可靠 correspondence，则用：

\[
L_{\rm set}
=
CD(\hat P,P^{GT})
+
\lambda_HD_{\rm soft}(\hat P,P^{GT}).
\]

也就是说：

\[
\boxed{
\text{有track就用强监督，没有track也能用无序点集监督}
}
\]

这样数据规模不会被 6DoF/object pose 卡住。

---

# 8. 为什么 world head 不能看 G

因为：

\[
G=P_{t+24}^{goal}.
\]

如果：

\[
W(H,\tau,G)\rightarrow E
\]

那么 h24 最简单的答案就是：

\[
\hat E_{24}=G.
\]

世界模型会走 shortcut。

所以结构上固定：

\[
\boxed{
A:\;H,G\rightarrow\tau
}
\]

但：

\[
\boxed{
W:\;H,\tau\rightarrow E
}
\]

这样 goal 告诉 policy “想去哪”，world model 负责判断“这么做实际上会去哪”。

这两个功能不能混。

---

# 9. Learned R：我们现在已经有现成基础

这一层没必要重新发明：

\[
\boxed{
R(s_t,\hat\tau)
\rightarrow
q^{des}_{PD,t}
}
\]

沿用现在已经验证成功的 actual-PD retargeter。

我们已经知道：

\[
GT\text{-}\tau + R
\]

在当前任务上能完成：

\[
8/8\text{ hold + controlled placement}.
\]

所以第一版 WAM **冻结 R**。

这样新的研究问题就只剩：

\[
\boxed{
(H,G)\rightarrow\tau
}
\]

是否通过 world pretraining 得到改善。

不要同时改 executor。

---

# 10. 大规模人类视频怎么训练

这就是这套方案最大的优势。

从一段 30 Hz human manipulation video 中取窗口：

\[
t-3,\ldots,t+24.
\]

自动生成：

\[
H=P_{t-3:t}
\]

\[
G=(P_t,P_{t+24})
\]

\[
\tau^{GT}=Hand_{t+1:t+24}
\]

\[
E^{GT}=P_{t+1:t+24}.
\]

所以不需要：

- language；
- task label；
- object 6DoF；
- robot action；
- success/failure label。

只要能够获得：

\[
\boxed{
\text{3D hand trajectory + scene/object point observations}
}
\]

就能训练。

这就很接近 PointWAM 的大规模视频优势。

---

# 11. 人类数据阶段的训练目标

这里**不要把 actor 输出的 \(\hat\tau\) 直接送进 world model 再用真实未来监督**，否则存在因果标签错配。

正确做法是两条监督并行：

### Action branch

\[
(H,G)
\rightarrow
\hat\tau
\]

监督：

\[
\boxed{
L_\tau
=
d(\hat\tau,\tau^{human})
}
\]

### World branch

用真实发生的人手轨迹：

\[
W(H,\tau^{human})
\rightarrow
\hat E
\]

监督：

\[
\boxed{
L_E
=
d(\hat E,E^{actual}).
}
\]

两个 branch 共享：

\[
F_{\rm scene}.
\]

总 loss：

\[
\boxed{
L_{\rm human}
=
L_\tau
+
\lambda_E L_E.
}
\]

这才是最干净的 PointWAM-style pretraining。

world prediction 强迫：

\[
Z_H
\]

包含 hand-object dynamics。

action head 又必须利用同一个：

\[
Z_H
\]

去产生 hand trajectory。

---

# 12. 机器人阶段：加 PD supervision

到了 DExplore 数据，我们有：

\[
H,\quad
G,\quad
\tau^{teacher},\quad
q^{des}_{teacher}.
\]

继续训练：

\[
(H,G)\rightarrow\hat\tau.
\]

在原来的：

\[
L_\tau+L_E
\]

之外加入：

\[
\boxed{
L_{PD}
=
\left\|
R(s,\hat\tau)
-
q^{des}_{teacher}
\right\|
}
\]

于是：

\[
\boxed{
L_{\rm robot}
=
L_\tau
+
\lambda_E L_E
+
\lambda_{PD}L_{PD}.
}
\]

这就是我们目前 main 已经证明有效的“PD gradient 回到 trajectory actor”，只是现在 actor backbone 已经经过大规模 WAM 预训练。

---

# 13. DAgger 仍然保留

纯 offline robot BC 之前已经证明不够。

所以继续：

\[
\pi(H,G)\rightarrow\tau\rightarrow R\rightarrow q^{des}
\]

rollout student。

进入 student state：

\[
s_t^{student}
\]

以后向 DExplore teacher 查询：

\[
q_{teacher}^{des}(s_t).
\]

这部分数据没有可靠 expert future hand trajectory，所以只训练：

\[
\boxed{
L_{PD}^{DAgger}
}
\]

不要伪造：

\[
L_\tau.
\]

这正好沿用现在已经成功的 PD-DAgger 逻辑。

---

# 14. 最终部署

部署时可以支持两种入口。

| 场景 | Goal 输入 |
|---|---|
| 机器人 planner 给任务 | 当前点集 \(P_t\) + 目标点集 \(P_{goal}\) |
| 人类示范给任务 | demo start point set + demo goal point set |

进入：

\[
F_G
\rightarrow
Z_G.
\]

当前机器人：

\[
H_{robot}
\rightarrow
F_{\rm scene}
\rightarrow
Z_H.
\]

然后：

\[
(Z_H,Z_G)
\rightarrow\tau.
\]

再：

\[
\tau
\rightarrow R
\rightarrow q^{des}.
\]

当前已经成功的 baseline 是每个 control tick 重规划，我建议第一版继续：

\[
\boxed{\text{actor period}=1}
\]

不要重新回到 8-step open-plan reuse。

---

# 15. World head 部署时怎么办？

它可以有两个层次。

**第一版**主要把它当 auxiliary pretraining head。部署时甚至可以不运行完整 future head，只保留被 world loss 训练过的：

\[
F_{\rm scene}.
\]

这已经是 PointWAM-style 的主要思路。

后面如果希望世界模型真正在线参与决策，可以再做：

\[
\hat\tau=A(H,G)
\]

\[
\hat E=W(H,\hat\tau)
\]

然后比较：

\[
d(\hat E_{24},G).
\]

如果偏差大：

- refine \(\tau\)；
- sample multiple \(\tau\)；
- MPC selection；
- RL。

但这属于第二阶段。

第一版不要同时加入。

---

# 16. 后续 RL 怎么接

先把 WAM+DAgger baseline 做到能自主完成任务。

然后才进入 RL。

我会保留：

\[
\tau
\]

作为 policy action representation。

可以冻结大部分：

\[
F_{\rm scene},F_G
\]

只训练小 trajectory head：

\[
A_\theta.
\]

这时候就有一点 RLToken 的味道：

\[
\boxed{
\text{大规模WAM预训练backbone}
+
\text{小RL trajectory head}
}
\]

但基础 representation 是 world-action pretraining 学出来的。

RL reward 使用真实：

- lift；
- hold；
- controlled placement；
- point-goal tracking；
- clipping/control cost。

而不是 reference q tracking。

---

# 17. 最关键的科学对照

最后不能直接拿“小 MLP DAgger”和“5000 万参数 PW-WAM”比较，否则无法说收益来自 world supervision。

必须做 matched ablation：

\[
\boxed{\text{Hand-only}}
\]

和

\[
\boxed{\text{WAM}}
\]

两边：

- 同一个 Point backbone；
- 同一个 Goal Encoder；
- 同一个 trajectory decoder；
- 同一个 robot data；
- 同一个 DAgger；
- 同一个 R；
- 同样训练步数。

唯一差别：

\[
\lambda_E=0
\]

vs

\[
\lambda_E>0.
\]

这样如果：

\[
\text{WAM}>\text{Hand-only},
\]

我们才能说：

\[
\boxed{
\text{world/scene future supervision genuinely improves the action policy}
}
\]

这才是核心贡献。

---

# 18. 评价不能继续只测当前 airplane nominal

因为现在 DExplore 和我们的 DAgger 都已经接近：

\[
100\%.
\]

所以最终 evaluation 应该有 headroom：

\[
\text{held-out short goals}
\]

\[
\text{不同 reference trajectories}
\]

\[
\text{initial object pose perturbation}
\]

\[
\text{initial hand pose perturbation}
\]

以及后面：

\[
\text{new object instances}.
\]

否则 100% vs 100% 看不出 WAM 的价值。

---

## 整套框架可以压缩成这一张主线

\[
\boxed{
\begin{array}{ccccc}
\text{Current point history }H
&\rightarrow&
F_{\rm scene}
&\rightarrow&
Z_H
\\
\text{Start/goal point sets }G
&\rightarrow&
F_{\rm goal}
&\rightarrow&
Z_G
\\
&&Z_H,Z_G
&\rightarrow&
\hat\tau_{1:24}
\\
&&Z_H,\tau_{\rm executed}
&\rightarrow&
\hat E_{1:24}
\\
&&\hat\tau
&\rightarrow&
R(s,\hat\tau)
\rightarrow q^{des}_{PD}
\end{array}}
\]

大规模视频预训练：

\[
\boxed{
L_{\rm human}
=
L_\tau+\lambda_E L_E
}
\]

机器人适配：

\[
\boxed{
L_{\rm robot}
=
L_\tau+\lambda_E L_E+\lambda_{PD}L_{PD}
}
\]

student rollout：

\[
\boxed{
L_{\rm DAgger}=L_{PD}^{query}
}
\]

最后再：

\[
\boxed{\text{WAM pretrained policy}\rightarrow\text{RL fine-tune}}
\]

---

我认为这条路线已经比较完整了：**目标不用 language、不依赖刚体 6DoF、大规模视频能直接训练、机器人端能复用现在已经成功的 PD-DAgger，而且 PointWorld 不再只是一个外挂 evaluator，而是真正成为 action policy 的 world-action backbone。**