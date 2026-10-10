有，而且看下来我觉得**我们现在不该死磕一个纯监督的 \(\tau\rightarrow A\) inverse model**。已有工作里，更成熟的做法通常是：

\[
\boxed{\text{kinematic reference }\tau
\;+\;
\text{current state}
\rightarrow
\text{closed-loop tracking controller}
\rightarrow
A}
\]

也就是说，把 \(\tau\) 当“参考轨迹”，不是把它当唯一输入去反解 action。

最相关的几篇是：

- **DexTrack（ICLR 2025）**：这篇最接近我们。它直接训练一个 generalizable neural tracking controller，让机器人跟踪 human/robot kinematic reference。它明确指出 dexterous manipulation 的难点就是 contact dynamics；做法不是解析求 \(\tau\to A\)，而是用 **RL + imitation learning + per-trajectory optimization** 学闭环 tracking controller。其代码还提供两种 action space：`cumulative residual positional targets with kinematic bias` 和 `relative positional targets`。[国际学习代表大会会议录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d5a1f97d2b922da92e880d13b7d2bf02-Abstract-Conference.html?utm_source=chatgpt.com)
- **REGRIND（2026）**：先把 human hand-object motion retarget 成一个保持 hand-object spatial/contact relationship 的 robot reference，然后训练 **residual RL policy** 去跟踪 object-centric keypoints。它的核心思想就是：kinematic reference 给出“应该怎么动”，RL 去补 friction、contact force、timing 等纯几何没表达的东西。[arXiv](https://arxiv.org/abs/2607.11874?utm_source=chatgpt.com)
- **DexMV（ECCV 2022）**：也是先做人手→机器人轨迹 retarget，再做 demonstration generation 和 policy learning。它的代码说明 demonstration generation 不只是 hand retargeting，还包括 trajectory generation、time alignment、inverse dynamics、hindsight，然后再用 IL/RL 学控制。[欧洲计算机视觉协会](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5261_ECCV_2022_paper.php?utm_source=chatgpt.com)
- **DexPilot / AnyTeleop 类**：主要做几何 retargeting，把人手关键点映射到机器人关节目标，适合解决“姿态怎么对应”，但它们本身不真正解决 contact preload / dynamic feasibility。DexPilot证明了全手 teleop 可行，但它更偏 kinematic mapping。[arXiv](https://arxiv.org/abs/1910.03135?utm_source=chatgpt.com)
- 更近的一篇 **Contact-Anchored Retargeting + Residual Policy Learning（2026）** 甚至直接强调：人类示范里缺失真正决定抓取成功的 contact force，因此用 contact-anchored retargeting 保留接触结构，再用 residual RL 把 kinematic recording 修成 physically consistent trajectory，并适配 robot actuation。[arXiv](https://arxiv.org/abs/2609.24093?utm_source=chatgpt.com)

所以这些工作的共识其实很明显：

\[
\boxed{
\tau\text{ 负责表达“想怎么运动/接触”}
}
\]

但

\[
\boxed{
A\text{ 不一定应该由一个纯 supervised inverse 一次性反解}
}
\]

而是更常见地用：

\[
A_t
=
A^{ref}_t
+
\delta A_t
\]

其中 \(A^{ref}\) 来自 kinematic retarget/reference，\(\delta A_t\) 由闭环 policy 根据当前 \(q,\dot q\)、object state、contact-related state 去动态补偿。

这正好解释我们现在的现象：

> hand trajectory RMSE 可以只有几毫米，但抓取仍然掉。

因为纯几何跟踪并不等于“接触力/PD preload正确”。

### 对我们最有启发的不是 PointWAM，而是 DexTrack / REGRIND

我现在会把我们的路线 B 改成更像：

\[
\boxed{
\tau^{hand}_{future}
+
s_t
\rightarrow
\pi_{\text{track}}
\rightarrow
A_t
}
\]

而不是一定要求：

\[
R(\tau,s)\rightarrow A_{t:t+23}
\]

一次性预测完整 24-step action chunk。

也就是说每一步：

\[
(\tau_{t:t+T},q_t,\dot q_t,\text{object/contact state})
\rightarrow A_t
\]

执行一步，重新观测，再输出下一步。

这样 finger preload、接触变化、物体漂移都可以通过闭环 policy 自己补。

如果再具体一点，我们甚至可以直接借 **REGRIND 的结构**：

\[
\boxed{
A_t
=
A^{kinematic}_t
+
\delta A^{RL}_t
}
\]

其中：

- \(A^{kinematic}\)：由我们的 11-point \(\tau\) 做解析/优化 retarget，至少 wrist 现在已经能做；
- \(\delta A^{RL}\)：专门负责 contact、preload、friction、timing；
- reward：主要跟踪 hand/object trajectory，不需要再硬监督“正确 preload 是多少”。

这个方向其实比我们现在训练一个 giant \(\tau\to A\) supervised decoder更符合已有 dexterous-control 文献。

**所以答案是：这个问题有人解决，而且主流不是把它当普通 inverse regression，而是把它当 reference tracking control。DexTrack 和 REGRIND 是我们现在最应该重点参考的两条。**