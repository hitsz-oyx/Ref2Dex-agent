我看了主分支最新状态。现在 `main` 已经推进到 **ref7：GT-hand retarget/control upper-bound**，最新提交是 `b259de4`。这次实验非常有信息量，而且我认为已经暴露出一个比“retargeter 网络没学好”更本质的问题。

目前结果是：

| 路线 | held | 手轨迹 RMSE |
|---|---:|---:|
| reactive teacher | **484** | 参考 |
| 固定 GT PD target servo | **117** | **4.35 mm** |
| learned retargeter | **42** | 135.89 mm |
| learned repeat | 0 | 136.05 mm |

另外刚刚又补了 `receding1`：每一步都重新预测，只执行最新 chunk 的第 0 个动作，结果仍然 **held=0**。所以 ACT 的 `8-step / temporal ensemble / hard switch` 这条排查线基本可以先放下了。`open_loop24` 能抓，receding1/8 都不能，问题不是简单缩短执行步数能解决的。

### 最关键的新发现：现在这个“GT servo”其实还不是我们真正想要的 GT-hand upper bound

代码里非常明确。

`retarget_execution.py` 初始化时：

```python
self.source_targets = commanded_targets(
    packet['dof_position'][:-1, 0],
    packet['actions'][:, 0]
)
```

执行时：

```python
desired = np.stack((
    self.source_targets[tick],
    self.chunk[tick % 24],
    self.chunk[tick % 24]
))
```

也就是说，那个 **Direct GT target servo** 实际做的是：

\[
\boxed{
\text{另一条成功 rollout 的 recorded PD target}
\rightarrow
\text{fresh rollout}
}
\]

然后利用当前 \(q_t\) 做机械反馈跟踪。

它并不是：

\[
\text{GT future hand geometry}
\rightarrow
\text{在线 inverse/retarget}
\rightarrow
\text{control}.
\]

所以它 held117 并不能证明：

> “hand trajectory 作为 action representation 不行。”

它证明的是另外一件事：

> **即使 fresh rollout 的手在世界坐标系中只偏离 source hand 4.35 mm，固定 replay source 的 PD target 仍不足以维持接触。**

这个区别很重要。

---

## 我现在最怀疑的是：世界系 hand trajectory 本身定义错了

现在 learned R 的输入构造也是：

```python
future = self.source_hand[...]
current_hand = live_hand
hand = future - current_hand
```

也就是：

\[
\boxed{
\tau^{hand}_{source,world}
-
H^{hand}_{live,world}
}
\]

它在追踪**另一条 rollout 的世界坐标手轨迹**。

但是这里完全没有当前 object pose。

假设 source 中：

\[
p^{hand}=0.500
,\qquad
p^{obj}=0.495
\]

两者相距 5 mm，保持接触。

fresh rollout 中物体因为 PhysX/contact 漂了 4 mm：

\[
p^{obj}_{live}=0.499
\]

你仍然让手去：

\[
p^{hand}=0.500.
\]

从“手轨迹误差”看：

\[
\text{RMSE}\approx0
\]

但从真正重要的 interaction 看：

\[
p^{hand}-p^{obj}
\]

已经完全不同。

所以现在出现：

\[
\boxed{
\text{hand RMSE}=4.35\text{ mm}
\quad\text{但 grasp 丢失}
}
\]

其实一点都不矛盾。

这甚至和我们前面一直讨论的 **interaction \(I\)** 是同一个问题。

---

# 我认为下一步不要训练 \(H\rightarrow V\)

仓库当前写的：

> 下一优先级是直接 GT servo 失抓窗口的接触/执行诊断

这个决定是对的。

但我会把实验再明确一点。

### 第一步：直接检查 tick 176 附近

GT servo 是在 tick 176 后第一次失抓。

不要先训练任何东西，直接用现有 packet 对齐：

\[
t=140\sim200
\]

比较 source / reactive teacher / GT servo：

- object pose；
- hand 世界系位置；
- **hand-object relative pose**；
- fingertip-object gap；
- contact force；
- object linear/angular velocity；
- q / PD target；
- 哪根手指出现第一个 contact loss。

尤其应该画：

\[
e_{world}^{hand}
\]

和

\[
\boxed{
e_{rel}^{hand-object}
}
\]

放在一起。

我预计很可能看到：

> world-hand 仍然非常准，但 hand-object relative geometry 已经开始漂。

如果是这样，原因就基本找到了。

---

# 第二步应该做真正的 GT upper bound

不要再用固定 source world hand：

\[
\tau^{hand}_{source,world}.
\]

改成保存 source 中的：

\[
\boxed{
\tau^{hand|obj}_{source}
=
T^{-1}_{obj,source}
T_{hand,source}
}
\]

也就是**手相对物体的 GT trajectory**。

在 fresh rollout 时，根据当前 live object pose：

\[
T_{obj,live}
\]

重新生成目标：

\[
\boxed{
T^{target}_{hand,live}
=
T_{obj,live}
T^{GT}_{hand|obj}
}
\]

这意味着：

> 物体漂 5 mm，目标手也跟着物体漂 5 mm。

而不是逼着手回到旧世界坐标。

这才是我们真正想测试的：

\[
\boxed{
\text{GT interaction-relative hand trajectory}
\rightarrow
\text{control}
}
\]

---

## 而且这个实验甚至暂时不需要 learned retargeter

我们已经有 source 的完整 \(q\)。

可以直接构造一个非常强的 oracle servo：

- finger shape：复用 source finger \(q\)；
- wrist：按照 `live object pose / source object pose` 的 SE(3) 变化，把 source wrist transform 一起搬过去；
- 得到 live desired wrist + source finger configuration；
- 再走现有 native PD controller。

本质上：

\[
T^{target}_{wrist}
=
T_{obj,live}
T^{-1}_{obj,source}
T_{wrist,source}.
\]

这样完全绕过 neural retargeter。

如果这个 **object-relative GT servo** 可以从 held117 恢复到接近 teacher 的 484，例如 >430，那么结论就非常干净：

\[
\boxed{
\text{hand trajectory action 没问题，}
\text{之前错的是 world-frame/fixed-source target。}
}
\]

接下来才值得重新设计 learned retargeter。

---

# 如果这个也失败，结论才更严重

如果：

\[
\text{GT object-relative trajectory}
\rightarrow
\text{exact/analytic control}
\]

仍然抓不住，那么才说明：

单纯

\[
\text{11 hand keypoints}
\]

可能不足以决定抓持。

因为抓持还需要：

\[
\boxed{
\text{geometry}
+
\text{contact force / compliance / velocity}
}
\]

也就是说 PointWAM-style geometry action 对我们 Inspire + Isaac Gym 这套接触敏感任务可能不够，需要把 action representation 扩成：

\[
A^{geom}
=
(\text{hand-object relative trajectory},
\text{desired contact})
\]

而不是单纯 hand position trajectory。

---

## 所以我现在对仓库主线的判断是

之前我们想的是：

\[
H
\rightarrow
\tau^{hand}
\rightarrow
R
\rightarrow
A.
\]

现在应该稍微修正：

\[
\boxed{
H
\rightarrow
\tau^{hand|object}
\rightarrow
R
\rightarrow
A
}
\]

也就是从**绝对手轨迹**改成**手物相对轨迹**。

这其实更符合我们整个 Cm 课题，因为真正有意义的本来就不是：

> 手去了哪里？

而是：

> **手相对于物体怎么运动，物体因此怎么运动？**

后面的 PointWorld 也因此会更自然：

\[
\underbrace{\tau^{hand|obj}}_{\text{candidate action}}
\rightarrow
\underbrace{\tau^{obj}}_{\text{effect}}
\rightarrow
\text{evaluator}.
\]

我目前**不建议重新训练 ref7 的 retargeter**。3 train / 1 val / 1 test 的数据确实太少，但这不是当前最优先的问题。先利用已有 packet 做 tick176 接触诊断，然后做一次 **object-relative GT servo upper bound**。这一步比再加数据、加网络、改 ACT 都更有判别力。