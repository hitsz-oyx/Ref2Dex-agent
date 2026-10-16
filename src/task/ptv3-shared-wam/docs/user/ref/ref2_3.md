有，而且这点现在可以说得更明确：

\[
\boxed{\text{DExplore 里本来就有一条 reference robot motion}}
\]

它不是只有“人手轨迹”。`base_dexplore_task.py` 每个时刻都会从 `hoi_data` 里读当前参考和未来参考：

```python
self._curr_ref_obs = self.hoi_data[self.data_id[env_ids], ts]
ref_obs = self.hoi_data[self.data_id[env_ids], next_ts]
```

而这个 reference 里明确包含：

\[
\boxed{q^{ref}}
\]

因为 reward 里直接取：

```python
ref_dof_pos = hoi_ref[:, 392:392+self.num_dof]
```

同时还有：

- 物体参考位置/姿态 `ref_obj_pos/ref_obj_rot`
- 手部关键 body 的参考位置、姿态、速度
- reference contact
- object velocity 等。

所以 DExplore 本身的结构其实就是：

\[
\boxed{
\text{reference robot trajectory}
\rightarrow
\text{RL tracking policy}
\rightarrow
\text{PD target}
\rightarrow
\text{real robot state}
}
\]

这正是我们刚才想找的东西。

---

但要特别区分三个量：

\[
q^{ref}_t
\]

是**参考轨迹希望机器人处于的位置**；

\[
q^{PD}_t
\]

是 DExplore actor 根据当前误差、contact、future reference 等算出来的**反馈控制 setpoint**；

\[
q^{real}_t
\]

是仿真最后真正走到的位置。

我们之前的 R 学的是第二个：

\[
\boxed{
R(s,\tau)\rightarrow q^{PD}
}
\]

所以才是在做 policy inversion。

而我现在说应该优先检查的，其实是第一个：

\[
\boxed{
R_{\rm kin}(\tau,s)
\rightarrow q^{ref}
}
\]

这才是真正接近 PointWAM 的 retargeting。

---

### 为什么之前 GT realized-q 实验不能回答这个问题？

因为刚才我们测的是：

\[
q^{real}_{t+1:t+8}
\]

拿来当 controller target。

它不是：

\[
q^{ref}_{t+1:t+8}.
\]

所以“realized q replay 需要 velocity compensation”并不意味着 **DExplore reference 本身不适合作为动作规划目标**。

这是两件不同的事。

---

### DExplore actor 为什么还需要存在？

因为 reference 只是告诉它：

> “手应该走成什么样。”

actor 则处理：

> “现在实际偏了多少、是不是接触了、应该多压一点还是少压一点。”

代码里它确实会读未来 reference，并结合当前实际状态和接触信息生成 action。

所以最自然的拆分就是：

\[
\boxed{
\tau_{\rm WAM}
\rightarrow
R_{\rm kin}
\rightarrow
q^{ref}
}
\]

然后：

\[
\boxed{
(q^{ref},s_t,\text{contact})
\rightarrow
\text{controller / residual RL}
}
\]

---

## 现在最应该做的其实很简单

**先不要训练任何新 R。**

直接从 RL660 对应的原始 DExplore motion 中，把：

\[
q^{ref}_{t:t+K}
\]

以及对应 reference wrist/body trajectory 提出来。

然后做一个新的 oracle replay：

\[
\boxed{\text{reference-q replay}}
\]

和刚才三个东西并排：

1. 原 teacher PD；
2. realized-q；
3. **reference-q**。

看同样的：

- hand EPE；
- wrist EPE；
- object trajectory；
- held frames。

如果 `reference-q` 能比较好地复现机器人运动，那我们就找到了真正应该训练的 R 标签：

\[
\boxed{
\text{WAM hand trajectory}
\rightarrow
\text{DExplore reference robot trajectory}
}
\]

而不是继续拟合 actor 的 PD 输出。

所以答案是：**有，而且很可能我们真正想要的 target 一直就在 DExplore 的 reference motion 里，只是之前把 actor 后面的 command 当成了 R 的监督。**