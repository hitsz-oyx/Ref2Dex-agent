看了最新 `main`，HEAD 是 `cd3245e`。现在要分开说：

\[
\boxed{\text{Current BC：已经可以进入训练阶段}}
\]

但

\[
\boxed{\text{Scheduled BC：还差新 }\tau\text{ 数据接线}}
\]

而

\[
\boxed{\text{正式 PPO：还不能直接开长训}}
\]

不是研究路线还有 blocker，而是最后几处训练入口还没切到新合同。

### 1. Current BC：现在可以开始

之前最严重的问题已经解决了。

现在 660 条已经重新用 **DExplore checkpoint teacher 同轮实际 rollout** 回采，不再拿 `reference_obj` 当 task goal。最终筛选得到：

- TRAIN：501 条、54,544 个 action，覆盖 50 个物体；
- DEV：62 条、5,985 个 action，覆盖 42 个物体；
- DOF force、连续 contact force、object state、hand、action、PD、CAD512 都是同轮数据；
- 新 `TaskSpec` 也是从这条 teacher 实际 object/contact trajectory 构造。

所以现在：

\[
(s_t,G_t,Q_t)\rightarrow a_t^{teacher}
\]

在监督语义上已经自洽了。

而且 `TeacherRolloutCorpus` 和 `MeasuredInspirePolicy` 已经写好了：

\[
347_{\text{measured}}
\xrightarrow{+\ shape64}
411_{\text{robot}}
+
10_{\text{task}}
=
\boxed{421D}.
\]

shape encoder 也是可训练的 PointNet，与 actor 联合训练。

**现在缺的只是一个正式的 measured-BC trainer。**

当前仓库里的 `inspire_full_bc.py` 反而明确拒绝新 corpus：

> extended force corpus 必须走 `TeacherRolloutCorpus/MeasuredInspirePolicy421`，旧 319D preparation 会丢 force/shape/phase。

所以这不是“不能训练”，而是作者故意防止你误用旧 trainer。

我认为现在最该做的就是：

\[
\boxed{\text{直接把 Current421 BC trainer 接出来并开始训练}}
\]

不需要再做架构讨论。

---

### 2. Scheduled BC 还不能直接沿用旧缓存

这个必须注意。

旧 scheduled BC 的 \(\tau\) 是基于之前错误的 `reference_obj` goal 生成的，所以仓库现在已经明确禁止静默复用。

新 scheduled BC 应该是：

\[
\boxed{
[robot411,\ task10,\ \tau264,\ valid]
\rightarrow a
}
\]

即 686D。

但这个 \(\tau\) 必须重新按：

\[
\text{teacher rollout TaskSpec}
\rightarrow scheduler
\rightarrow G_{local}
\rightarrow WAM
\]

生成。

而我们现在还有一个选择需要明确：

- **v5 baseline**：不吃 CONTACT/RELEASE；
- **v6 event WAM**：吃我们刚设计的 CONTACT/RELEASE + relative frame。

v6 已经真实训练了 1000 步 continuation，并且迁移/恢复都通过，但现在只是 probe：

- event on DEV hand EPE：33.167 mm
- event off：33.285 mm

这个差异不能说明 event 有净收益。

所以我建议 scheduled BC **不要阻塞 Current BC**：

```text
现在：
Current421 BC 直接训练

并行：
重新生成 teacher-rollout goal 下的 τ cache
├─ v5 τ baseline
└─ v6 event τ candidate
```

然后再做 Scheduled686 BC。

---

### 3. PPO 为什么还不能现在直接开？

PPO 在线物理这边其实已经比我预想的更接近了。

最新 `inspire_physics_host.py` 已经会发：

- `dof_force`
- `contact_forces`
- `dof_force_valid`
- `contact_force_valid`

所以**在线 force 不再是 blocker**。

CAD/object geometry 本来也可以绑定，所以 online `shape64` 也能由同一个 encoder 得到。

真正的问题是当前正式 PPO bridge 仍然是**旧合同**。

`inspire_ppo_bridge.py` 现在仍然在用：

\[
CURRENT\_DIM / INPUT\_DIM
\]

那套旧 observation，并且 reward 仍然是：

```python
shadow_path_reward(...)
```

也就是旧的：

- finger/wrist distance；
- local path error；
- hand-up；
- confirmed progress；
- endpoint bonus。

它还没有切成新写好的：

\[
\boxed{\text{TaskController + TaskReward}}
\]

所以目前如果直接启动旧 PPO 命令，你训练的**不是我们刚才确定的系统**。

---

# 4. 新 `TaskReward` 已经可以用了

它已经实现：

\[
ACQUIRE:
\Phi_A=-d_{\rm palm}-d_{\rm tip}
\]

\[
TRACK:
\Phi_T=progress-tracking\_error
\]

\[
RELEASE:
\Phi_R=
-terminal\_error
-object\_speed
-0.01contact\_force
+hand\_clearance
\]

统一：

\[
r_t=
\gamma\Phi(s_{t+1},Q_{t+1})
-\Phi(s_t,Q_t)
+r_{\rm event}
+r_{\rm terminal}
-r_{\rm time}
-r_{\rm effort}.
\]

代码本身有了，**只是 PPO env 还没调用它**。

---

# 5. 所以我建议现在按这个顺序推进

这次不要再同步改一大堆东西。

**第一步现在就做 Current BC。**

接 `TeacherRolloutCorpus → MeasuredInspirePolicy(with_tau=False)`：

\[
\boxed{421D\rightarrow12D}
\]

保持 Shadow 配方：

- hidden `1024/1024/512/512`
- ELU
- 50k steps
- TRAIN/DEV
- shape encoder 联合训练

先看 offline DEV MAE，再做 closed-loop BC evaluation。

**与此同时重新生成新 TaskSpec 下的 \(\tau\)。** 最好同时保留 v5 和 v6 两份，避免把 event WAM 的有效性和 executor 的有效性绑死。

然后：

\[
Current421
\rightarrow
Scheduled686
\]

按 Shadow 一样，新增 \(\tau\) 列 zero-init，验证 step0 动作逐位保持。

最后才把：

```text
Measured online state
        ↓
InspireAdapter
        ↓
robot411 + task10 + tau
        ↓
PPO
```

以及：

```text
TaskController
      ↓
TaskReward
```

接进官方 PPO engine。

---

## 结论

如果你问“**现在能不能开始工作**”，答案是：

\[
\boxed{\text{可以，而且现在就应该开始新 Current BC。}}
\]

如果问“**现在能不能直接开 BC→PPO 六卡长训**”，答案是：

\[
\boxed{\text{还不行。}}
\]

当前剩余的是明确的工程接线，而不是路线问题：

1. **Current421 BC trainer** —— 现在应该立即接并训练；
2. **重新生成 teacher-goal 对应的 τ** —— 才能做 Scheduled686 BC；
3. **PPO bridge 切到 421/686 + TaskController + TaskReward**；
4. 先做小规模 closed-loop smoke，BC 能至少出现合理 acquire/track 行为后，再开正式 PPO。

现在已经不应该继续设计新架构了。**下一步就是 Current BC。**