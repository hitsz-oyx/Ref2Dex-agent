我重新按代码把昨天那版 `wam-residual-ppo` 审了一遍。结论比昨天记录里的“这个 residual 配方不行”要严重一些：

**昨天的负结果不能干净地归因于 RL 本身。代码里至少有 4 个比较明确的问题，其中两个我认为足以显著影响结果。**

最关键的是这条链：

\[
\text{旧 WAM+R base}
\rightarrow
\text{phase-reset rollout}
\rightarrow
\text{12D residual PPO}
\]

r5 虽然跑完了 200 updates，但训练合同本身有问题。昨天看到的现象是 residual RMS 从约 0.155 增到 0.209，但 hold fraction 一直只有约 0.006–0.010，最终最好也只是 held 5 帧、抬高 93 mm 后掉落。

## 1. 最明确的 bug：rollout 截断时 bootstrap 被硬设成 0

代码现在是：

```python
_logs, rows = episode(ROLLOUT, phases, False, True)

...

result = update(
    actor,
    value_net,
    batch,
    torch.zeros(N, device=task.device),
    actor_optimizer,
    value_optimizer
)
```

而：

```python
ROLLOUT = 96
PHASES = (0, 32, 48, 62)
```

这 96 步都**不是任务终止**。

代码甚至明确规定：

```python
if step < steps - 1 and done.any():
    raise ValueError(...)
```

也就是说 rollout 是人为截断，而不是 terminal。

但是 GAE 里：

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
\]

最后一步却被算成：

\[
\boxed{
\delta_{95}=r_{95}-V(s_{95})
}
\]

因为：

\[
V(s_{96})=0.
\]

这和我们昨天刚讲的 bootstrap 问题完全一样。

正确做法应该是 rollout 结束后再构造真实的：

\[
o_{96}
\]

然后：

\[
\boxed{
bootstrap=V(o_{96})
}
\]

只有真正 `done=True` 才置 0。

这不是调参问题，是 PPO return 定义错了。

---

# 2. 我认为更隐蔽、但可能非常致命：phase reset 后的 H4 是假的

昨天 AI 为了提高接触阶段数据比例，搞了：

```python
PHASES = (0, 32, 48, 62)
```

这个想法本身合理。

但是看 WAM 查询：

```python
def window(rows, env):
    frames = [row[env] for row in rows]

    while len(frames) < 4:
        frames.insert(0, frames[0])

    return np.stack(frames[-4:])
```

每次：

```python
place(phases)
```

以后，`logs` 是全新的。

刚开始只有当前这一帧。

所以例如从 frame 48 reset：

WAM 应该看到：

\[
H=
[s_{45},s_{46},s_{47},s_{48}]
\]

实际看到的是：

\[
\boxed{
H=
[s_{48},s_{48},s_{48},s_{48}]
}
\]

下一步是：

\[
[s_{48},s_{48},s_{48},s_{49}]
\]

再下一步：

\[
[s_{48},s_{48},s_{49},s_{50}]
\]

直到第三四步以后才恢复成真正的 live H4。

这对于我们的 WAM 很严重，因为 H4 里面包含的正是：

- 手的运动趋势；
- object motion；
- velocity / acceleration/history displacement。

也就是说 phase 48、62 这些**最关键的接触阶段 reset**，WAM 前三步反而被喂了一个“物体和手突然静止”的假历史。

---

## 更糟的是 previous action 也同时被置零

代码里：

```python
previous = torch.zeros(N, 18)
```

而 `control_context()` 明确规定输入的是：

> `previous actually applied command`

然后第一步：

```python
previous = logs['applied'][-1] if logs['applied'] else zeros
```

所以从 phase 32/48/62 reset 后：

\[
\boxed{
a_{t-1}=0
}
\]

但实际上这个机器人在 demonstration phase 48 前当然不可能上一拍 action 是全零。

于是 phase-reset 第一拍同时出现：

\[
\boxed{
\text{fake H4}
+
\text{fake previous action}
}
\]

这会一起污染：

1. WAM 的 trajectory；
2. control adapter；
3. residual actor observation。

而昨天所谓 zero-residual gate：

```text
diagnostic_min_gap = 0.42 mm
```

只测了：

```python
episode(..., starts=[0]*N)
```

**它根本没有验证 phase 32/48/62 reset 后 zero residual 的 base 是否还正常。**

这就是明显的实验设计漏洞。

---

# 3. phase reset 后 reward 定义也变了

这个我认为也是比较严重的。

每个 episode 开头：

```python
z0 = current['obj'][:, 2, 3].clone()
```

然后：

```python
lift = current_z - z0
```

held 定义是：

```python
held = near & ~supported & (lift >= 0.03)
```

所以 frame0：

\[
lift=z_t-z_{\text{task start}}
\]

是合理的。

但是从 phase62 reset：

\[
\boxed{
lift=z_t-z_{62}
}
\]

如果 phase62 时物体已经开始抬起来了，那么**保持原来的稳定抓持反而没有 held reward**。

它必须从 phase62 的高度：

\[
\text{再抬 }3\text{ cm}
\]

才能拿：

\[
+1
\]

的 held reward。

这很危险，因为我们最终观察到的失败模式恰好就是：

> 接触 → 抬高 → 抛起来 → 掉落。

所以 phase curriculum 可能正在给 PPO 一个错误信号：

\[
\boxed{
\text{“保持当前高度”不算 held，
继续往上抬才算 held}
}
\]

至少一旦 reset phase 已经处于抬升阶段，这个 reward 就是错的。

---

## `previous_held` 也被清零了

每个 phase reset：

```python
previous_held = torch.zeros(...)
```

如果 reset 的那个状态本来已经在抓持：

正确状态应该是：

\[
previous\_held = True
\]

现在却是 false。

所以如果刚 reset 完立即把东西掉了：

```python
lost = previous_held & ~held
```

不会产生：

\[
-0.5
\]

的 lost penalty。

因此后半段 curriculum 的 reward state 并没有正确恢复。

---

# 4. PPO 的 KL backtracking 有一个真实代码 bug

这个也比较确定。

当前 `residual_ppo.py`：

```python
state = ...
optimizer_state = copy.deepcopy(actor_optimizer.state_dict())

for _attempt in range(4):
    actor_optimizer.step()

    if kl <= target_kl:
        accepted = True
        break

    actor.load_state_dict(state)
    actor_optimizer.load_state_dict(optimizer_state)
    actor_optimizer.param_groups[0]['lr'] *= 0.25
```

看起来它想尝试：

\[
3e{-4}
\rightarrow
7.5e{-5}
\rightarrow
1.875e{-5}
\rightarrow
4.6875e{-6}
\]

但实际上不是。

因为每次失败都：

```python
load_state_dict(optimizer_state)
```

而 `optimizer_state` 一直保存的是最初：

\[
3e-4
\]

所以：

第一次：

\[
3e-4
\]

失败后变：

\[
7.5e-5
\]

第二次用：

\[
7.5e-5
\]

如果失败，又恢复：

\[
3e-4
\]

再乘 0.25：

\[
7.5e-5
\]

于是第三、第四次仍然：

\[
\boxed{7.5e-5}
\]

根本不会继续减。

也就是说所谓 “4 次 backtracking” 实际是：

\[
3e{-4},
7.5e{-5},
7.5e{-5},
7.5e{-5}.
\]

你们普通 `ppo.py` 后来的实现其实已经写对了：

```python
next_lr = current_lr * .25

restore()

lr = next_lr
```

但 `residual_ppo.py` 没同步这个修复。

---

# 5. 还有一个昨天 AI 明明应该记得的问题：\(\lambda=0.95\) 又用回去了

这个特别让我觉得昨天写代码的 AI 没有把仓库历史串起来。

我们 10 月 10 日那轮 PPO 已经做过非常清楚的诊断。

当时长期持有成功样本：

- \(\lambda=0.95\)：7 条里面 **0/7** startup advantage 为正；
- \(\lambda=1\)：变成 **6/7** 为正；
- sampled action probability：
  - λ=.95：1/7 上升；
  - λ=1：5/7 上升。

也就是说我们已经有过明确证据：

\[
\boxed{
\lambda=.95
\text{ 对长时间 grasp credit 太短}
}
\]

结果昨天新写的：

```python
def update(..., gamma=0.99, gae_lambda=0.95)
```

又直接用了：

\[
\lambda=0.95.
\]



---

## 对 30 Hz 控制来说它有多短？

GAE 的有效传播系数：

\[
\gamma\lambda
=
0.99\times0.95
=
0.9405
\]

30 步，也就是大约 1 秒：

\[
0.9405^{30}\approx0.16
\]

60 步，约 2 秒：

\[
0.9405^{60}\approx0.025
\]

所以一个 residual：

> 现在让接触更紧 → 1.5 秒以后因为姿态不对掉落

那个掉落 penalty 传回来已经非常弱。

反而即时的：

```python
contact = +0.2
```

和短暂：

```python
held = +1
```

很强。

这正好容易学成：

\[
\boxed{
\text{短期碰上 / 抬起来}
>
\text{长期稳定}
}
\]

而 r5 的真实失败模式就是这个。

---

# 6. residual action 的概率模型和实际执行 action 不一致

现在：

\[
\delta\sim\mathcal N(\mu,\sigma)
\]

PPO log probability 算：

\[
\log\pi(\delta)
\]

但是环境真正执行：

```python
base + 0.15 * delta
```

然后：

```python
.clamp(-1, 1)
```

也就是：

\[
a=
\operatorname{clip}
(a_{\text{base}}+0.15\delta,-1,1).
\]

于是两个不同的：

\[
\delta_1,\delta_2
\]

可能最终都变成：

\[
a=1.
\]

但是 PPO 仍然认为它们是两个不同概率的动作。

严格来说：

\[
\boxed{
\pi(\delta)
\neq
\pi(a_{\rm executed})
}
\]

尤其 base 本来就在 action boundary 附近的时候会很明显。

---

## 而仓库以前其实已经发现过这个坑

`CmResidual V1.14.3/4` 已经发现：

> residual saturation 集中在 finger DOF 8、14、15，部分 reference phase 的 nominal target margin 已经为 0。

后来专门做了**local feasible residual authority**。

昨天这版却又回到了：

```python
ALPHA = 0.15
```

所有 12D：

\[
\boxed{\text{同一个 global scale}}
\]

然后直接 clamp。

甚至昨天的 reward 都没有 clipping penalty。

而且 r5 没认真报告 residual saturation ratio。

所以我们甚至不知道 PPO 有多少探索动作实际被 clamp 成了同一个 command。

---

# 7. base policy 本身其实不满足一个好的 residual-RL 初始化条件

昨天用的是：

```text
WAM + control adapter + old Chunk-R
```

它之前的完整 542 control rollout 已经明确是：

```text
held = 0
contact occurs
object becomes airborne
then falls
```

而 teacher / GT-R 是四百多帧持有。

也就是说：

\[
\boxed{
\pi_{base}
\text{不是“已经会抓，只是不稳定”}
}
\]

而是：

> 能接触，但接触后的运动模式就是错的。

昨天启动 RL 的唯一 gate 是：

\[
\min gap<30\text{ mm}.
\]

r5：

\[
0.42\text{ mm}
\]

所以通过。

但是：

\[
\boxed{
\text{能碰到}
\neq
\text{附近存在一个小 residual basin 可以抓稳}
}
\]

正确的前置实验应该是：

在**这个 exact WAM+R pair** 上先做局部固定扰动：

\[
\delta_i\in
\{-d,0,+d\}
\]

看看有没有几个方向能从 held0 变成明显更长 hold。

我们之前在另一条 tau executor 上确实找到过：

```text
thumb_pitch_minus
```

的 residual basin。

但那不是昨天这个 exact frozen pair。

所以昨天直接开 PPO，有点跳步骤。

---

# 8. phase curriculum 最大的问题其实可以总结成这一张图

AI 本来想做：

```text
frame 48 的真实物理状态
+ frame45:48 的真实 H4
+ frame47 实际 command
+ 正确 grasp reward state
             ↓
          PPO rollout
```

实际做成了：

```text
frame 48 的 q/object
+ [48,48,48,48] 假 H4
+ previous_action = 0
+ lift baseline = frame48
+ previous_held = false
             ↓
          PPO rollout
```

这已经不是“从 frame48 恢复任务状态”。

只是：

\[
\boxed{
\text{把部分 simulator configuration 搬到了 frame48}
}
\]

但 policy 的 Markov state / reward state 没完整恢复。

这点我认为非常关键。

---

# 9. r5 的 resume 也不是严格 resume

r5 是从 r4：

```text
update0050.pt
```

继续的。

但是 checkpoint 只保存：

```python
actor
value
alpha
update
```

没有：

- actor optimizer state；
- value optimizer state；
- CUDA RNG state；
- phase/reset RNG state。

所以 r5 的 update51：

\[
\boxed{
\text{不是严格意义上的 update50 continuation}
}
\]

Adam moment 全重置。

随机序列也从固定 seed 重新开始。

这个不至于单独导致失败，但这是又一个训练实现不严谨的地方。

---

# 我现在对昨天失败原因的优先级

| 优先级 | 问题 | 判断 |
|---|---|---|
| **S** | phase reset 后 H4 被重复当前帧填充 | **明确实现错误** |
| **S** | 非终止 96-step rollout bootstrap=0 | **明确 PPO 错误** |
| **S** | phase reset 的 lift/held/lost reward 状态没恢复 | **明确 reward-state 错误** |
| **A** | previous actual action 在中途 reset 时被设 0 | **合同错误** |
| **A** | λ=.95，忽略之前已有的 long-credit 负证据 | **很可能直接影响稳定抓持** |
| **A** | raw Gaussian residual 后直接 clamp，logprob 与执行动作不一致 | **PPO 实现不理想** |
| **A** | KL backtracking 实际没有逐级降 LR | **明确代码 bug** |
| B | base 本身 held0，却只用 min-gap gate | 实验设计太弱 |
| B | 12D 统一 alpha，无 joint-wise feasible margin | 已有旧证据提示不合适 |
| C | resume 不保存 optimizer/RNG | 不严谨，但不太可能是主因 |
| C | 393D observation 无 normalization | 值得修，但暂不是首要归因 |

---

# 所以我不会把昨天结果解释成“RL 没法救 WAM”

更准确是：

\[
\boxed{
\text{昨天那轮 residual PPO 本身还没有被实现得足够干净}
}
\]

尤其前四个问题，我认为值得**修完后再做一次 matched rerun**。

而且这次不用马上再跑 200 updates。

我建议先做一个很小的工程验证：

1. 对 phase \(0,32,48,62\)，正确恢复 H4、previous action 和 reward state；
2. zero residual 跑 96 steps，分别看 base trajectory；
3. rollout 末端独立核对：
   \[
   bootstrap=V(s_{96})
   \]
   而不是 0；
4. 把 \(\lambda\) 先改回我们已经有正向机制证据的：
   \[
   \lambda=1;
   \]
5. 记录每个 active DOF 的 residual saturation；
6. 修掉 KL backtracking；
7. 先跑 10–20 updates，看：
   - hold fraction；
   - slip fraction；
   - residual mean/std；
   - value explained variance；
   - phase-specific return。

**如果修完这些，hold fraction 还是死在 1% 左右，再怪 base WAM 或 residual parameterization 才比较公平。**

我现在最想先修的不是 reward 权重，而是 **phase-reset state reconstruction + bootstrap**。这两个属于基础正确性问题，昨天的 AI 确实漏掉了。按照这个修改