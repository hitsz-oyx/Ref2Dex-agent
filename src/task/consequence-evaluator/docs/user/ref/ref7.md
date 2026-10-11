我看了主分支最新状态，当前 `main` 最新提交是 `b880148`，也把 PointWAM 论文的 action/retargeter/执行协议仔细对了一遍。现在我对推进方向的判断和前两轮不一样了：

**不要继续围着当前 `H → 24×18 native action` 这个 ACT-like baseline 修小问题了。PointWAM 给出的更关键启发，是把“手的未来几何轨迹”提升为真正的 action representation，再单独 retarget 到机器人控制。**

### 1. 先说现在 ACT 到底出了什么问题

仓库现在已经基本排除了几个最直观解释。

当前 `NativeActionChunkProposal` 本质上就是：

\[
H_t\rightarrow A_{t:t+23}\in\mathbb R^{24\times18}
\]

1442 维 native observation 先过 MLP，然后一个很小的 2-layer Transformer，一次回归 24 步 native control。训练目标直接是 `pre_physics_step` 捕获到的 18D control。

数据最初只有同一 `s3_airplane_lift / airplane_base` 的 **21 条成功 episode，17 train + 4 val**。离线 MSE 已经从均值基线 `9.38e-4` 做到了 `1.61e-4`，全数据拟合甚至能到 `4.53e-5`。

但行为非常奇怪：

| 执行方式 | held |
|---|---:|
| reactive teacher | ~484 |
| ACT `open_loop24` | **478** |
| `receding8` | **0** |
| `overlap8` | **0** |
| `temporal1` | **0** |

而且最新实验已经把我之前怀疑的 temporal ensemble 问题排掉了：

- receding8 边界 wrist change：33.56 mm
- overlap8：19.23 mm
- temporal1：8.58 mm
- 但抓取仍然全部是 0。

所以现在不能再说：

> “8 步硬切 chunk 导致失败。”

至少它**不是充分解释**。

更重要的一点是：

> `open_loop24` 不是只预测一次然后一路执行。它整个 542 step 中每 24 步还是会重新 query 一次，而它能 held478。

因此也不能简单说：

> “只要 ACT 进入自己产生的状态就 OOD。”

比较准确的说法是：

\[
\boxed{
\text{当前直接 }H\to A\text{ 的闭环映射非常脆弱，}
\text{但原因并不是简单的 chunk 长度或 temporal ensemble。}
}
\]

仓库的 deployment-conditioned 实验确实说明 distribution gap 存在：在真实 deployment history 上重新 fit 后，跨 launch MSE 大约 `1.83~1.90e-4`，原 clean checkpoint 则约 `1.93~2.02e-3`，差了接近一个数量级。但目前这仍只是 **action-space 证据**，没有证明重新训练以后闭环抓取就回来。

---

# 2. PointWAM 实际上不是怎么“修 ACT”的，而是绕开了这个问题

这点非常重要。

PointWAM **没有把核心结构定义成**

\[
H_t\rightarrow \text{joint/native action chunk}.
\]

它是：

\[
(P_t,Q_t,\ell)
\rightarrow
(U_{t,1:T},V_{t,1:T})
\]

其中：

- \(P_t\)：scene point cloud
- \(Q_t\)：当前 hand keypoints
- \(U\)：未来 scene point trajectories
- \(V\)：未来 hand keypoint trajectories

然后再单独做：

\[
\boxed{
(V_{t,1:T},s_t)\rightarrow A_{t,1:T}
}
\]

也就是 **Action Retargeter**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

这和我们现在的 ACT 有本质区别。

我们的：

\[
H\rightarrow A
\]

PointWAM：

\[
H
\rightarrow
\underbrace{V}_{\text{未来手几何}}
\rightarrow
A
\]

而且它的 retargeter **显式读取当前 robot state \(s_t\)**。未来每一个 hand trajectory step 有自己的 embedding，然后 Transformer decoder 一次生成完整 action chunk。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

---

# 3. PointWAM 解决我们这个问题最值得借鉴的，其实有四点

### 第一，action 不首先是关节控制，而是“未来手怎么运动”

PointWAM 明确把 hand point trajectory 称为 **intermediate action representation**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

这个和你之前一直强调的：

> “先不要把 a 看成真实关节角，就用点流代表 a；先解决 planning，再解决 control。”

其实完全一致。

而我们后来又绕回：

\[
\text{native action}
\rightarrow
\text{future hand}
\rightarrow
\text{PointWorld}
\]

于是不得不训练现在这个：

\[
G(H,A)\rightarrow \text{future hand}
\]

结果仓库里的 hand-execution bridge 就出了很典型的问题：

- hand RMSE 比 persistence 好；
- 但 zero-A test 几乎不变化；
- action sensitivity gate 失败。

这不是偶然。

因为我们要求模型学习的是：

> 给我一个低层 native control，然后预测接触动力学下真正会出现什么手轨迹。

这实际上已经掺进了 execution dynamics。

PointWAM反过来：

> **先决定我要的手轨迹是什么，再把它转换成控制。**

这个方向干净很多。

---

### 第二，retargeter 不是预测“执行结果”，而是做“几何目标 → 控制”

PointWAM 的 retargeter 输入是：

\[
(V,s_t)
\]

输出：

\[
A
\]

而不是我们的：

\[
(H,A)\rightarrow V.
\]

而且它训练的是实际 robot demonstrations 中记录的 action。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

换句话说它把问题拆成：

**规划层**

\[
\text{current geometry}
\rightarrow
\text{desired hand geometry}
\]

和

**执行层**

\[
\text{desired hand geometry + current robot state}
\rightarrow
\text{robot control}
\]

这正好可以避免我们现在 ACT 中最别扭的一点：

> 从一个巨大、混合语义的 1442D observation 直接生成 432 个 native control 数。

---

### 第三，它的 action 是相对当前状态定义的

PointWAM 对位置、手指等 action 都相对于当前 robot state 表示，rotation 用 rotation vector，然后**每个 horizon 单独标准化**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

也就是说：

\[
A_{t,\tau}
=
\Delta(s_t,s_{t+\tau})
\]

而不是简单把整个 native actor 输出当作一个统一尺度的 `[−1,1]` 回归目标。

这对我们很值得试。

因为当前 ACT 的：

\[
H_t\rightarrow raw\ native\ control
\]

其实把：

- wrist translation
- wrist rotation
- fingers
- coupling
- PD semantics

全混进同一个 MSE 里。

PointWAM 是显式按 embodiment 和 horizon 处理的。

---

### 第四，PointWAM 根本没有证明“chunk 必须短”

这一点对我们现在很重要。

PointWAM 在 DexJoCo 是 **30-step chunk**，一个 control step 20 ms，所以整段大约：

\[
30\times20\text{ms}=0.6s
\]

而 RoboDojo 也是 30 步，甚至对应约 **1.2 s**，同步 evaluator 会**整个 chunk 执行完才重新 query**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

所以：

\[
\boxed{\text{“8步是不是已经太久了？”——从 PointWAM 看，不是。}}
\]

甚至我们自己的实验也是：

\[
24\text{步 open-loop} \gg 8\text{步 receding}
\]

因此现在不要再花主要精力 sweep `K=4/8/12/24`。

---

# 4. 这其实和我们现在的 PointWorld 接口天然对上了

这是我觉得最值得推进的地方。

你们当前 `POINTWORLD_TEMPORAL_INTERFACE.md` 已经明确把 action 定义成：

> `time=1..24` 的 hand/keypoint trajectory

而且每个 horizon 单独保留 action temporal identity：

```text
scene: time 0
hand action: time 1 ... 24
```

然后：

\[
\text{hand trajectory}_{1:24}
\rightarrow
\text{scene/object flow}_{1:24}.
\]

所以现在的问题其实是：

> **我们的 world model 已经站在 PointWAM 那种 action representation 上了，只有 baseline policy 还停留在 raw native action 空间。**

这才导致中间不得不插：

\[
A
\rightarrow
\hat V
\rightarrow
PW
\]

这个很难学的 `execution bridge`。

如果改成：

\[
\boxed{
H\rightarrow V
}
\]

那么 PointWorld 可以直接吃 \(V\)。

---

# 5. 我建议现在把整条链改成这样

不是完全照抄 PointWAM，而是借它最适合我们问题的一层设计。

## 第一层：Point-trajectory baseline

先训练：

\[
\boxed{
\pi_V:
H_t
\rightarrow
V_{t,1:24}
}
\]

其中 \(V\) 就用你们已经统一过的 **11 个 Inspire hand keypoints × 24 步**。

监督数据根本不用重新 fork。

已有成功 Gym rollout 本来就保存了真实 hand geometry。

所以每个时刻都有：

\[
(H_t,V^*_{t,1:24})
\]

这一步甚至可以先只做 airplane。

---

## 第二层：retargeter

单独训练：

\[
\boxed{
R(V_{t,1:24},s_t)
\rightarrow
A_{t,1:24}
}
\]

而不是现在的：

\[
H_t\rightarrow A_{t,1:24}.
\]

标签仍然直接来自现有 native successful rollout。

这里我建议基本照 PointWAM：

- input：future hand keypoints + current robot state；
- 每个 horizon 一个 step embedding；
- 2-layer Transformer decoder 就够；
- action target 相对当前 state 表示；
- wrist position / rotation / finger 分开 normalization；
- 每个 horizon 独立 statistics；
- 先用 L1，不必执着 MSE。

这个模型不会特别大。

PointWAM 自己的 retargeter只有大约 4.6M 参数；大模型主要在前面的 world/trajectory backbone。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

---

# 6. 但我不建议一上来就训练完整的 \(H\rightarrow V\)

按我们之前一直强调的“先上界再实现”，第一步应该更简单。

直接做：

\[
\boxed{
V_{\rm GT}\rightarrow R\rightarrow A
}
\]

也就是：

**把成功 expert rollout 的真实 future hand trajectory 直接喂给 retargeter。**

然后运行真实 Gym。

如果：

\[
V_{\rm GT}
\rightarrow A
\rightarrow \text{robot}
\]

都不能稳定复现 expert hand trajectory / grasp，

那说明问题在：

\[
\text{retarget/control}
\]

我们根本没必要训练 policy。

反过来，如果它可以：

例如做到：

- GT-hand retarget：held ≥ teacher 的 90%
- hand trajectory execution RMSE 明显小于当前 6cm 级 bridge；
- open-loop 24 能稳定抓取；

那么我们就证明：

\[
\boxed{
\text{hand-point trajectory 是一个可执行的 action space}
}
\]

这个证据的价值非常高。

然后才训练：

\[
H\rightarrow V.
\]

---

# 7. 再下一步，PointWorld 就非常自然了

baseline：

\[
H
\rightarrow
V_0
\rightarrow
R
\rightarrow
A
\]

加入 Cm / PointWorld 后：

\[
H
\rightarrow
\{V_1,\ldots,V_K\}
\]

对于每个候选：

\[
(P_t,V_i)
\xrightarrow{\text{PointWorld}}
U_i
\]

然后：

\[
Q(H,V_i,U_i)
\rightarrow
S_i
\]

选：

\[
V^*=\arg\max_i S_i
\]

最后才：

\[
R(V^*,s_t)
\rightarrow
A^*.
\]

所以整个候选规划都发生在：

\[
\boxed{\text{hand point trajectory space}}
\]

而不是 native action space。

这比现在：

\[
A_i
\rightarrow
G(H,A_i)
\rightarrow
\hat V_i
\rightarrow
PW
\]

少了一整个最不稳定的环节。

---

# 8. 这还正好解决我们之前“candidate 怎么加”的问题

现在你们是在 native action 上做：

\[
a_0+\delta a_i.
\]

这很难解释，因为一个很小的 joint residual 到底在空间里对应什么 hand interaction，并不清晰。

改成 trajectory candidate 后，可以直接：

\[
V_i=V_0+\Delta V_i.
\]

比如：

- wrist translation；
- wrist rotation；
- thumb/finger closing trajectory；
- grip width / fingertip approach；
- 整个 hand shape 的低维 basis。

候选的意义直接是：

> “未来 0.8 秒手往哪里走。”

这才真正和 PointWorld 的：

\[
\text{hand motion}\rightarrow\text{object motion}
\]

因果接口匹配。

---

# 9. PointWAM还有一个结果支持我们这样拆

PointWAM 本身做过 world-model ablation：

保持 3D input、hand trajectory head 等不变，**去掉 scene trajectory supervision**，成功率从：

\[
64.4\%\rightarrow53.5\%.
\]

也就是 world modeling 单独带来大约 **+10.9pp**。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

这个结果的意义很符合我们想证明的东西：

先有一个能工作的：

\[
H\rightarrow V\rightarrow A
\]

baseline。

然后再问：

\[
\boxed{
\text{预测 scene consequence 是否能让这个 baseline 更好？}
}
\]

而不是现在 baseline 自己都不稳定，就开始判断 PointWorld 有没有价值。

这才是一个非常干净的论文链条。

---

# 10. 因此我会调整目前的推进优先级

我现在**不建议**继续做：

- 再调 ACT temporal ensemble；
- 再试 `K=4/8/16`；
- 再给当前 H→A 模型堆 epoch；
- 再扩大 `G(H,A)→hand` 的数据；
- 现在就接 candidate evaluator。

我建议按下面顺序推进：

1. **GT hand trajectory → native action retargeter**  
   先验证真实手轨迹是否能稳定转成 Inspire 控制。

2. **真实执行 hand tracking 验证**  
   不只看 action MSE，要真正比较执行后的 11 keypoint trajectory。

3. **H → future hand trajectory baseline**  
   先只做 airplane，复用成功 Gym rollout。

4. **H → V → retargeter 的完整 baseline**  
   先达到能稳定抓起来。

5. **再接现在已经训练好的 PointWorld**  
   让候选直接是 \(V_i\)，彻底删除 `A → predicted hand trajectory` 这层。

6. **最后做 PointWorld-on/off matched comparison**。

---

所以如果让我现在给主代理下一个任务，我不会写“继续修 ACT”。

我会改成：

\[
\boxed{
\textbf{把 ACT baseline 重构成 PointWAM-style hand-trajectory policy + action retargeter}
}
\]

但第一步**只做 GT-hand retargeter upper-bound probe**，不要立刻训练完整模型。

PointWAM真正值得我们学的不是它的模型规模，而是这个接口：

\[
\boxed{
\text{geometry planning}
\rightarrow
\text{hand trajectory}
\rightarrow
\text{robot control}
}
\]

它和我们现在的 PointWorld 接口几乎正好拼上。PointWAM 本身也是先共同预测 scene/hand trajectory，再把 hand trajectory retarget 成 action，而不是让 world model围着 raw joint action 转。[Anonymous View](https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjEwLjAyODQwdjE)

如果沿这个方向走，我认为我们前面那个 **6 cm execution predictor** 基本可以直接降级为历史诊断，不需要再把它当主线去救。