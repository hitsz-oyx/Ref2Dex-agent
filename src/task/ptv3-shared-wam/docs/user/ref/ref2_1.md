对，**现在 R 还有问题，但我认为原因已经不是“PointWAM 能做，我们数据不够”这么简单了。** 我重新按 PointWAM 论文把 retargeter 合同逐项对齐后，发现一个更根本的差异：

\[
\boxed{\text{我们的 R 目前其实没有在学 PointWAM 那个 retargeting 问题。}}
\]

我们在学的是一个明显更难的 **低层闭环控制命令蒸馏问题**。

### 最关键的区别：action label 根本不是一类东西

PointWAM 的 robot state 是：

\[
s_t=
[\text{EEF pose},\ \text{finger joint angles}]
\]

它的 action 是：

\[
[\text{EEF xyz},\ \text{rotation},\ \text{finger joint targets}]
\]

然后他们**不会直接回归原始 action**，而是把整个未来 action chunk 全部改写成相对于当前状态 \(s_t\) 的几何量：

\[
\Delta p_{t+k}=p^{des}_{t+k}-p_t
\]

\[
\Delta q^{finger}_{t+k}=q^{des}_{t+k}-q_t
\]

orientation 用当前姿态到目标姿态的 rotation vector。最后每个 horizon、每个 action coordinate 再单独标准化。[arXiv](https://arxiv.org/abs/2610.02840)

也就是说 PointWAM 实际学的是：

\[
\boxed{
(\text{future hand geometry},s_t)
\rightarrow
\text{future desired robot geometry}
}
\]

这个映射确实非常接近你说的“确定性 retargeting”。

---

而我们现在 `ChunkRetargeter` 的 label 是：

```text
applied[t:t+7]
```

即 **DExplore 真正送给 native controller 的 normalized command**。

这就完全不一样了。

尤其 wrist 前 6 维，我们的 native command 在每一个时刻实际上都是相对于**那个时刻的实时 q**解释的：

\[
a_{t+k}
\rightarrow
q^{PD}_{t+k}
=
q_{t+k}
+
offset
+
scale\cdot a_{t+k}.
\]

所以当我们在 \(t\) 时刻要求 R 一次性预测：

\[
a_t,a_{t+1},...,a_{t+7}
\]

实际上未来第 \(k\) 个 label 的坐标基准是：

\[
q_{t+k},
\]

而不是：

\[
q_t.
\]

于是：

\[
\boxed{
(s_t,\tau)\rightarrow a_{t+k}
}
\]

隐含要求模型自己知道：

\[
q_{t+k}.
\]

这已经不是简单 retargeting 了，而接近：

\[
\boxed{\text{inverse dynamics + future closed-loop policy imitation}}
\]

---

# 这就是我们和 PointWAM 最大的区别

| | PointWAM | 我们当前 R |
|---|---|---|
| 输入 hand | future hand trajectory | future hand trajectory |
| state | EEF pose + finger q | q/dq + hand + object pose/vel |
| 输出 | **几何 action target** | **native low-level command** |
| wrist target | 相对当前 \(s_t\) | 相对每个未来 \(q_{t+k}\) 的 command |
| fingers | 相对当前 finger q | native absolute/coupled command |
| action normalization | per horizon/coordinate | per horizon/coordinate |
| 本质 | kinematic retargeting | policy/control inversion |

所以表面上都是：

\[
g(V,s)\rightarrow A
\]

但条件熵完全不一样。

---

## 还有一个非常重要的点：PointWAM 的 retargeter 并没有靠 1.15M 视频数据预训练

这点可以排除你刚才的疑问。

论文明确写：

> trajectory forecaster 用 human + robot；**action retargeter 只在 robot demonstrations 上训练，而且从头训练。** [arXiv](https://arxiv.org/abs/2610.02840)


PointWAM fine-tuning 机器人数据是：

\[
959\text{ demonstrations}
\]

我们：

\[
660\text{ trajectories}
\]

量级确实差不多。

所以：

\[
\boxed{\text{不能再解释成“PointWAM 有很多视频，所以 R 才学得好”。}}
\]

视频主要救 forecaster，不直接救 retargeter。

你这个质疑是成立的。

---

# 数据“数量相近”，但 action supervision 的干净程度并不相近

PointWAM 在 DexJoCo 中先 replay demonstrations，原 1000 条里保留了 **959 条成功 replay**。[arXiv](https://arxiv.org/abs/2610.02840)

这些 demonstration 的 action 本身就是：

\[
\text{EEF target}+\text{finger target}
\]

也就是和 hand trajectory 有直接几何关系。

我们 RL660 的来源则是：

\[
\text{DExplore closed-loop RL actor rollout}.
\]

我们虽然用 10 cm tracking gate 去掉明显失跟窗口，但没有要求：

\[
\text{每条都是成功稳定抓取 demonstration}.
\]

更重要的是，DExplore 的 command 是**策略输出**。

策略可以根据：

- 当前速度；
- tracking error；
- reference phase；
- 接触响应；
- 当前 state；
- 其他 policy observation

输出不同 command。

但是我们的 R 只拿：

\[
state87+\tau.
\]

也就是说我们实际在做：

\[
\pi_{\rm DExplore}(o_t)
\]

的蒸馏，但：

\[
o_t\not\subseteq(state87,\tau)
\]

是有可能的。

这会产生真正的 conditional ambiguity。

PointWAM 则没有这么严重：

\[
\text{desired hand trajectory}
\rightarrow
\text{desired joint/EEF target}.
\]

---

# 还有一个证据：我们自己以前的 PDRetargeter 其实已经接近 PointWAM 了

这个很重要。

我们之前的 `PDRetargeter` 不是预测 raw native command。

它预测：

\[
\boxed{\text{actual PD target}}
\]

并且 wrist target 相对当前 q 编码。

当时 `competent-pd-retargeter`：

\[
GT\tau\rightarrow PD
\]

validation XYZ p95 大约：

\[
\boxed{12.5\text{ mm}}
\]

而 τ model 相比 state-only normalized MSE：

\[
0.0248\rightarrow0.0102
\]

约降低：

\[
59\%.
\]

shuffle τ 后又恶化约 5.5 倍。

后面的 physical Decision 里，GT-\(\tau\)+这个 R 甚至能保持抓取。

这已经非常强烈地暗示：

\[
\boxed{\text{hand trajectory}\rightarrow\text{physical target 是可学的}}
\]

真正把事情变难的是后来我们为了模仿 PointWAM “action chunk”，把 target 换成了：

\[
\text{raw native applied action chunk}.
\]

问题是 PointWAM 的 action chunk **不是 raw low-level actuator command chunk**。

---

# PointWAM 其实也是 all-chunk loss

这也修正一下我们前面的判断。

PointWAM 明确写：

\[
L_A
=
\frac{1}{T d_a}
\sum |\hat A-A^*|
\]

也就是所有 horizon/action coordinates 平均，所有 loss coefficient 都是 1。[arXiv](https://arxiv.org/abs/2610.02840)

所以：

> “我们 all8 等权，而 PointWAM 强调 first action”

**不是根本区别。**

PointWAM 之所以 all-chunk 能学：

因为它整个 chunk：

\[
A_{t,1:T}
\]

都在**统一的当前 \(s_t\) 相对坐标合同里**。

而我们未来 native command：

\[
a_{t+k}
\]

是 future closed-loop state-dependent 的。

这才是核心。

---

# 而且 PointWAM 真的会执行 chunk，不是只用 a0

PointWAM DexJoCo：

\[
T=30
\]

控制频率约 50 Hz，一个 chunk 覆盖：

\[
0.6s.
\]

evaluator 在 buffer 剩余不到 24 时才请求新 chunk，也就是说一个预测至少实际往前消费若干步，而不是每 tick 丢掉 29 个动作。[arXiv](https://arxiv.org/abs/2610.02840)

所以它训练 whole chunk 也是 deployment-aligned 的。

我们目前：

\[
T_R=8
\]

launcher 默认：

\[
K_{\rm exec}=4.
\]

这个其实比 K1 更接近 PointWAM。

因此如果我们要真正复刻 PointWAM 思路，方向不应该继续变成：

> “给 first action 更大权重。”

那只是暂时修症状。

---

# 我现在认为真正应该改的是 R 的输出定义

不要再预测：

\[
\boxed{\text{native normalized command}}
\]

而改成 PointWAM 式的：

\[
\boxed{
\text{desired physical target chunk}
}
\]

例如对 Inspire：

\[
y_{t,k}
=
[
\Delta p^{wrist}_{k},
\Delta R^{wrist}_{k},
\Delta q^{finger}_{k}
].
\]

其中全部相对于**当前 \(t\)**：

\[
\Delta p_k
=
p^{des}_{t+k}-p_t
\]

\[
\Delta R_k
=
\log(R_t^\top R^{des}_{t+k})
\]

\[
\Delta q^{finger}_k
=
q^{des}_{t+k}-q_t.
\]

然后：

\[
\mu_{k,j},\sigma_{k,j}
\]

TRAIN-only per-horizon normalization。

这才真正对应 PointWAM 的 action representation。[arXiv](https://arxiv.org/abs/2610.02840)

---

## 执行时再把 physical target 转成 native command

也就是：

```text
R output
    ↓
desired wrist pose / finger joint target
    ↓
live q_t
    ↓
native PD/action conversion
    ↓
Isaac
```

而不是让 R 自己学习：

```text
hand trajectory
    ↓
native controller internal encoding
```

这可以把：

\[
\text{geometry reasoning}
\]

和：

\[
\text{controller representation}
\]

彻底拆开。

---

# 最关键的一次实验其实很便宜

我不会立刻重训 WAM。

直接使用现有 RL660，同一个：

\[
state87,\ GT\tau
\]

做三个 R：

1. 当前：
\[
R_{\rm native}:\tau,s\rightarrow a^{native}_{1:8}
\]

2. PointWAM-like：
\[
R_{\rm PD}:\tau,s\rightarrow PDTarget_{1:8}^{relative-to-s_t}
\]

3. 更严格 PointWAM：
\[
R_{\rm geom}:\tau,s\rightarrow
[\text{wrist pose delta},\text{finger q delta}]_{1:8}
\]

完全相同：

- TRAIN/dev；
- sampler；
- 模型大小；
- steps；
- normalization；
- seed。

只换 label representation。

然后比较**恢复到最终实际 PD target 后**的误差。

---

## 我现在的预期很明确

如果：

\[
R_{\rm native}
\approx40\text{ mm}
\]

而：

\[
R_{\rm PD/geom}
<10\sim15\text{ mm},
\]

那基本就坐实：

\[
\boxed{\text{我们之前不是“R 学不好”，而是 R target 定义错了。}}
\]

而且这和我们旧 `PDRetargeter` 的结果完全一致。

---

# 还有两个次要区别

PointWAM retargeter 本身也比我们的 R 大。

论文报告：

\[
\boxed{4.6M\ parameters}
\]

而且 fine-tune：

\[
60k\ iterations,\quad batch=192,\quad lr=10^{-4}.
\] :chatgpt-content-reference{index="6"}


我们当前是 128D、2 decoder blocks 的小 R，联合大训练也只有约 20k optimizer steps。

但我现在**不认为这是第一原因**。

因为我们自己的 256-window experiment 已经能：

\[
65.9\rightarrow8.6\text{ mm}.
\]

说明至少局部 capacity 足够。

先修 target，再看是否需要把 R 扩成 4–5M。

---

# 另外 PointWAM 的 hand keypoints 设计也更“控制对齐”

它不是随便选 10 个手点。

Allegro 的 10 点是：

- 6 个 palm rigid offsets；
- 4 个 fingertip **contact pad** points。

论文甚至明确说不用 distal joint-axis point，因为那种点反映不了 finger flexion。[arXiv](https://arxiv.org/abs/2610.02840)

我们现在的 11 native links：

- hand base；
- proximal links；
- tips；

在语义上合理，但不一定是对 12 active controls 最可逆的一组点。

这是下一层问题，但也不是当前第一优先级。

---

## 所以我现在对问题的判断变了

之前我们在问：

> 为什么 PointWAM R 能学，我们 R 学不好？

现在答案比较清晰：

\[
\boxed{\text{因为我们实际上给 R 定义了一个比 PointWAM 困难得多的问题。}}
\]

PointWAM：

\[
\text{hand trajectory}
\rightarrow
\text{relative geometric robot target}
\]

我们：

\[
\text{realized hand trajectory}
\rightarrow
\text{RL policy's low-level future native commands}.
\]

后者混入了：

- future feedback；
- servo representation；
- policy-specific corrections；
- contact/preload；
- future q dependence。

**数据集数量差不多并不能消除这个差异。**

所以我下一步不会继续围着当前 raw-native Chunk-R 调 loss 了。

我会直接把 R 的 target 改回 **PointWAM-equivalent physical target representation**，并用 RL660 做一个 matched label-representation experiment。

这一步如果做对，我们才算真正和 PointWAM 在同一个 retargeter 问题上比较。