我把这条链仔细对了一遍。结论很明确：

**目前 54 mm 这么大的 GT-\(\tau\)→R 偏差，首要原因不是坐标、时序或 PD 换算 bug，而是我们当前 R 的训练目标又退回到了一个已经被仓库历史证明不适合“每步重规划、只执行第一步”的 objective。**

而且这不是小问题，基本可以解释为什么“理论上接近确定的第一步映射”却学得这么差。

### 先澄清 54 mm 到底是什么

当前 dev 里的 `GT/PD_xyz_mm` 是这样算的：

\[
\Delta a_0
=
\operatorname{clip}(R(s_t,\tau^{GT})_0)-a_t
\]

然后：

\[
E_{xyz}
=
\left\|
\Delta a_{0,xyz}
\odot s^{PD}_{xyz}
\right\|_2.
\]

所以 54 mm 是：

> **第一步 wrist XYZ 的 PD target 偏差**

不是 24 步 hand trajectory EPE，也不是 FK 后的实际手腕位置误差。

这条 metric 本身没算错。当前代码的 action→PD scale、finger `/2`、时钟 `action[t] -> state[t+1]` 都对得上。

---

## 真正最严重的问题：R 学的是 8 步 closed-loop action，但我们只执行第 1 步

现在 `robot_loss()`：

\[
L_{\text{anchor}}
=
\operatorname{mean}
\left|
\frac{
R(s_t,\tau^{GT})-
a_{t:t+7}
}{
\sigma_a
}
\right|.
\]

也就是说 R 输入只有：

\[
s_t,\tau_{t+1:t+24}
\]

却要一次预测：

\[
a_t,a_{t+1},...,a_{t+7}.
\]

这里有个根本问题。

### \(a_t\) 接近你说的“确定性映射”

在给定：

\[
q_t,dq_t,\tau^{GT}
\]

的情况下，

\[
a_t
\]

确实应该非常可学。

但：

\[
a_{t+1}
\]

真正是 teacher 在看到：

\[
s_{t+1}
\]

之后产生的。

而 \(s_{t+1}\) 包含：

- 实际接触响应；
- q/dq；
- object response；
- PhysX contact state；
- 前一步控制产生的误差。

这些在时刻 \(t\) 都没有完整给 R。

因此：

\[
\boxed{
(s_t,\tau)\rightarrow a_{t+1:t+7}
}
\]

并不是严格确定的。

更准确地说：

\[
a_{t+k}
=
\pi_{\text{teacher}}(s_{t+k},reference_{t+k})
\]

而当前 R 在尝试：

\[
a_{t+k}
\approx
R_k(s_t,\tau).
\]

这相当于让一个**开环 chunk decoder 模仿未来 closed-loop controller**。

但我们实际部署又是：

> 每一个 tick 重新读状态、重新预测，只执行 \(a_0\)。

所以训练目标和部署方式是不匹配的。

---

## 更离谱的是：真正要执行的第一步只占 loss 的 1/8

一个 action chunk：

\[
8\times12=96
\]

个标量。

当前 action loss 是全部平均。

所以第一步所有 12 DOF 只占：

\[
\frac{12}{96}=12.5\%.
\]

我们最关心的 wrist XYZ 更只占：

\[
\frac{3}{96}
=
\boxed{3.125\%}.
\]

而总 loss 里面还有：

\[
L=
L_{hand}
+
L_{scene}
+
L_{action}
+
L_{anchor}.
\]

所以 optimizer 根本没有在强力优化：

\[
\boxed{\text{第一步 wrist PD XYZ}}
\]

这个最终执行指标。

这就是为什么你现在看到：

训练 loss：

\[
4.73\rightarrow1.80
\]

降得很好，

但 GT-\(\tau\) 第一控制还剩：

\[
54\text{ mm}.
\]

不是矛盾。

模型大量精力花在了：

- hand forecasting；
- scene forecasting；
- action 2–8；
- finger；
- rotation；

而不是我们真正 execute 的第一步 XYZ。

---

# 更关键的是：我们之前已经做过这个实验

10 月 12 日那个：

`P-20261012-goal-point-wam-ref25-control-aware-R`

其实就是专门修这个问题的。

旧 Chunk-R 用的就是现在类似的：

\[
L_{\text{old}}
=
\text{mean all8 standardized L1}
\]

结果：

> GT-\(\tau\) physics：**held 0**

第一步 PD XYZ p95 大约：

\[
31.09\text{ mm}.
\]

后来改成：

\[
\boxed{
L
=
4L_{\text{first}}
+
L_{\text{tail}}
+
L_{\text{PD0}}
}
\]

其中：

\[
L_{\text{PD0}}
=
\frac13
\left(
\frac{E_{xyz}}{10\text{ mm}}
+
\frac{E_{rot}}{0.1}
+
\frac{E_{finger}}{0.1}
\right).
\]

并且 checkpoint 不再按 all8 L1 选，而是优先：

\[
\boxed{\text{first PD XYZ p95}}
\]

结果直接变成：

\[
31.09
\rightarrow
19.51\text{ mm}
\]

实际 GT-\(\tau\) 执行：

- teacher：held 73
- old qualified R：75
- 新 control-aware Chunk-R：**74**

即：

\[
\boxed{\text{GT execution gate PASS}}
\]

所以这不是我现在猜出来的。

**仓库已经实验证明：all8 等权 objective 会把 Chunk-R 训练坏；改成 first-action/PD-aware 后会显著改善。**

但现在 RL660 又把这个修复丢掉了。

---

# 第二个明确问题：checkpoint selection 也选错目标了

现在 robot finetune 用：

\[
\text{total dev objective}
\]

选 `best.pt`。

所以当前：

\[
best = step4500.
\]

但是看真正的控制指标：

| step | GT-\(\tau\) PD XYZ |
|---|---:|
| 4500 | 61.75 mm |
| 9000 | **54.39 mm** |

predicted-\(\tau\) 也是：

\[
63.88
\rightarrow
57.27\text{ mm}.
\]

也就是说：

\[
\boxed{
\text{step9000 的 R 明显比 step4500 更适合执行}
}
\]

但因为 hand/scene dev 变差：

\[
4.7704\rightarrow4.9944
\]

系统把 step4500 叫做 “best”。

所以如果训练结束后直接按仓库现在写的：

> “拿 dev best 做 native execution”

很可能会拿错 checkpoint。

这已经是现成数据证明的。

应该至少分成两个 best：

\[
\boxed{
C_{\text{forecast-best}}
}
\]

按 hand/scene physical EPE；

以及：

\[
\boxed{
C_{\text{control-best}}
}
\]

按：

\[
GT\tau\rightarrow R
\]

第一步 PD error。

不能共用一个 total objective best。

---

# 第三个问题：50 物体以后，R 的输入其实不完整

当前 R 的 state87：

\[
[q,dq,\ hand_{local},\ object\ pose,\ object\ velocity]
\]

再加：

\[
\tau_{24\times33}.
\]

但它**不知道物体是什么**。

没有：

- object geometry；
- object size；
- mass/inertia；
- contact state；
- contact force；
- previous action；
- teacher reference/phase。

在单 airplane 时问题没那么明显。

现在是：

\[
50\text{ 类物体}
\]

一起训练。

如果这是纯 kinematic inverse：

\[
\tau\rightarrow q^{des}
\]

物体身份确实不重要。

但我们监督的不是纯 IK。

监督的是：

\[
\boxed{\text{official DExplore actor 实际输出的 command}}
\]

它是在接触动力学下产生的。

例如同样一条手轨迹：

\[
\tau
\]

抓一个重物和轻物，teacher 可能输出不同的 PD target/preload。

可是 R 看不到这个区别。

所以：

\[
p(a|s,\tau)
\]

在 multi-object 数据上会比单物体更宽。

---

# 但我不会先加 object token

因为我们现在还有一个更基础的问题没修：

\[
\boxed{\text{objective 就错位了}}
\]

先修这个。

如果修完 first-action objective 后：

\[
GT\tau\rightarrow a_t
\]

仍然很差，再讨论 object geometry/contact conditioning。

---

## 第四个问题：当前 Chunk-R 还是一个很小的 128D 模型

现在 WAM 已经扩成：

\[
101.6M
\]

但 R 仍然是：

- width 128
- 2 decoder blocks
- 4 heads

也就是一个非常小的网络。

不过我暂时不把 capacity 排第一。

因为旧实验里同样 128D Chunk-R：

\[
\text{all8 loss}\rightarrow31\text{ mm}
\]

换 objective：

\[
\rightarrow19.5\text{ mm}.
\]

说明 objective 的收益比扩模型更加直接。

---

# 还有一个非常有价值的历史证据

我们以前那个 `competent-pd-retargeter` 不预测 8 步 chunk，而是直接做：

\[
\boxed{
R(s_t,\tau)\rightarrow PD_t
}
\]

只预测第一控制。

结果 validation：

\[
XYZ\ p95=12.5\text{ mm}
\]

而且：

\[
L_{\tau}=0.0102
\]

对比 state-only：

\[
0.0248
\]

下降：

\[
59\%.
\]

shuffle \(\tau\) 后：

\[
0.0563
\]

恶化 5.5 倍。

所以我们其实已经证明：

\[
\boxed{
\tau\text{ 对第一步控制是高度有信息的}
}
\]

你“这应该接近确定性映射”的直觉并没有错。

只是当前这个 R **没有按照第一步 inverse controller 的方式去训练**。

---

# 我认为最便宜、最干净的 debug 现在应该这样做

不要停当前 19k 训练，它可以继续跑。

另外单独做一个 **R-only matched diagnostic**，完全不动 WAM。

固定现有 RL660 TRAIN/dev，直接给：

\[
\tau^{GT}.
\]

比较三个模型：

| 模型 | 输出 | loss |
|---|---|---|
| A 当前 | 8×12 | all8 L1 |
| B | 8×12 | **4×first + tail + PD0** |
| C | **1×12** | **只预测 first PD/native action** |

尤其 C。

它实际上在测试：

\[
\boxed{
(s_t,\tau^{GT})
\rightarrow
a_t
}
\]

到底有没有你预期的确定性。

---

## 我会设置一个非常明确的判断

先看 TRAIN，不要先看 dev。

如果 C 在 TRAIN 都只能：

\[
30\sim50\text{ mm}
\]

那说明：

\[
(s,\tau)
\]

确实不足以唯一恢复 teacher command，接下来查：

- object identity；
- contact；
- previous action；
- reference；
- hidden state。

如果 TRAIN 能做到：

\[
<5\sim10\text{ mm}
\]

而 dev 是：

\[
30\sim50\text{ mm},
\]

那是：

\[
\boxed{\text{generalization / multi-object coverage}}
\]

问题。

如果 TRAIN/dev 都能：

\[
<10\sim15\text{ mm},
\]

那基本就坐实：

\[
\boxed{\text{当前 54 mm 主要就是 all8 objective + selection 搞出来的}}
\]

而不是这个 inverse 本身不可学。

---

# 我还会要求按阶段拆指标

至少拆：

\[
tick<16
\]

startup，

和：

\[
tick\ge16.
\]

最好再根据 hand-object distance 分：

- pre-contact；
- near-contact；
- post-contact。

因为我预期：

\[
\boxed{
\text{pre-contact }(s,\tau)\rightarrow a_t
}
\]

应该非常容易。

如果 pre-contact 都 50 mm：

> 那基本还是实现 / objective / capacity 问题。

如果：

\[
precontact=8\text{ mm}
\]

但：

\[
contact=60\text{ mm},
\]

那才说明我们真的撞到了：

> contact force / object dynamics 未观测导致 inverse ambiguity。

---

# 所以我现在的原因排序

| 优先级 | 原因 | 判断 |
|---|---|---|
| **1** | 8-step closed-loop command 等权训练，但只执行 first | **目前最大问题** |
| **2** | checkpoint 按 total forecast objective 选，不按 control | **明确错位** |
| **3** | first XYZ 在 action loss 中只占 3/96 | **严重稀释监督** |
| **4** | 50-object R 不看 geometry/contact | 很可能造成接触阶段 ambiguity |
| **5** | R 只有 128D×2 blocks | 可能不足，但先别扩 |
| 低 | timing / coordinate / PD conversion bug | **目前没发现，已有 audit 支持正确** |

所以我现在不会说：

> “R 这个映射本身不确定，所以 54 mm 正常。”

**54 mm 不正常。**

更准确地说：

\[
\boxed{
\text{第一步 inverse 本应远比现在容易，但我们的 loss 并没有在专门学它。}
}
\]

而且我们两天前已经有实验说明，把 objective 改成 first-action/PD-aware 能把同类问题从 31 mm 拉到约 20 mm 并恢复 GT 物理执行。

我认为当前 RL660 训练跑完之前，完全可以并行做这个 **GT-\(\tau\) R-only first-action probe**。这个实验会非常快，而且它比继续猜 WAM、加数据或扩模型更能直接定位现在 54 mm 到底从哪来。