我看了最新仓库。现在 `agent/cm-interaction-oracle` HEAD 是 **`adeb952dc8`**；真正最新的科研结论在前一个提交 **`6f0271b66a`：`Record reviewed randomized intervention gate and physical-control limits`**。

这次其实已经把我们刚才说的 **randomized physical action intervention** 完整做了一遍。结果比之前的 proxy-advantage 实验更有解释力。

### 这次实验做得比较对

已经不是从旧 trajectory 猜动作价值了，而是真正在仿真里随机干预：

\[
a_t=\pi(H_t)+\delta_j
\]

七个 arm：

\[
0,\quad wrist_x\pm0.01,\quad wrist_z\pm0.01,\quad finger\ synergy\pm0.1
\]

连续作用 4 step，然后回到原 actor。

最终：

- 336 个完整 episode；
- 320 个实际 intervention；
- 每个 arm 37–54 次；
- 320 个完整 32-step future window；
- intervention dose **100% 真正执行**；
- train/test 按 simulator environment 分组隔离；
- deterministic actor，没有原先的 episode noise；
- 独立 review 验证了 PD target、时序、E/I 重建和 split。

所以这次至少不存在“动作其实没打进去”的问题。

---

# 最重要的结果：动作确实改变了物理，但目前没有改变我们关心的任务结果

手本身的响应非常明确。

例如 wrist ± intervention 后，第 4 step 手部 x/z 位移 contrast 大约：

\[
13.3\text{ mm},\quad 13.8\text{ mm}
\]

所以：

\[
\boxed{\delta\rightarrow \text{hand motion}}
\]

成立。

进一步做 randomized-arm adjustment 后，也看到一个弱的物体响应：

\[
\text{max partial explained variance of E/I}\approx7.79\%
\]

主要来自物体 y-axis rotation。

wrist x+/x− 相对 zero 的调整后旋转 contrast 大约：

\[
+0.0271,\quad -0.0502\text{ rad}
\]

family-max permutation tail 是 0.045。

这个只能叫 exploratory signal，但至少说明：

\[
\boxed{\delta\rightarrow E}
\]

不是完全不存在。

---

## 但是 learned \(H+a\rightarrow E/I\) 没学出来

测试集：

| Predictor | E/I MSE | E MSE | I MSE |
|---|---:|---:|---:|
| \(H\) | **0.5748** | **0.7272** | **0.4442** |
| \(H+a\) | 0.5841 | 0.7387 | 0.4516 |
| \(H+\) shuffled action | 0.5836 | 0.7376 | 0.4515 |

所以：

\[
H+a
\]

不但没比 \(H\) 好，反而差约 1.6%。

而打乱 action 基本也不影响结果。

因此当前这个数据/模型/干预范围下：

\[
\boxed{a\rightarrow E/I}
\]

虽然物理统计上有弱响应，但还没有形成一个**可泛化预测的 action-conditioned consequence model**。

---

# 更重要的是 \(S\) 的结果

这次也真正实现了我们前面讨论的四路比较：

\[
H
\]

\[
H+a
\]

\[
H+\hat E+\hat I
\]

\[
H+E^{GT}+I^{GT}
\]

结果：

| scorer | task MSE | outcome ranking |
|---|---:|---:|
| H | 0.3767 | **66.78%** |
| Direct H+a | 0.3767 | 66.15% |
| predicted E/I | 0.3728 | 65.17% |
| **GT E/I** | **0.3641** | 61.18% |

所以整体来看：

### GT E/I

绝对误差改善：

\[
3.34\%
\]

但 ranking 反而：

\[
-5.61\text{ pp}
\]

### predicted E/I

MSE 改善：

\[
1.02\%
\]

ranking：

\[
-1.61\text{ pp}
\]

而且把 mediated scorer 的 action 打乱以后，ranking 反而上升约 4pp。

因此这轮的 **7 个预注册 gate 全失败**。

仓库没有启动 selector，也没有继续在线 policy，这个决定是对的。

---

# 但有一个我认为非常重要的正信号

不要被 aggregate GT ranking 的 61% 掩盖。

**GT I 对 contact retention 的作用非常明显。**

16-step retention：

\[
\text{MAE}:0.07915\rightarrow0.06232
\]

ranking：

\[
66.31\%\rightarrow74.21\%
\]

32-step retention：

\[
\text{MAE}:0.07805\rightarrow0.06984
\]

ranking：

\[
74.09\%\rightarrow\boxed{88.78\%}
\]

这个结果和我们之前 Gate1 里 I 比 E 更有希望的趋势其实是连起来的。

它说明：

\[
\boxed{
I^{GT}\text{ 很可能真的包含“未来接触是否维持”的有效信息}
}
\]

只是把：

- \(\Delta z\)
- retention
- held
- drop

全揉进 aggregate task scorer 后，height / held 等 head 把整体结果拖坏了。

因此我现在会比之前更加谨慎地区分：

> **“E/I 没有 task value”——不成立。**

和：

> **“当前这版 E/I8→Y16/32 aggregate scorer 不适合动作选择”——成立。**

---

# 这次最大的实验问题其实非常清楚

这 320 次 intervention **全部发生在 pre-lift**。

仓库统计：

\[
\text{max current lift}=8.743\text{ mm}
\]

没有任何 intervention 发生在：

\[
\boxed{\text{early hold}}
\]

也没有任何 decision state 属于真正的 drop-risk 状态。

所以我们刚才真正最关心的：

> 动作能否改变 grasp retention / slip / drop？

**这次其实还没有测试。**

它测试的是：

> 在刚接触、还没真正提起来的时候，4-step 小 residual 能否产生短期 E/I，并进而预测 Y16/32。

这是一个合理的第一版 physical intervention，但它没有命中我们最终真正关心的 control regime。

---

# 这里出现了一个很有意思的矛盾

完整 336 episode 中：

\[
105/336
\]

后来达到了 45-step hold，

其中：

\[
76
\]

后来又 drop，

只有：

\[
29
\]

达到 hold 后没有记录到后续 drop。

说明这批 trajectory **其实拥有大量我们需要的 hold/drop variation**。

但是 intervention 是在 episode 很早的时候打的。

所以：

\[
\text{intervention time}
\ll
\text{interesting grasp-stability time}
\]

这就导致当前 \(a_t\) 到后面的 retention/drop 信号被几十步 actor feedback 淹掉。

我认为这比“网络太小”更可能是主要问题。

---

# 所以我对仓库现在的判断和之前稍有变化

我们刚才说：

> 做 randomized physical intervention。

**这件事已经做了，不应该原样继续。**

当前结果已经足够关闭：

\[
\boxed{
\text{pre-lift + 4step residual + E/I8 + Y16/32}
}
\]

这个具体合同。

不要：

- 换 seed 再跑；
- 加大 MLP；
- 去掉 PCA 重训；
- 再换 advantage；
- 再重新组合 aggregate Y。

仓库现在的 closing decision 也是这个意思，我同意。

---

# 但下一步非常明确

如果还继续 Cm 主线，我认为只值得再做一个**不同科学问题**：

## Early-hold randomized intervention

不是：

> 第一次满足 contact/proximity 就 intervention。

而是只在进入：

\[
\boxed{\text{object lifted + contact established + early hold}}
\]

之后 intervention。

例如一个 decision state 至少满足：

\[
\Delta z > 3\text{cm}
\]

且：

\[
\text{contact currently retained}
\]

然后随机打短 chunk。

此时直接研究：

\[
\delta
\rightarrow
I^{GT}
\rightarrow
\text{retention/drop}
\]

我甚至不会再把 E 和 I 平均对待。

因为当前最明确的结果已经是：

\[
\boxed{GT\ I\rightarrow retention}
\]

尤其 32-step ranking 达到 **88.8%**。

所以真正值得验证的是：

> **当抓取已经形成以后，不同动作能否改变这个高预测价值的 interaction / retention variable。**

---

## 我会把下一轮缩成三个 Gate

### Gate A：动作真的能改变 retention-related I 吗？

\[
a+\delta
\rightarrow
I^{GT}_{t:t+K}
\]

随机 arm 之间要有稳定 contrast。

### Gate B：GT I 能区分谁会继续 hold、谁会 drop 吗？

我们现在其实已经看到很强的初步证据，但要在 **early-hold intervention cohort** 上重新验证：

\[
S(H,I^{GT})\rightarrow
\text{retention/drop}
\]

### Gate C：Cm 能否预测这个 action-sensitive I？

\[
(H,a)\rightarrow\hat I
\]

只有 A+B 都过，才值得训练这个 Cm。

然后最终：

\[
a_i
\rightarrow
\hat I_i
\rightarrow
S(H,\hat I_i)
\]

选 candidate。

---

# 当前仓库最重要的科学结论

我会把现在状态压成一句话：

\[
\boxed{
\text{我们终于证明“动作会影响一些物理量”，也发现“GT I 很能预测接触保持”，}
}
\]

但还没有证明：

\[
\boxed{
\text{当前允许的动作能操纵这些真正决定抓取保持的 I。}
}
\]

这已经比几小时前清楚很多了。

因此我现在**不会再回到 RECAP/A，也不会重做这版 pre-lift intervention**。如果继续，我会直接把问题收缩到 **early-hold 的 interaction retention control**。这应该是现在最有判别力的一步。