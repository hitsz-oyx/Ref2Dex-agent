我看了最新仓库。`agent/cm-interaction-oracle` 现在 HEAD 是 **`d7a72df09b`**，而且你上一轮提到的 **early-hold / retention-control** 实验已经完整跑完了。

这次结果其实很关键，因为它把问题进一步缩窄了。

## 当前最重要的结论

现在可以比较有把握地区分两件事：

\[
\boxed{I^{GT}\text{ 对“之后能不能保持抓取”很有预测价值}}
\]

但还没有证明：

\[
\boxed{\text{当前这类动作能够有效操纵这个 }I}
\]

也就是说：

> **prognostic value 很强，control value 还没建立。**

这比之前“E/I 到底有没有用”清楚得多。

---

## 这轮数据比上一轮靠谱很多

这次 intervention 真正放到了 early-hold：

- 1008 个完整 episode；
- 494 次 early-hold 随机干预；
- 7 个 action arms，每个 59–79 次；
- 每次都有完整 32-step future；
- 所有 intervention 前都已经满足：
  \[
  \text{lift}\ge3\text{ cm}
  \]
  且连续 6 step contact proxy；
- 494/494 都是 at-risk 状态。

后面实际有：

\[
214/494
\]

发生 32-step failure，其中：

- 212 个有真实物体高度下降；
- 99 个有连续 contact-loss；
- 98 个在前 8 step 已经开始 failure。

所以这次已经不是之前 pre-lift 那种“根本没进入抓持阶段”的问题。

---

# Gate A：动作能不能操纵 retention-related I？

**没有通过。**

预先要求 short contact fraction 的 arm effect：

\[
\ge 10\text{ percentage points}
\]

实际只有：

\[
\boxed{5.684\text{ pp}}
\]

permutation tail：

\[
0.1335
\]

要求是 \(\le0.10\)。

整个 I14 family：

\[
\text{tail}=0.5005
\]

最大 partial explained variance 也只有：

\[
2.736\%
\]

因此当前证据不支持：

\[
\boxed{
a_t+\delta
\rightarrow
I_{\rm retention}
}
\]

至少不支持**现在这个四步 feedback residual operator**。

这一点是当前最大的瓶颈。

---

# Gate B：如果我已经知道 GT I，能不能判断之后会不会保持？

这里反而非常强。

H-only：

\[
\text{normalized MSE}=0.3553
\]

加入 GT I8：

\[
\boxed{0.1792}
\]

改善：

\[
\boxed{49.6\%}
\]

retention32 MAE：

\[
0.1387\rightarrow0.0775
\]

failure32 MAE：

\[
0.2200\rightarrow0.1688
\]

failure AUC：

\[
0.899\rightarrow\boxed{0.949}
\]

registered pair ranking：

\[
58.46\%\rightarrow77.33\%
\]

提升：

\[
\boxed{+18.86\text{ pp}}
\]

这个数值上已经不是“弱信号”了。

---

## 为什么 Gate B 最后还是 UNCLEAR？

不是因为效果小。

而是因为预注册要求：

> 至少 3 个充分支持的 stratum。

实际只有：

\[
\boxed{2}
\]

所以正式结论只能写 `UNCLEAR`。

而且原始 macro rank 里有一些只有 1–2 pair 的小 strata，会放大结果。

如果只描述性地看那些真正有 ≥10 pair 的 strata：

\[
H:87.76\%
\]

\[
H+GTI:96.32\%
\]

仍然有：

\[
\boxed{+8.55\text{ pp}}
\]

所以虽然不能升级 Gate，但 **GT I 的 prognostic value 已经相当可信。**

---

# 还有一个很重要的检查

可能会怀疑：

> GT I 只是因为前 8 step 已经看到“抓取开始失败”，所以预测后面继续失败。

仓库专门做了一个 post-hoc 检查。

在 test 中把前 8 step **还没有 failure** 的 92 个 trial 单独拿出来：

其中后来仍有 26 个 failure。

结果：

\[
\text{MSE}:0.3039\rightarrow0.1854
\]

改善：

\[
39.0\%
\]

failure AUC：

\[
0.865\rightarrow0.920
\]

所以 GT I 的价值**并不完全是“看见已经掉了，所以预测以后还会掉”**。

这个结果我认为很重要。

---

# Gate C 没有执行

因为 A 没过。

所以没有训练：

\[
(H,a)\rightarrow\hat I
\]

也没有训练：

\[
(H,\hat I)\rightarrow S
\]

更没有 selector 或 policy。

这也是正确的。

因为现在最大的未知量不是：

> “我们能不能把 I 预测得更准？”

而是：

> **“我们能不能通过允许的动作把有价值的 I 朝更好的方向改变？”**

如果这个都没有，预测 I 再准也只能做状态诊断。

---

# 所以现在对 I 的判断已经和几轮之前不同了

我现在不会再说：

> I 可能有用。

我会更具体地说：

\[
\boxed{
I\text{ 对 grasp-retention prognosis 有很强的证据}
}
\]

但：

\[
\boxed{
I\text{ 是否是一个可控的 action mediator 仍未建立}
}
\]

这是目前整个 Cm policy-utility 问题的核心。

---

# 为什么四步 residual 可能失败？

当前 intervention 是：

\[
a_t=\pi(o_t)+\delta
\]

持续 4 step，然后又交回：

\[
\pi(o)
\]

这种反馈控制本身可能很快把 perturbation 抵消掉。

所以 A 的失败至少有几种解释：

1. **动作幅度不够**；
2. **4 step 太短**；
3. baseline feedback 在 intervention 后迅速恢复原轨迹；
4. 当前 residual 方向（wrist x/z + finger synergy）没有对准真正 controlling retention 的自由度；
5. 我们测的 I14 并没有包含真正可操纵的 slip/contact geometry；
6. retention 很大程度上由接触微状态/摩擦决定，确实不是这个动作空间容易控制的。

当前实验**不能区分这些解释**。

所以仓库 closing note 写：

> 下一问题应该区分 action/operator timescale 与 feedback cancellation。

我同意。

---

# 我认为现在不应该做什么

不要继续：

- 增大 I predictor；
- 加 GRU/Transformer；
- 多跑几个 Cm seed；
- 补到 3 个 strata 只为了让 Gate B 过；
- 回到 RTG；
- 回到 RECAP；
- 直接开始 PPO。

因为 **B 已经不是当前最重要的问题。**

即使 B 正式过了：

\[
I^{GT}\rightarrow\text{retention}
\]

我们还是卡在：

\[
a\not\Rightarrow I.
\]

---

# 如果继续，我只建议验证一个东西

下一次实验应该非常窄：

> **是不是四步 feedback residual 太容易被 baseline controller 抵消？**

也就是先不训练任何网络，只做 physics response。

例如比较不同 intervention duration：

\[
K=4,\ 8,\ 16
\]

或者短时间固定 target offset，而不是每一步：

\[
\pi(o_t)+\delta
\]

让 policy feedback 不断重新补偿。

然后直接看：

\[
\Delta I_{\rm retention}
\]

和：

\[
\Delta P(\text{loss/drop})
\]

是否随 intervention dose/duration 呈现可重复的 monotonic response。

不用先训练 Cm，也不用训练 \(S\)。

如果延长/改变 operator 后仍然：

\[
a\rightarrow I
\]

没有稳定响应，那我会开始真正怀疑：

> **我们选择的 I 虽然非常会“看出抓取要坏了”，但不是一个可以靠当前 action 修正的变量。**

那它更适合 observation/diagnostic，而不是 world-model control mediator。

---

## 现在项目状态可以压成这张图

\[
a
\overset{?}{\longrightarrow}
I
\overset{\checkmark}{\longrightarrow}
\text{retention/drop}
\]

右半边现在已经相当有希望。

左半边仍然没建立。

所以我认为当前仓库相比上一轮其实是**有实质进展的**：我们已经不需要再争论“interaction 有没有价值”，真正的问题已经变成了更具体的：

\[
\boxed{\text{有没有一个动作/operator 能稳定操纵这个有价值的 interaction？}}
\]

这应该是下一步唯一值得优先回答的问题。

**目前还不能把问题归结为“Cm 预测 \(I\) 不准”。** 更准确地说，Cm 预测 \(I\) 这一步在最新 early-hold 实验里甚至还没有被正式执行，因为前置 Gate A 没过。

现在证据链是：

\[
a \overset{\color{red}{?}}{\longrightarrow} I
\overset{\color{green}{\checkmark}}{\longrightarrow}
\text{retention/drop}
\]

右边已经比较明确：给定 GT \(I\)，对后续 grasp retention 的判断明显变好。比如 normalized error 从 \(0.355\) 降到 \(0.179\)，failure AUC 从 \(0.899\) 到 \(0.949\)。所以 **\(I\) 作为“抓持稳定性的未来表征”是有价值的。**

现在卡住的是左边：

\[
\boxed{a\rightarrow I}
\]

当前 4-step residual 干预下，动作对 retention-related \(I\) 的影响太弱：short-contact arm range 只有约 **5.68 pp**，没有达到预设 10 pp；整个 I14 family 也没有稳定 randomized-arm effect。

所以存在两种完全不同的可能性：

1. **控制问题**：当前动作/operator 根本不能有效改变 \(I\)。例如 4 step 太短、feedback actor 很快抵消 residual、动作方向不对。
2. **Cm 问题**：动作确实能改变 \(I\)，但 learned Cm 学不会这个映射。

目前我们连 1 都没有排除，因此还不能进入 2。

也就是说，如果现在直接训练：

\[
Cm(H,a)\rightarrow\hat I
\]

然后发现 \(\hat I\) 很差，我们根本不知道：

> 是 Cm 不行，

还是：

> 对给定 \(H\)，不同 \(a\) 本来就只造成非常小的 \(I\) 差异，state 本身已经解释了绝大部分 \(I\)。

这也是为什么最新实验把 Gate C 停掉了。

### 真正应该先确认什么

下一步应该先不训练 Cm，只问：

\[
\boxed{\text{能不能找到一个 action operator，使 }I\text{ 对 action 有足够大的响应？}}
\]

比如固定相同 early-hold decision region，比较：

\[
K=4,\ 8,\ 16
\]

的 residual duration，或者改变执行方式，减少 baseline feedback 对 perturbation 的快速抵消。直接测：

\[
\Delta I
\]

以及：

\[
\Delta\text{retention/drop}.
\]

如果这时出现清楚的 dose-response：

\[
a_1,a_2,\ldots
\Rightarrow
I_1,I_2,\ldots
\]

那么才进入真正的 Cm 问题：

\[
\boxed{Cm(H,a)\text{ 能否预测这些 action-dependent }I?}
\]

如果预测不出来，才可以说“Cm 预测 I 是瓶颈”。

所以目前的诊断顺序应该是：

\[
\underbrace{I\rightarrow Y}_{\text{已有强信号}}
\]

↓

\[
\underbrace{a\rightarrow I}_{\text{当前主要瓶颈}}
\]

↓

\[
\underbrace{(H,a)\rightarrow\hat I}_{\text{Cm，尚未真正测试}}
\]

↓

\[
\hat I\rightarrow\text{action selection}
\]

**因此现在的主要问题不是 Cm 预测 \(I\)，而是我们还没有证明“当前可执行动作能够充分操纵那个有预测价值的 \(I\)”；Cm 是下一层问题。**