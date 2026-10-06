我看完最新仓库了。**ref9 已经把我们刚才讨论的关键实验完整跑掉了，而且现在瓶颈定位得比之前清楚很多。**

最新是 `P-20261005-conditional-consequence.md`，commit `5db4f08`。它基本严格照我们前面说的做了：

\[
H\rightarrow(E,I)
\]

vs

\[
H+a\rightarrow(E,I)
\]

vs shuffled-action，对训练集又做严格 environment OOF；下游还保留了 direct \(H+a\rightarrow Y\)、OOF predicted consequence，以及 GT oracle。

## 最关键的结果

第一，**action 确实进入了模型，而且对 \(I\) 有预测信息。**

\(I\) 的 test MSE：

- H：1.900
- H+a：1.739

点估计改善 **8.5%**。

更重要的是，把已经训练好的 Ha predictor 的 test action 打乱以后：

\[
I\text{ error} +11.96\%
\]

95% CI 是 **4.17%–22.24%**。

所以现在基本不能再说：

> “模型根本没利用动作。”

它确实在利用 \(a\) 来预测 \(I\)。

---

但是第二点更重要：

**它没有可靠地学到我们真正需要的 action contrast。**

逐 action arm 比较 GT 和预测的 \(\Delta I_a\)：

- correlation：**0.293**
- sign agreement：**62.29%**
- amplitude ratio：**0.369**
- 相比预测 zero，MSE 只改善 **2.22%**

预设要求是 corr ≥ 0.5、sign ≥ 0.65、MSE gain ≥ 10%。

所以：

\[
\boxed{
a\rightarrow I\text{ 有信息}
}
\]

不等于

\[
\boxed{
C_m(H,a)\text{ 已经能正确预测不同动作造成的 } \Delta I
}
\]

后一个目前明显还不行。

---

## E 更糟

Ha 相比 H：

\[
E\text{ MSE 反而恶化 }20.79\%.
\]

所以当前模型大致呈现：

\[
a\rightarrow I : \text{有弱到中等可预测性}
\]

但

\[
a\rightarrow E : \text{当前泛化失败}.
\]

这其实和前面我们观察到的现象挺一致：**现在任务价值主要也是 I 在贡献。**

ref8：

- H+E：+34.5%
- H+I：+43.7%
- H+E+I：+46.25%

---

# 下游结果也非常有意思

同预算：

| 模型 | Primary test MSE |
|---|---:|
| H | 0.7931 |
| direct H+a | **0.6478** |
| GT E/I | **0.4080** |
| predicted E/I from H | 0.7993 |
| predicted E/I from H+a | 0.6730 |
| H+a + predicted E/I | **0.5995** |

所以 predicted E/I 并不是完全没用。

它保留了 GT oracle gain 的：

\[
\boxed{31.18\%}
\]

bootstrap CI 是 2.90%–56.56%。

这实际上是一个值得保留的信号。

但关键比较：

\[
P_{Ha}=0.673
\]

还输给：

\[
direct\ Ha=0.648.
\]

即：

> 把 action 先转换成 predicted E/I，并没有比直接把 action 给 task model 更好。

即使：

\[
Ha + P_{Ha}=0.5995
\]

比 direct Ha 点估计好 **7.45%**，置信区间仍跨零。

所以现在还不能进入 selector。

---

# 最重要的新发现其实是：问题很可能是泛化，不是信息不存在

这个结果我认为比单纯的 A/B/C failed 更重要。

仓库后来专门做了 anomaly review，发现：

### 简单 train-mean baseline

E MSE：

\[
0.754
\]

I MSE：

\[
0.891
\]

### persistence baseline

E：

\[
0.883
\]

I：

\[
1.046
\]

而我们的 Ha MLP：

- E：1.909
- I：1.739

也就是说：

\[
\boxed{\text{简单 baseline 居然明显优于 learned predictor}}
\]

但同时训练集：

\[
Ha\ joint\ MSE \approx 0.065
\]

而 test：

\[
1.817.
\]

这是非常明显的：

\[
\boxed{\text{严重 train-fit / test-generalization gap}}
\]

而不是“模型连训练数据都学不会”。

独立 review 又检查了：

- OOF leakage
- time indexing
- action shuffle
- scaling
- normalizer
- E/I labels
- downstream scorer
- saved weights

没有找到关键 bug。

所以我现在对 ref9 的解释是：

> **我们已经跨过“有没有 action→I 信息”这个问题了，现在真正卡住的是：当前模型能不能从有限 randomized states 中学到可泛化的 action-conditioned physical law。**

---

# 所以我不建议继续给这个 MLP 加 epoch

仓库 root 的判断也是这样。

它虽然最后 200 epoch loss 还在降，但训练误差已经很低，test 极差。

再训练：

\[
\text{train error}\downarrow
\]

大概率只会继续 memorization，而不是解决：

\[
\text{test generalization}.
\]

现在应该改变**学习问题本身**，而不是训练更久。

---

# 我认为下一个 Probe 应该非常克制

不要马上换成 Transformer、GRU、大 Cm。

我会先做一个最简单的 **physics innovation / residual prediction**。

现在模型预测的是：

\[
(H,a)\rightarrow E,I.
\]

但是我们已经知道 persistence / mean 非常强。

那就改成：

\[
C_m(H,a)\rightarrow
\Delta(E,I)_{\text{over simple baseline}}
\]

比如：

\[
\hat I
=
I_{\text{persistence}}
+
f(H,a)
\]

以及：

\[
\hat E
=
E_{\text{simple dynamics}}
+
g(H,a).
\]

这样网络不需要重新学习绝大多数“状态本来会怎样”，只需要学：

\[
\boxed{\text{action 引起的 innovation}}
\]

也就是我们真正关心的：

\[
\Delta I_a,\Delta E_a.
\]

这和现在的科学问题反而更匹配。

---

还有一点我会认真考虑：

ref9 现在 action 用的是 **14D categorical one-hot arm**。

这对这个固定实验没问题，但是从 Cm 的角度并不自然。

真实的物理关系应该更接近：

\[
(H,\Delta a_{\text{continuous}})
\rightarrow
\Delta(E,I).
\]

尤其我们已经有实际：

- joint residual；
- PD target；
- physical range；
- actual joint response

这些信息。

所以如果下一步改协议，我更倾向于：

\[
\boxed{
\text{continuous action}
+
\text{predict physical innovation}
}
\]

而不是继续训练：

\[
\text{one-hot arm}
+
\text{absolute E/I}.
\]

---

## 现在整个研究状态我会这样概括

已经比较稳的：

\[
a\rightarrow I
\]

**有。**

\[
GT(E,I)\rightarrow Y
\]

**很强。**

\[
\Delta(E,I)_a\rightarrow\Delta Y_a
\]

**有正向证据。**

而现在失败的是：

\[
\boxed{
(H,a)\rightarrow \widehat{\Delta(E,I)}
}
\]

的**跨环境泛化质量**。

这和早期的“Cm 可能根本没有可学信号”已经是完全不同的问题了。

所以我认为这次 ref9 虽然标签仍然是 `UNCLEAR`，但它实际上把研究问题收窄得很好：**下一步不该继续证明 I 有用，也不该继续扩大动作，而应该解决 action-conditioned consequence predictor 的泛化问题。**

不是。这里需要把“当前 ref9 实验”和“我们原本的 OI-CmV2 架构”区分开。

**ref9 里的 action 不是点流。**代码里它直接把 15 个 intervention arm 编成类别变量：

\[
a=\text{14D one-hot}
\]

zero 是全 0，其余如 `index+ / index- / middle+ / ... / synergy-` 各占一个维度。底层仿真真正执行的当然还是 native joint/PD residual，但喂给 predictor 的不是关节轨迹，也不是 hand point flow，而只是“选了哪个 arm”。

同样，**ref9 也没有用 OI-CmV2 那套网络。**它的 predictor 明确就是：

\[
[H,\ a]\rightarrow \text{MLP}(64\rightarrow32)\rightarrow(E12,I14)
\]

其中 \(H\) 是历史/actor obs/context/base action 经 PCA32，再拼 current physical72。代码没有调用 `ObjectInteractionCmv2` 的 spatial/token/point-flow 模型。

所以这一轮实际上测试的是一个更简化的问题：

\[
\boxed{
\text{PCA state}+\text{categorical action}
\rightarrow
\text{compact }E12/I14
}
\]

而不是我们原先更物理化的：

\[
\boxed{
\text{geometry / point-flow }H
+
\text{continuous action effect}
\rightarrow
E/I
}
\]

这点很重要。

另外，**ref9 的 \(E\) 也不是 Ref5 那个 point-flow \(E\)**。这里的 E12 是 compact object consequence：

- translation change；
- rotation-vector change；
- linear velocity change；
- angular velocity change。

而 Ref5 那条才是真正的 point-flow consequence 路线。

因此我会稍微修正上一轮对 ref9 的解读：

> ref9 证明的是“一个简单 PCA+one-hot MLP 在 conditional consequence 上泛化不好”，不是“我们的 OI-CmV2 / point-flow 架构泛化不好”。

这两者差别很大。

事实上 ref9 的几个异常现在也更容易理解。one-hot `index+` 和 `index-` 对网络只是两个互不相关的类别，它不知道：

\[
a_{\text{index+}}\approx -a_{\text{index-}}
\]

也不知道 `index+` 与 `middle+` 都属于相似的连续关节空间扰动，更不知道这些 action 最终造成了怎样的 hand surface motion。

所以它必须靠 688 个样本自己记住 14 个类别在各种状态下的物理规律，跨环境很容易过拟合。

而 OI-CmV2 原本希望利用的恰恰是这种结构：

\[
a
\rightarrow
\text{hand geometric motion / point flow}
\rightarrow
\text{object interaction}
\]

这在归纳偏置上合理得多。

### 因此，我现在认为下一步不应该简单做“给这个 MLP 加正则”

更值得问的是：

**我们是不是应该回到真正的 OI-CmV2 action representation。**

例如把 native action 通过当前 hand state/FK 转成未来 hand geometry change / point-flow：

\[
F_a =
P_{\text{hand}}(H,a)-P_{\text{hand}}(H)
\]

然后预测：

\[
(H,F_a)\rightarrow(E,I).
\]

这样：

- 正负 action 有自然的几何关系；
- 不同 finger action 可以通过实际 surface displacement 比较；
- mimic coupling 自然包含进去；
- 同样 20% joint range 导致 9 mm thumb / 29 mm ring 这种差异也直接反映在输入里；
- 不需要把 14 个 arm 当成互不相关的类别记忆。

**所以答案很明确：当前 ref9 不是 point-flow action，也不是 OI-CmV2。**

它更像是为了快速回答 conditional predictability 而搭的一个 controlled MLP Probe。它失败以后，我反而觉得我们应该认真讨论：**下一步是否正式把 ref7 的动作转回 OI-CmV2 所需要的 continuous geometric/point-flow action representation，再做 \(H,a\rightarrow E/I\)。**这和单纯“换更大的网络”是两个完全不同的方向。