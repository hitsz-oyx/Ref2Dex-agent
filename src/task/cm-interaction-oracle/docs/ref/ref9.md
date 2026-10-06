我看了 `agent/cm-interaction-oracle` 的最新状态。仓库已经往前走了一步：**ref8 已经把我们刚才说的 GT \(a\rightarrow(E,I)\rightarrow Y\) 实验做了。**

当前最重要的新实验是：

`P-20261005-gt-consequence-sufficiency.md`  
实验 commit `08975e1`，之后还有 1 个提交补了审计、STATE 和实验卡。分支目前相对 `main` 是 **ahead 428 / behind 0**。

## 1. ref8 做的基本就是我们刚才讨论的实验

没有重新采集，直接复用 ref7：

- 854 个 randomized intervention windows
- 15 个 action arms
- K8 / 20% driver range
- 688 train / 166 test
- 125 / 31 个 environment clusters，严格环境隔离

然后固定容量比较了：

\[
H
\]

\[
H+a
\]

\[
H+E
\]

\[
H+I
\]

\[
H+E+I
\]

\[
H+a+E+I
\]

还额外预注册了一组 signed-force 扩展。

所以你刚才问的：

> “动作对 I/E 的影响幅度是不是足以确定 Y？”

**现在已经真正开始回答了。**

---

## 2. 最大的新结论：GT \(E/I\rightarrow Y\) 非常强

结果很明显：

| 输入 | primary test MSE | 相对 H 改善 |
|---|---:|---:|
| H | 0.7314 | — |
| H+a | 0.5925 | **19.0%** |
| H+E | 0.4789 | **34.5%** |
| H+I | 0.4117 | **43.7%** |
| H+E+I | **0.3931** | **46.25%** |

而且 `H+E+I` 的 46.25% 改善 bootstrap 95% CI 是：

\[
34.02\%\sim57.03\%.
\]

不是之前那种靠一个 episode 撑起来的弱信号。

对**真实 physical height failure**：

\[
H\rightarrow HEI
\]

误差也改善了 **32.52%**，95% CI：

\[
11.24\%\sim49.92\%.
\]

因此目前可以比较明确地写：

\[
\boxed{\text{GT }(E,I)\text{ 含有很强的未来任务结果信息}}
\]

而且值得注意：

\[
I\text{ alone}: 43.70\%
\]

已经非常接近：

\[
E+I:46.25\%.
\]

所以 **I 目前不是一个弱辅助量，它在这个 early-hold 问题里实际上是主要信息源之一。**

---

# 3. 更关键：加入 E/I 后，action 还剩多少信息？

这正是你上一条问的核心。

比较：

\[
HEI
\]

和

\[
HaEI.
\]

结果点估计：

- primary：
  \[
  -0.325\%
  \]
- physical failure：
  \[
  -0.925\%
  \]

也就是说，**加入 \(E/I\) 后，再告诉模型 action，点估计上几乎一点帮助都没有，甚至稍差。**

这非常接近我们想看到的：

\[
a\rightarrow(E,I)\rightarrow Y
\]

即：

> action 对 Y 有信息，但一旦知道实际产生的 \(E/I\)，action 本身基本不再提供额外信息。

因为前面已经看到：

\[
H+a
\]

比 H 好 **18.99%**。

所以形成了一个很漂亮的现象：

\[
a\Rightarrow Y
\]

有信息；

\[
(E,I)\Rightarrow Y
\]

信息更强；

而：

\[
a+(E,I)\Rightarrow Y
\]

没有明显超过 \(E/I\)。

**如果只看 point estimate，我会说这已经非常符合“E/I 捕获 action 后果”的假设。**

---

## 4. 但为什么仓库仍然写 UNCLEAR？

因为预注册标准比“点估计没增益”更严格。

要宣称 E/I 足够，需要排除：

> action 其实还能提供超过 5% 的额外信息，只是这批样本没测出来。

目前 `HaEI - HEI` 的 one-sided 95% upper bound：

- primary：**7.96%**
- physical：**9.64%**

都超过预注册的 5%。

所以现在只能说：

\[
\boxed{
\text{没有观察到明显剩余 action 信息}
}
\]

但不能说：

\[
\boxed{
E/I\text{ 已经被证明充分}
}
\]

这就是现在 `GT prognosis = PROMISING`，但 `full sufficiency = UNCLEAR` 的原因。

这个判断我认为是合理的，不应该为了推进路线把它改成 positive。

---

# 5. 还有一个重要的麻烦：signed force 后结果变了

ref8 还专门测试了更丰富的 signed-force representation。

结果：

\[
HEI_{\text{signed}}
\]

再加入 action 后：

\[
HaEI_{\text{signed}}
\]

反而有：

- primary：**+8.68%**
- physical failure：**+12.91%**

的改善。

这说明现在还存在一个重要的不确定性：

> “action 的剩余信息”究竟是真实存在，还是因为当前 E/I representation、模型容量、H 压缩、优化不足造成的。

实验卡里也明确记录了模型还没有明显训练收敛：

最后 50 epochs 的 train loss 还在下降约 **38–46%**。

因此我们现在不能简单说：

> action 全部通过 E/I 作用于 Y。

---

# 6. 不过，还有一项结果让我更看好这条链

ref8 不只是做预测模型，还做了跨 half 的 **action-arm consequence bridge**。

它先估计每个 action 的：

\[
\Delta E_j,\Delta I_j
\]

以及：

\[
\Delta Y_j
\]

然后用一半数据的 action consequence 去预测另一半里的 task effect。

对于 \(E+I\)：

### Contact

相对 zero baseline 改善：

\[
21.65\%
\]

correlation：

\[
0.550
\]

### Height failure

相对 zero baseline 改善：

\[
16.14\%
\]

correlation：

\[
0.723
\]

这已经开始回答你之前特别关心的问题：

> “动作对 I 的影响幅度到底够不够对应 Y？”

答案目前是：

**至少跨 action direction 来看，\(E/I\) 的 signed/vector variation 和后续 Y variation 是明显相关的。**

不是只有：

\[
a\rightarrow I
\]

和：

\[
I\rightarrow Y
\]

两个互不相干的结果。

现在已经有第三块：

\[
\boxed{
\Delta(E,I)_a
\text{ 能预测一部分 }
\Delta Y_a
}
\]

虽然这还不是严格的 causal mediation identification。

---

# 7. 所以现在研究状态和刚才已经不同了

我会把当前证据链写成：

### 第一段

\[
a\rightarrow I
\]

**PROMISING / 已经比较明确。**

ref6/ref7 已经证明强局部 response。

### 第二段

\[
(E,I)_{\rm GT}\rightarrow Y
\]

**PROMISING，而且现在相当强。**

46.25% primary error improvement。

### 第三段

\[
\Delta(E,I)_a\rightarrow\Delta Y_a
\]

**有正向证据。**

跨 half EI bridge：

- contact correlation 0.55
- height failure correlation 0.72

### 但完整的

\[
a\rightarrow(E,I)\rightarrow Y
\]

**仍然是 UNCLEAR，不是因为没有信号，而是 sufficiency / representation / uncertainty 还没有闭合。**

这是一个明显比昨天更积极的状态。

---

# 8. 接下来我同意仓库里的决定：现在该做 \(H+a\rightarrow E/I\)

ref8 最后一段的 root decision 我认为方向是对的。

现在不应该继续：

- 加大 action；
- 扰动更多 finger；
- 扫更多 amplitude；
- 再证明 GT I 对 Y 有用；
- 直接上 PPO。

下一步应该直接问：

\[
\boxed{
C_m:\quad(H,a)\rightarrow(E,I)
}
\]

到底能不能学。

而且实验应该非常干净地比较：

\[
H\rightarrow E/I
\]

vs

\[
H+a\rightarrow E/I.
\]

这里最核心的不是普通 reconstruction MSE，而是要看：

\[
\hat{\Delta E}_a,\hat{\Delta I}_a
\]

能不能保留 action contrast。

然后把预测值接入完全冻结的 downstream scorer：

\[
(H,\hat E,\hat I)\rightarrow Y
\]

和 direct：

\[
(H,a)\rightarrow Y
\]

比较。

这样就真正回答：

\[
\boxed{
\text{预测出来的 consequence 是否比直接看 action 更有 task value}
}
\]

---

## 我对目前仓库的总体判断

这一次更新是**实质性的进展**，不是又做了一个外围 Probe。

之前我们只有：

> I 好像有用。

后来有：

> action 能明显改变 I。

现在 ref8 又补上：

> **强 action 数据上，GT E/I 对 Y 很有用，而且 action-specific E/I variation 和 Y variation 有明显对应关系。**

所以现在继续卡在“\(a\) 到底对不对 I 有影响”已经没有必要。

**下一步正式进入 conditional consequence prediction 是合理的。**

真正的下一个关键实验应该是：

\[
\boxed{
H+a\rightarrow(E,I)
\rightarrow Y
}
\]

而这一次如果 predicted \(E/I\) 也能保留 ref8 的 task gain，我们就真正到了可以讨论 oracle / candidate selector / policy integration 的位置。

有，而且我觉得有几条必须提前钉死，不然这个实验很容易又变成“模型 MSE 好像不错，但回答不了核心问题”。

1. **先把问题定义死：这次不是再证明 GT E/I 有用，而是验证 conditional predictability。**  
   核心比较必须是
   \[
   H\rightarrow(E,I)
   \quad\text{vs}\quad
   H+a\rightarrow(E,I)
   \]
   再加一个
   \[
   H+\text{shuffled }a\rightarrow(E,I)
   \]
   如果真实 \(a\) 没明显优于 H，或者和 shuffled \(a\) 差不多，就不能说 Cm 学到了 action consequence。

2. **不要只看整体 E/I MSE。**  
   这是最重要的一点。当前真正关心的是：
   \[
   \Delta(E,I)_a
   \]
   能不能被预测出来。必须报告每个 arm 的 GT contrast 与 predicted contrast：
   \[
   \Delta I_j = I(a_j)-I(a_0),\qquad
   \Delta \hat I_j=\hat I(a_j)-\hat I(a_0)
   \]
   E 同理。至少看方向一致性、相关性、幅度误差和跨 half / held-arm 泛化。否则模型完全可能靠预测状态均值降低 MSE，却忽略 action effect。

3. **ref7/ref8 数据、split 和表示尽量冻结。**  
   我不建议现在重新采数据，也不要顺手扫 seed、width、horizon、action amplitude。最好直接复用 ref8 的 environment-cluster split，所有 normalization / PCA 只能 train fit，test 环境完全隔离。这样新结果才能和 ref8 的 GT upper bound直接对齐。

4. **action 输入只能是决策时可知的 action。**  
   用 assigned/intended randomized action。不要把后续实际 feedback action、PD target sequence、未来 baseline correction 等输入 predictor。那些都是 action 执行后的信息，会把问题从
   \[
   p(E,I\mid H,a)
   \]
   偷换成“看了一部分未来以后预测未来”。

5. **E 和 I 要分开报告，再报告联合。**  
   现在 ref8 显示 I alone 已经有 43.7% task-error gain，E 是 34.5%，EI 是 46.25%。所以不能只给一个 26D aggregate MSE。至少分别给：
   - E12 prediction；
   - I14 prediction；
   - E+I joint；
   - 每个 action arm 的 contrast preservation。

6. **signed-force 扩展只能作为 sensitivity，不能事后救主结果。**  
   ref8 已经告诉我们 signed representation 会明显改变“剩余 action 信息”的判断。因此主 representation 应该冻结为当前 E12/I14；signed18 可以预先声明为 secondary sensitivity，但不能主结果不好时换成 signed 重新宣布通过。

7. **downstream 必须用 OOF predicted E/I。**  
   训练 task scorer 时，训练样本上的 predicted E/I 必须来自 cross-fitted predictor，不能用同一个 predictor 对自己的训练集输出。否则很容易制造虚假的
   \[
   \hat E,\hat I\rightarrow Y
   \]
   增益。test 可以用 full-train predictor。

8. **direct \(H+a\rightarrow Y\) baseline 必须保留。**  
   这是整个实验最容易被忽略但最重要的对照之一。最后至少比较：
   \[
   H
   \]
   \[
   H+a
   \]
   \[
   H+\hat E+\hat I
   \]
   最好再有
   \[
   H+a+\hat E+\hat I.
   \]
   如果 predicted E/I 不能超过 direct \(H+a\)，或者至少提供稳定的额外增益，就还不能说 Cm representation 对决策有独特价值。

9. **最好量化“保留了多少 oracle gain”。**  
   ref8 已经给了 GT oracle：
   \[
   H:0.731,\qquad HEI:0.393.
   \]
   所以新实验可以直接定义一个很有解释力的指标：
   \[
   R=
   \frac{L(H)-L(H+\hat E+\hat I)}
   {L(H)-L(H+E+I)}.
   \]
   它回答的是：
   > predicted consequence 保留了 GT consequence task value 的多少比例？
   
   这比单纯说“提高了 3%”更有意义。

10. **不要把预测成功写成 mediation 已证明。**  
    即使最后得到
    \[
    a\rightarrow \hat E,\hat I
    \]
    且
    \[
    \hat E,\hat I\rightarrow Y,
    \]
    仍然只能说 consequence representation 对 action-conditioned task prediction 有效。不能直接说“E/I 是 action→Y 的因果中介”。现在的数据不是 same-state counterfactual mediation design。

11. **保留所有 early failure，不做 survivor filtering。**  
    延续 ref8 的时间合同：step8 E/I，Y 看 step9..32。不能因为某些 trial 在 step8 已经坏掉就删掉；否则会重新引入 post-treatment conditioning。

12. **训练充分性要提前处理。**  
    ref8 有一个明显警告：最后 50 epochs train loss 还下降 38–46%。所以这次如果还是固定 200 epochs，很可能又把“action 信息不存在”和“模型没训好”混起来。建议预先规定一个更充分但仍固定的 optimization budget，或者只基于 train loss 的停止规则；绝不能看 test 后再续训。

我最希望 AI 不要犯的一个错误是：

> **看到 \(H+a\rightarrow(E,I)\) 的整体 MSE 有提升，就宣布 action-conditioned Cm 成功。**

这还不够。

这次真正的成功链应该是：

\[
a
\overset{\text{predictable}}{\longrightarrow}
(\hat E,\hat I)
\]

并且

\[
(\hat E,\hat I)
\overset{\text{retain oracle task value}}{\longrightarrow}
Y
\]

同时最好满足

\[
H+\hat E+\hat I
\gtrsim
H+a
\]

或者至少在 direct \(H+a\) 上提供稳定增益。

**前半段证明 Cm 真学到了动作后果，后半段证明这个后果表示值得存在。** 这两个缺一不可。