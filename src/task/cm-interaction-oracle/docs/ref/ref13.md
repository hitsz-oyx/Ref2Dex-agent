我看了 `agent/cm-interaction-oracle` 分支，当前 HEAD 是 **`ff09a0c`**（`Record ref12 OOF flow task chain results and audited limits`）。重点核对了 `gt-consequence-sufficiency`、`conditional-consequence`、`oracle-hand-flow`、`oracle-flow-task`、`spatial-consequence` 这一串实验。

先给结论：

> **我建议现在暂停“继续把 Y/E/I 预测得更准”这条惯性路线，改成真正从后往前做必要性验证。**
>
> 下一步最重要的问题不是“Y 能不能预测”，而是：
>
> **如果我有一个足够好的 Y，它能不能真的让我选出更好的抓取动作/点流？需要好到什么程度才能产生抓取收益？**
>
> 在这个问题通过之前，不应该再做 PPO、不应该继续扫 Cm 结构，也不应该默认 `F → E/I → Y` 是必须的链。

这里我先统一分支里的语义：

\[
H=\text{当前状态/历史},\quad
F=\text{未来手部点流},
\]

\[
M=(E,I)=\text{物理后果},
\]

\[
Y=\text{step 9 以后 continuation outcome},
\]

最终真正关心的我另记为

\[
Z=\text{真实抓取目标，例如稳定持有/不掉落/成功}.
\]

这最后一个 \(Z\) 很重要，因为你现在仓库里的 **Y 本身不是最终抓取成功率**，文档也明确把它叫 continuation proxy，而不是 final episode success。

---

## 一、客观看，现在其实已经得到一个很重要的结果

| 问题 | 当前结果 | 我认为能下的结论 |
|---|---:|---|
| GT E/I 对 Y 有用吗？ | ref12：H 0.6395 → GT E/I 0.4544，**+28.95%** | **有用** |
| 真实未来点流 F 对 Y 有用吗？ | H 0.6395 → Flow 0.4579，**+28.40% [12.23, 41.48]** | **非常明确有用** |
| 真实 F 能预测 E/I 吗？ | Chunk E **+7.42% [0.46,14.04]**；I +8.07%，CI 跨 0 | E 有证据，I 尚不稳 |
| 预测 E/I 对 Y 有用吗？ | PredEI：+14.45%，CI **[-1.57,29.47]** | 有趋势，但不稳 |
| E/I 比直接 F→Y 多带来东西吗？ | Flow+PredEI 比 Flow **+4.57% [-1.96,10.41]** | **目前没有证据** |
| intended action→E/I 足够好吗？ | I 点估计改善，但 action contrast 很弱 | **不够** |
| nominal geometry 能解决吗？ | spatial / geometric probes 均未通过 | **现在不能** |

这里最值得注意的其实不是 PredEI 的失败。

而是：

\[
\boxed{
F_{\text{oracle}}\rightarrow Y
\quad\text{和}\quad
(E,I)_{\text{GT}}\rightarrow Y
}
\]

在 ref12 的共同 H 合同下，**信息量几乎一样**：

\[
28.40\%\quad vs\quad 28.95\%.
\]

而

\[
F\rightarrow \hat E,\hat I\rightarrow Y
\]

反而只保留了一部分。

这给我一个很明确的判断：

> **目前没有理由继续把 E/I 强制设成 F→Y 的信息瓶颈。**

E/I 是有物理意义的，这一点已经证明了。

但目前没有证明：

\[
F\rightarrow E/I\rightarrow Y
\]

是控制上必须经过的路径。

这两个命题完全不同。

---

# 二、真正的问题：现在“Y 预测得准”并不等价于“抓取得更好”

这是目前最容易继续绕进去的地方。

ref12 的 Flow→Y 结果很好：

\[
28.4\%\ \text{prediction gain}.
\]

但这个 F 是 **actual future hand flow**。

也就是：

\[
H_t
\rightarrow
\underbrace{F_{t:t+8}^{\text{actual}}}_{\text{未来已经发生}}
\rightarrow
Y.
\]

它证明的是：

> “如果我知道手未来实际上怎么运动，我能更好判断之后会不会保持住。”

这属于 **prognosis**。

它没有证明：

> “在当前状态给我几个候选点流，我能用 Y 模型知道应该选哪个。”

后者才是 planning。

数学上，真正需要的是：

\[
\hat Y_k=f(H,F_k),
\]

对同一个 \(H\) 给 K 个候选点流：

\[
F_1,F_2,\ldots,F_K,
\]

然后

\[
k^*
=
\arg\max_k U(\hat Y_k).
\]

执行 \(F_{k^*}\) 后，真正的抓取结果 \(Z\) 要变好。

### 所以我们需要的不是

\[
\operatorname{MSE}(\hat Y,Y)\downarrow
\]

而是：

\[
\boxed{
\text{candidate ordering 正确}
}
\]

以及最终：

\[
\boxed{
Z(F_{k^*}) > Z(F_{\text{baseline}})
}
\]

这才是 Y 存在的理由。

这和 decision-aware / value-equivalent model learning 的出发点其实完全一致：模型没有必要把世界每一项都预测准，关键是保存规划所需要的价值关系。[NeurIPS 会议论文集](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html?utm_source=chatgpt.com)

---

# 三、所以我认为下一步应该反过来：先做 **Oracle-Y Utility Gate**

这是我现在最推荐的实验。

**先不训练新的 Y predictor。**

甚至 E/I、Cm 都暂时拿掉。

直接回答：

> **如果 Y 是完美的，它到底能不能改善我们的抓取决策？**

这一步过不了，后面的 Y predictor 全部没有意义。

---

## 实验应该长这样

选一个现在最清楚的阶段：

> **early hold / 已经抓住之后，如何避免掉落。**

先不要整个 grab 全任务。

给同一个抓取前缀状态 \(H_i\)，定义一个非常小的候选集合，例如 K=5～8 个局部点流/干预：

\[
\mathcal F_i=
\{
F_i^0,F_i^1,\ldots,F_i^{K-1}
\}.
\]

其中：

- \(F^0\)：baseline；
- 其他是小幅 thumb / finger / wrist / grip-flow 改变；
- 不追求覆盖动作空间。

然后真正执行每一个 candidate，得到：

\[
Y_i^k,\qquad Z_i^k.
\]

这里第一次真正出现：

\[
(H_i,F_i^1,Y_i^1),
(H_i,F_i^2,Y_i^2),
...
\]

**同状态候选结果。**

这是现在仓库一直缺的证据。

---

# 四、PhysX warm state 不能 snapshot，不代表做不了 paired experiment

你之前担心这个问题是对的。

仓库现在也确认了：warm PhysX solver cache 不能完整序列化，所以不能简单：

> 跑到 t → copy state → 恢复 → fork action。

但其实我们不需要这么做。

可以做：

\[
\text{reset seed }s
\rightarrow
\text{完全相同 prefix}
\rightarrow
t
\rightarrow F_1
\]

然后重新：

\[
\text{reset same seed }s
\rightarrow
\text{完全相同 prefix}
\rightarrow
t
\rightarrow F_2.
\]

也就是说：

> **prefix replay branching，而不是 snapshot branching。**

因为整个接触历史也重新跑了一遍，PhysX solver cache 会随着相同 prefix 自然形成。

当然第一件事情要验证：

\[
\|H_t^{(1)}-H_t^{(2)}\|
\]

足够小。

而且可以先做：

\[
F_0 \text{ vs } F_0
\]

的重复 branch，得到 simulator 自身的 noise floor。

如果 baseline-vs-baseline 都差很大，这批状态直接不作为 paired evidence。

这个实验的价值远比再训练一个 predictor 大。

---

# 五、然后我们才能真正回答你问的：“Y 要预测多好才有帮助？”

这甚至可以 **不用先训练模型**。

有了 paired candidate 数据以后，直接使用 GT Y。

先得到：

\[
k_{\text{oracle}}
=
\arg\max_k U(Y_i^k).
\]

比较：

\[
Z_i^{oracle}
\]

和：

\[
Z_i^{baseline}.
\]

### 第一关：

如果 GT Y oracle 都无法明显改善 Z：

\[
Z^{oracle}\approx Z^{baseline},
\]

那直接结束。

说明：

> **当前 Y 定义就不是一个值得预测的决策变量。**

不要再改善 Y predictor。

可能要换 Y。

---

如果 oracle 明显提升，就进入一个特别有价值的实验：

## 人为破坏 GT Y

构造：

\[
\tilde Y=Y+\epsilon.
\]

逐渐增大噪声。

每个噪声水平都用：

\[
k^*
=
\arg\max_k U(\tilde Y_k)
\]

重新做 selection。

就可以画出一条非常关键的曲线：

\[
\boxed{
\text{Y prediction quality}
\quad\longrightarrow\quad
\text{grasp decision improvement}
}
\]

这才真正回答：

> **Y 要预测到什么水平才值得做。**

例如可能发现：

- pairwise accuracy 55%：没有作用；
- 65%：略好；
- 75%：已经保留 70% oracle gain；
- 85% 后几乎饱和。

那以后你训练任何模型，都有一个明确目标。

不需要再说：

> “MSE 从 0.6 变成 0.5，看起来不错。”

---

# 六、以后 Y 的主要评价指标也应该换掉

我建议 MSE 降级为诊断指标。

真正第一优先级应该是：

### 1. Pairwise ordering

同一个 \(H\)：

\[
F_i,F_j
\]

是否预测对：

\[
\operatorname{sign}
[
U(\hat Y_i)-U(\hat Y_j)
]
=
\operatorname{sign}
[
U(Y_i)-U(Y_j)
].
\]

### 2. Top-1 regret

模型选：

\[
\hat k=\arg\max U(\hat Y_k).
\]

真实最优：

\[
k^*=\arg\max U(Y_k).
\]

看：

\[
R=
U(Y_{k^*})-U(Y_{\hat k}).
\]

### 3. 最重要的：selected-action outcome

也就是模型实际选出来那个 candidate：

\[
Z(F_{\hat k})
\]

相对于 baseline 到底提高多少。

---

# 七、而且我甚至不建议主要预测 absolute Y，应该预测 **ΔY**

这是我觉得你们现在最值得改变的一点。

现在的 factual prognosis 很容易学成：

> “这个状态本来就要掉了。”

那么 H 已经贡献很多。

动作信息只占很小一部分。

真正用于规划的应该是：

\[
\boxed{
\Delta Y(H,F)
=
Y(H,F)-Y(H,F_0)
}
\]

即：

> **这个点流相比 baseline 会把结果改变多少。**

甚至直接建模：

\[
P(F_i \succ F_j\mid H).
\]

这样状态本身的“难度”被抵消了。

这和你现在一直追的：

> “动作到底改变了什么”

其实是一回事。

只是现在把它放到了最终 decision level。

---

# 八、E/I 下一步应该降级成什么角色？

基于当前实验，我不会删 E/I。

但我会改变它的地位。

以前隐含的架构是：

\[
F
\rightarrow
E/I
\rightarrow
Y
\rightarrow
action.
\]

我现在更支持：

\[
F
\rightarrow
\begin{cases}
\hat Y & \text{task decision}\\
\hat E,\hat I & \text{physical explanation / constraint}
\end{cases}
\]

即 **并行 heads**。

E/I 不再被强迫承担“所有任务信息都必须穿过我”的责任。

这样非常符合当前数据：

- GT E/I 明显有任务意义；
- E/I 可解释；
- 但是 PredEI 没证明比 direct Flow 更有效；
- hybrid 对 direct Flow 的额外 +4.57% 还没有统计上站稳。

所以现在最合理的解释是：

> **E/I 是好的物理监督和解释变量，但尚未证明它是必要的 task-sufficient bottleneck。**

以后只有出现下面任一情况，才把 E/I 升级回核心：

1. E/I 显著提高 candidate ranking；
2. E/I 显著提高跨状态/跨物体泛化；
3. E/I 提高 uncertainty calibration；
4. E/I 能提供明确的物理约束，避免 Y scorer 选择明显不合理的 flow。

否则它就做 auxiliary physics head。

---

# 九、Y 最好也不要直接塞进 PPO

至少第一阶段绝对不要。

如果现在做：

\[
[\text{obs},\hat Y]\rightarrow PPO
\]

最后 PPO 提升了，我们基本无法知道：

- 是 Y 真会选动作；
- 是额外网络容量；
- 是 representation shaping；
- 是 critic 更容易拟合；
- 还是某个偶然 shortcut。

这正是你说的不想做的“黑盒研究”。

### 第一版应该极其简单：

冻结原 baseline policy。

它给出：

\[
F_0.
\]

在它周围生成少量：

\[
F_1,\ldots,F_K.
\]

然后用 Y scorer：

\[
F^*
=
\arg\max_F U(\hat Y(H,F)).
\]

只做 **reranking / residual selection**。

这样如果抓取提升，我们能很明确地说：

> 同一个 baseline、同一个 state、同一个候选集合，仅仅因为 Y 排序，选出的动作更好。

这条因果链非常干净。

以后才考虑蒸馏成 policy 或放进 PPO。

TD-MPC2 这类方法本质上也是让 learned model 通过局部 trajectory optimization 直接服务动作选择，而不是要求世界模型把所有真实状态都高精度重建。[arXiv](https://arxiv.org/abs/2310.16828?utm_source=chatgpt.com)

---

# 十、因此我会把现在的路线改成下面这四关

这是我建议你让 agent 接下来真正推进的主线：

1. **Gate A — Y 本身有决策价值吗？**  
   用 paired candidate 数据 + GT Y 做 oracle selector。GT Y 都不能提升 stable grasp，就停止 Y 路线。

2. **Gate B — Y 要多准才有价值？**  
   给 GT Y 人工加噪，画 `ranking quality → grasp gain` 曲线。由下游任务反推出 predictor 必须达到的水平。

3. **Gate C — 哪种 representation 最接近这个要求？**  
   在完全相同 paired data 上比较 `H+F→Y`、`F→E/I→Y`、`F+E/I→Y`。评价 pairwise ranking、top-1 regret 和 selected outcome，不再以普通 MSE 为主。

4. **Gate D — 最后才碰执行和策略。**  
   当 flow-space selector 已经证明有效，再处理 `native action → realized F`，之后才是在线 reranker / MPC / residual PPO。

---

## 我现在对主方向的判断

如果现在一定要选一个研究假设，我会押：

\[
\boxed{
\text{H + candidate point-flow}
\rightarrow
\Delta Y
\rightarrow
\text{candidate selection}
}
\]

作为**主控制链**。

同时：

\[
\boxed{
\text{point-flow}\rightarrow E/I
}
\]

作为**物理解释链 / auxiliary physics model**。

而不是继续坚持：

\[
\boxed{
\text{point-flow}\rightarrow E/I\rightarrow Y
}
\]

必须是唯一主链。

因为你们自己的最新 ref12 已经第一次给出了很强的证据：**direct flow 的 task information 基本达到 GT E/I 的量级，而 predicted E/I bottleneck 反而在丢信息。**

所以现在最有科研价值的事情已经不是“再造一个更强 Predictor”。

而是把问题收缩成一句很干净的话：

> **什么样的短期手部运动会让当前抓取变得更稳；我们至少要多准确地判断这种相对改善，才能真的选出更好的运动？**

我认为这会比继续研究“Y MSE 能不能再降几个点”清晰很多，也更符合你说的 **从后往前、必要性探路、不做黑盒研究**。