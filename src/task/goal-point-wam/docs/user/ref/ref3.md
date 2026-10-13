有，而且这块现在已经有比较明确的经验可以借鉴。结合我们现在的问题，我觉得最有参考价值的不是“简单删静止窗口”，而是 **按样本的信息量、任务进展和交互阶段做窗口级筛选/加权**。

最对口的是 **SCIZOR（ICRA 2026 Oral）**。它专门研究大规模模仿学习的数据清洗，核心不是按整条轨迹粗暴删，而是识别两类低价值 transition：一类是“没有任务进展”的 suboptimal transitions，比如 pause、卡住、无意义来回运动；另一类是高度重复的 state-action pair，用去重保留多样性。它报告用更少的数据反而提升了下游策略表现。[Robot Perception and Learning Lab](https://ut-austin-rpl.github.io/SCIZOR/?utm_source=chatgpt.com)  
[SCIZOR 项目页](https://ut-austin-rpl.github.io/SCIZOR/?utm_source=chatgpt.com)

这和我们现在的情况非常像：我们的问题不是“静止窗口一定坏”，而是某些窗口里：

\[
G\approx G_0
\]

但手轨迹却可以千差万别。对于 goal-conditioned actor，这些样本会制造高条件熵：

\[
H(\tau\mid H,G)
\]

变大，模型会被迫平均多个不一致的未来。

另一个很值得参考的是 **DemInf / Robot Data Curation with Mutual Information Estimators**。它用 state-action mutual information 衡量示范质量，核心直觉是：好的数据应该既有动作多样性，又要让动作在给定状态后具有较强可预测性；数据清洗后在 RoboMimic 和真实机器人上都有收益。它还强调过度做 step-level 剪枝可能误删“难但必要”的交互段，所以更适合做**窗口/轨迹级打分，而不是逐帧硬删**。[arXiv](https://arxiv.org/abs/2502.08623?utm_source=chatgpt.com)

第三类是 **ReMix**。它不判断单个 transition，而是学习不同数据域/子集应该采多少。它在 Bridge/Open-X 上证明“数据比例”本身对性能影响很大，而且自动学习的 mixture weights 明显好于均匀采样和人工配比。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v270/hejna25a.html?utm_source=chatgpt.com)

这对我们也很有启发：我们现在 main-three 的 moving / near-static 本身就在固定比例采样，但这个比例并不是为：

\[
\text{object goal}\rightarrow\text{hand trajectory}
\]

这个任务专门设计的。也就是说，现在的 sampling contract 很可能继承了 PointWorld 的需要，却不一定适合 actor。

还有两个非常实用的工程经验。DROID 官方 policy pipeline 默认只保留成功轨迹；其公开数据说明还特别提醒，数据里大量 pause 会让 policy 学会在 home pose 输出 idle action，因此推荐过滤 idle-action frames。[GitHub](https://github.com/droid-dataset/droid_policy_learning/blob/master/examples/droid_dataloader.py?utm_source=chatgpt.com) Octo 的数据管线本身也把 filtering、goal relabeling、chunking 设计成 trajectory-level transforms，并支持 dataset-specific filter functions，这说明“先 curate，再 chunk/训练”已经是大规模机器人学习里的标准工程思路。[GitHub](https://github.com/octo-models/octo/blob/main/octo/data/dataset.py?utm_source=chatgpt.com)

另外，针对你特别关心的“接触前窗口不能误删”，**EgoLoc** 很值得借鉴。它专门做 hand-object contact / separation 的时间定位，而且用 hand dynamics 来选择有信息的时刻，而不是只看物体有没有发生宏观位移。[PubMed](https://pubmed.ncbi.nlm.nih.gov/42525935/?utm_source=chatgpt.com) 这和我们现在真正应该做的 cleaning 非常接近：

> object 可能还没动，但如果手正在高速接近、即将 contact，这个窗口反而是最重要的。

---

## 对我们现在的数据，我建议不要“静止就删”

我会把每个 24-step window 计算几个很便宜的量：

\[
m_o=\text{object motion magnitude}
\]

比如 translation + rotation；

\[
m_h=\text{hand motion magnitude}
\]

比如 11 个 hand point 的 future displacement；

\[
d_{\min}
=
\min_{t\in[1,24]}
d(\text{hand}_t,\text{object}_t)
\]

以及：

\[
\Delta d
=
d_t-d_{\min}
\]

表示是否正在靠近物体。

再加一个最关键的：

\[
g=\|P_G-P_0\|
\]

也就是 object goal 到底有没有信息。

然后把窗口分成四类：

| 类型 | 特征 | 对 actor 怎么处理 |
|---|---|---|
| 无信息 idle | \(g\) 小、\(m_o\) 小、\(m_h\) 小 | 大量丢弃/强降权 |
| 模糊手运动 | \(g\) 小、\(m_o\) 小、\(m_h\) 大、离物体远 | **不要强训练 goal→trajectory** |
| approach/contact | \(g\) 小或中等，但 \(d_{\min}\) 小/明显接近 | **重点保留** |
| manipulation | \(m_o\) 或 \(g\) 明显 | 正常全权重训练 |

我尤其建议第二类：

\[
\boxed{
G\approx const,\quad
hand\ moving,\quad
hand-object\ far
}
\]

不要继续用完整：

\[
L_\tau
\]

训练 goal-conditioned actor。

因为这类样本最容易造成：

\[
\text{同一个 goal}\rightarrow\text{很多不同 hand trajectory}
\]

从而污染监督。

但它们不一定是“废数据”。可以继续拿来训练：

- PTv3 scene representation；
- 无条件 hand dynamics；
- world branch；
- temporal encoder。

只是**不应该强迫 object goal 去解释这段 hand motion**。

---

## 我甚至建议把现在的 actor loss 改成带权

不用第一版就真的删数据。

先做：

\[
L_\tau
=
w_{\rm actor}
D(\hat\tau,\tau^{GT})
\]

其中一个简单版本：

\[
w_{\rm actor}=
\begin{cases}
0.05,&\text{idle}\\
0.1,&\text{static-goal + unrelated hand motion}\\
1.0,&\text{approach/contact}\\
1.0,&\text{object manipulation}
\end{cases}
\]

这样比 hard filtering 安全得多。

因为 DemInf 的经验也提醒我们：有些“看起来低进展”的段落可能实际上是任务必需的准备动作，粗暴逐帧删除容易把关键过渡删掉。[arXiv](https://arxiv.org/abs/2502.08623?utm_source=chatgpt.com)

---

## 还有一个更高级但很适合我们的指标

我们甚至可以直接测：

\[
\boxed{
\operatorname{Var}
(
\tau^{GT}
\mid
z_H,z_G
)
}
\]

也就是：

> 找到 embedding 里状态和 goal 很接近的 KNN 样本，看它们未来 hand trajectory 到底有多一致。

如果：

\[
(H_i,G_i)\approx(H_j,G_j)
\]

但：

\[
\tau_i\gg\tau_j
\]

差别很大，那么这就是一个**天然的高歧义样本区域**。

这其实就是 DemInf 的“state-conditioned action predictability”思想在我们任务里的具体化。

对我们而言，比传统的：

\[
\text{object moved > 5mm ?}
\]

更科学。

---

# 我认为最值得先做的不是重新训练，而是一个统计

先对现在 main-three 的所有 train windows 统计：

1. \(g=\) object endpoint displacement；
2. \(m_h=\) hand trajectory displacement；
3. 当前和未来最小 hand-object distance；
4. 是否发生 contact/proximity transition；
5. trajectory KNN conditional variance。

然后看四个比例：

\[
\boxed{
P(g<1{\rm mm}),
P(g<2{\rm mm}),
P(g<5{\rm mm})
}
\]

再把这些静止-goal window 分成：

\[
\text{hand-static}
\]

\[
\text{hand-moving-far}
\]

\[
\text{hand-approach}
\]

\[
\text{hand-contact}.
\]

我觉得这个统计非常可能直接解释我们现在的大规模训练为什么 goal utility 有，但接触阶段仍然很差。

如果最后发现例如：

> 30–40% 训练窗口都是 \(G\approx P_0\)，其中大部分 hand 又在执行 unrelated motion，

那我们就有非常充分的依据重新做 actor sampling。

---

### 最适合我们借鉴的几条工作

- **SCIZOR**：state-action level 的“任务进展 + 去冗余”清洗，和我们最接近。[Robot Perception and Learning Lab](https://ut-austin-rpl.github.io/SCIZOR/?utm_source=chatgpt.com)
- **DemInf**：用 state-action predictability / mutual information 判断数据质量，适合解决“同一个 condition 对应太多未来”的问题。[arXiv](https://arxiv.org/abs/2502.08623?utm_source=chatgpt.com)
- **ReMix**：不要只决定“删不删”，还可以直接学不同 motion/interaction strata 的采样权重。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v270/hejna25a.html?utm_source=chatgpt.com)
- **DROID pause filtering**：明确说明 idle/pause 数据会把 policy 教成 idle，是最直接的实际经验。[Hugging Face](https://huggingface.co/KarlP/droid/blob/main/README.md?utm_source=chatgpt.com)
- **EgoLoc**：用 hand dynamics 找 contact/separation 关键时刻，特别适合避免误删“物体没动但正在形成接触”的窗口。[PubMed](https://pubmed.ncbi.nlm.nih.gov/42525935/?utm_source=chatgpt.com)

所以我现在不会做：

\[
\boxed{\text{object static}\Rightarrow\text{delete}}
\]

而会做：

\[
\boxed{
\text{goal informativeness}
+
\text{hand motion}
+
\text{interaction proximity}
+
\text{conditional trajectory ambiguity}
}
\]

四个量联合决定 actor supervision 权重。

这比我们现在按 moving/near-static 的固定 0.6/0.2/0.2 sampling 更符合 Goal-Point-WAM 本身。