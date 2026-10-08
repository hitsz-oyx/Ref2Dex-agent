我们现在要重新定义 consequence evaluator 的 GT value。目标不是继续使用 airplane 特化的 `lift/hold/drop/S-P-M` 标签，而是得到一个可以直接监督候选动作排序的、reference-conditioned 标量标签。

核心定义：

\[
\boxed{
Y_t(A)=P_{t+K}-P_t
}
\]

其中 \(K=24\)。

这里的 \(P_t\) 不是 `t/T`，也不是当前 episode 的真实时间进度，而是：

\[
P_t=P(H_t\mid R)
\]

即“当前实际轨迹历史 \(H_t\) 在成功 reference trajectory \(R\) 中对应的任务阶段/进度”。

具体做法：

1. 每个任务提供一条或多条成功 reference trajectory：
   \[
   R=(r_0,\ldots,r_{N-1})
   \]

2. reference 的索引天然定义进度：
   \[
   p_k=\frac{k}{N-1}
   \]

3. actual rollout 和 reference 都转成统一的 3D trajectory feature，第一版至少包含：
   - object pose；
   - 11 个 hand keypoints 在 object frame 下的位置；
   - object motion；
   - hand-object relative motion。
   
   不要只用 object xyz。

4. 不允许用单帧最近邻，因为 reference 开头和结尾可能物理状态相似，例如“抓起后重新放回桌面”。必须使用短历史 clip 做 temporal alignment，例如最近 8 帧 actual clip 与 reference 各阶段 clip 比较。

5. temporal alignment 实现优先参考 TCC/XIRL；reward/matching 数据流可以参考 ROT/OTR。第一版可以先不用训练模型，直接用标准化后的几何序列距离得到每个 reference index 的匹配分布：
   \[
   q_t(k)
   \]

6. 当前 progress 定义为 soft reference index：
   \[
   P_t=
   \sum_k q_t(k)\frac{k}{N-1}
   \]

7. phase tracking 必须使用历史 prior：
   - 允许向前；
   - 允许停滞；
   - 允许向后；
   - 但限制单步突然跳到 reference 很远的位置。
   
   这样正常“抓起→放回”仍能继续匹配 reference 后段，而真正“抓起→意外掉落”可以表现为 progress 回退。

8. 对一个实际执行的候选动作 \(A\)，只使用到 \(t+24\) 为止的数据：
   \[
   P_t=P(H_{0:t}\mid R)
   \]
   \[
   P_{t+24}=P(H_{0:t+24}\mid R)
   \]
   禁止使用 \(t+24\) 之后的 episode future，避免 expert 后续恢复污染当前动作标签。

9. 最终 GT value：
   \[
   \boxed{
   Y_t(A)=P_{t+24}-P_t
   }
   \]

   - \(Y>0\)：候选让任务向成功 reference 推进；
   - \(Y\approx0\)：基本没有推进；
   - \(Y<0\)：候选导致任务退步。

10. candidate ranking 直接由 \(Y\) 产生：
   \[
   Y_i>Y_j+\epsilon
   \Rightarrow A_i\succ A_j
   \]
   差异太小时 abstain，不强行排序。

这套 GT value 的第一目标是作为离线标签生成器。先在现有 rollout 上验证 nominal success 的 \(P_t\) 基本前进、失败轨迹会停滞/回退；验证通过后，再用这些 \(Y\) 训练 evaluator。不要先继续扩 S/P/M 或训练新的 airplane-specific value head。

**如果你要的是“能处理同一状态在不同阶段出现、避免抓起再放下混淆”的源码参考，我最推荐两套结合看：**

- **TCC / XIRL**：专门做**时序阶段对齐**。不是单帧最近邻，而是学习不同轨迹中“同一操作阶段”的 embedding，并用 cycle consistency 检查时序对应关系。Google 官方代码完整。这个最适合解决“开始时物体在桌上”和“最后放回桌上”外观相似但阶段不同的问题。[谷歌研究](https://research.google/blog/robot-see-robot-do/?utm_source=chatgpt.com)  
  核心代码在 Google Research 的 `tcc/` 和 `xirl/`，尤其是 alignment loss、cycle-consistency evaluator、nearest-neighbour/reward 部分。

- **ROT / OTR**：已经完整实现**专家示例轨迹 ↔ 实际 rollout 的轨迹级对齐并自动产生 reward**。ROT 的核心就在 `ROT/rewarder.py` 和 `ROT/agent/potil.py`，用 cost matrix + Sinkhorn 对齐整条轨迹，而不是只比较最终物体位置。[GitHub](https://github.com/siddhanthaldar/ROT?utm_source=chatgpt.com) OTR 是更模块化的 reward-labeling 版本。

不过要注意：**ROT 原版的 OT 对齐本身不专门解决“重复状态对应不同阶段”的歧义**；如果特征里没有时序信息，仍可能混。最稳的工程参考是：

\[
\boxed{\text{TCC/XIRL 学时序 phase embedding}+\text{ROT/OTR 做 reference matching}}
\]

而最贴近我们最终想法的 **RARM** 确实直接用 rollout clip 对 successful reference clips 做匹配，并加 confidence gate，正是为 progress ambiguity 设计的；可惜目前项目公开信息里还没有完整训练代码可直接 fork。[RARM](https://rarm-robotics.github.io/?utm_source=chatgpt.com)

所以如果现在让 AI 去实现，我建议明确告诉它：**优先复用 Google TCC/XIRL 的 temporal alignment 逻辑，再参考 ROT 的 rewarder 数据流；不要自己从零写一个单帧 cosine matcher。**