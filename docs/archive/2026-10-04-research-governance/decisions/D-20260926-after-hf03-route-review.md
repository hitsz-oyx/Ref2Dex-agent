# Decision Memo — HF03 之后的 Cm 路线复盘（2026-09-26）

## 需要决定的问题

在 HF01 local-effect-ranking、HF02 temporal expert-option 和 HF03
contact-supported-credit 连续三个有效方向都没有让 North-star 的 matched
Cm-on/off 取得进展后，是否继续在当前 self-trained airplane substrate 上
寻找第四个局部 Cm 接法，还是冻结 policy-utility campaign 并重新定义后续
研究问题。

## 关键证据

1. HF01 已耗尽其 bounded local-effect-ranking 预算；短期物理效应和部分
   CATE 排序可学习，但既有 matched policy 结果没有稳定超过 Cm-off。
2. HF02 canonical six-expert temporal option 的 holdout `temporal_cm`
   相对 `history_only` 为 `+12.903 pp`，相对 `action_shuffled` 为
   `-9.677 pp`，未通过预设双侧 `+5 pp` gate。
3. HF03 在四个同 substrate 随机接触数据上，post-action handflow 对
   held-lift Brier 和 contact-supported lift 没有达到相对 action-aware 与
   placebo 各 `+5%` 的联合改进；held Brier 反而为 `-2.33%/-4.36%`。
4. 更早的 critic、auxiliary、单步/两步 option 和固定在线规则也没有形成
   跨训练 seed 的 matched policy gain。继续换 seed、窗口、阈值或模型头不会
   改变当前信息结构。

## Option A — 冻结当前 Cm policy-utility campaign（当前建议）

保留 HF01–HF03 的代码、数据和负结果，停止新的 GPU collection、PPO、
online Probe 和局部 representation sweep。将当前可支持的结论限定为：
Cm 在部分短期物理转移上有可学习信息，但在本 substrate 上尚未证明对最终
策略有因果增益。

* 成本：0 个 GPU；只做复现、论文边界和 Research Debt 整理。
* 去向：形成可审计的 negative-result closeout，避免把局部效应误写成
  policy utility。

## Option B — 新建 trajectory-level Cm/option critic 架构

用整段接触到承重的轨迹 token 或 recurrent critic 重新定义监督，再设计
新的 matched on/off/placebo Probe。这不是 HF02/HF03 的参数或 head 替换。

* 成本：预计设计实现 4–8 小时，至少一次新的物理 collection 和数小时
  GPU 训练；需要新的 goal、experiment ID、资源预算和完整 Decision
  Checkpoint。
* 成功去向：形成独立的高层 Cm credit 路线，再考虑 Validation。
* 失败去向：回到 Option A，关闭 Cm policy-utility campaign。

## Recommendation and current action

按 `AGENTS.md` 的三次无进展规则，当前执行 Option A。HF03 的 card、CPU
代码、测试、run manifest 和结果已经提交；不再用 HF02/HF03 的 seed、horizon、
metric、threshold 或 representation 继续消费预算。若要执行 Option B，必须
先建立新的高层 goal，而不是在当前分支追加实验。
