# Decision Memo — HF04 之后的 Cm policy-utility 路线复盘（2026-09-26）

## 需要决定的问题

在 HF01 local-effect-ranking、HF02 temporal expert-option、HF03
contact-supported-credit 和一次独立的 HF04 trajectory-level-credit screen
都没有让 North-star 的 matched Cm-on/off 取得进展后，是否继续在当前
self-trained airplane substrate 上寻找新的 Cm credit 接法。

## 关键证据

1. HF01 已耗尽 bounded local-effect-ranking 预算；短期物理效应和部分
   CATE 排序可学习，但既有 matched policy 结果没有稳定超过 Cm-off。
2. HF02 canonical six-expert temporal option 的 holdout `temporal_cm`
   相对 `history_only` 为 `+12.903 pp`，相对 `action_shuffled` 为
   `-9.677 pp`，未通过预设双侧 `+5 pp` gate。
3. HF03 的 post-action handflow 在同一 substrate 上没有达到相对 action-aware
   与 placebo 各 `+5%` 的 held-lift Brier 和 contact-supported lift 联合改进。
4. HF04 是不同的高层假设：用完整的 post-option trajectory token 预测首回合
   held-lift。fit 121 行、holdout 126 行，三臂和 finite checks 均通过；但
   相对 `pre_action` 只有 `+0.29%` 的 held-lift Brier 改善且连续 lift RMSE
   恶化 `-12.95%`，相对 trajectory-shuffled 只有 `+2.69%/+4.05%`，没有达到
   预先冻结的双侧 `5%` gate。
5. HF01–HF04 的失败发生在不同 credit 表示或时间范围上；继续换 seed、窗口、
   阈值、metric 或模型头不能提供新的可判别信息。当前没有通过门槛的 Cm
   policy-utility Probe，也没有理由启动 online、PPO 或新的 GPU collection。

## Option A — 冻结当前 Cm policy-utility campaign（当前建议）

保留 HF01–HF04 的代码、数据、测试和负结果，停止新的 GPU collection、PPO、
online Probe 以及局部 representation sweep。将当前可支持的结论限定为：
Cm 在部分短期物理转移上有可学习信息，但在本 substrate 上尚未证明对最终
策略有因果增益。

* 成本：0 个 GPU；只做复现、论文边界和 Research Debt 整理。
* 去向：形成可审计的 negative-result closeout，避免把局部效应写成 policy
  utility。

## Option B — 建立新的研究问题

如果仍要推进，需要明确不同于当前 C3 policy-utility credit family 的高层
问题，并新建 goal、experiment ID、资源预算和 Decision Checkpoint。新问题
必须先说明它如何改变 North-star 决策；不能通过重跑 HF02–HF04 的 seed、
horizon、metric、representation 或 online 变体来继续消耗旧 slot。

* 成本：由新 goal 预先定义；当前不启动任何资源。
* 成功后：形成独立路线，再决定是否值得进入正式 Validation。
* 失败后：保持 Option A 的 closeout。

## Recommendation and current action

建议 Option A，并提交用户 Decision Checkpoint。HF04 的 card、CPU evaluator、
测试、run manifest、结果 hash 和本复盘已提交；当前分支停止实验。没有启动
GPU、physical collection、PPO 或 online Probe。继续工作需先决定新的高层 goal
及预算，不能追加当前 HF04 family 的 slot。
