# Decision Memo — 重新开启 Cm 研究：保守因果干预门控（2026-09-26）

## 需要决定的问题

HF01–HF04 的 direct selector、temporal credit 和 trajectory credit 均未给出
稳定的 Cm policy utility。用户随后明确要求按代理方案继续研究，因此需要一个
新的高层假设；不能把旧 family 的 seed、窗口或模型头继续扫描。

## 关键证据

1. 随机接触干预仍显示 Cm one-step information 为 `PARTIAL`，但直接选择
   expert 或 wrist-z action 没有通过 policy-value gate。
2. 旧路线大多在每个状态都输出一个候选动作，或预测最终 held-lift；它们没有
   测试“信息不足时保持原动作”的安全决策结构。
3. 已有 seed250/251 fit、seed252/253 holdout 的三臂随机数据包含完整的
   pre-action state、`-1/0/+1` assignment、first-episode held-lift、contact
   fraction 和 contact-supported lift，可以在 CPU 上检验该结构，无需新采集。

## Option A — selective causal gate（本轮采用）

拟合 arm-specific causal outcome models，并用 fit-only bootstrap 的下置信界
决定是否介入：只有预测 contact-supported lift 比 no-op 高至少 5 mm，且
contact fraction 的上置信损失不超过 2 个百分点时，才选择 `-z` 或 `+z`；其余
状态执行 no-op。比较 always-base、point gate 和 treatment-shuffled gate。

* 成本：CPU-only，一次 fit/holdout，约 20 分钟，输出不超过 20 MB。
* 成功后：只提交一次独立 confirmation memo，再决定是否值得设计 matched
  online Probe；本卡不会自动启动 GPU。
* 失败后：HF05 标记 `UNPROMISING`，冻结该新假设，不再改门槛、seed 或目标。

## Option B — 保持冻结

停止所有新 Cm 分析，只做证据保全和 Research Debt。该选项仍是 HF05 失败后
的处置，不消费 GPU 预算。

## Decision and current action

用户的“按照你的想法继续研究”被记录为建立 HF05 新 goal 的授权。选择 Option A，
先提交预声明 card、CPU evaluator 和测试，再运行唯一的离线判别 Probe。HF01–HF04
保持冻结；禁止 online、PPO、Isaac Gym collection 和任何 GPU 进程。
