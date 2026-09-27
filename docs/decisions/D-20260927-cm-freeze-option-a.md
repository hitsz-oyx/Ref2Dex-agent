# D-20260927 — Cm 冻结 Option A

状态：Decision Checkpoint 已完成；Cm campaign 继续冻结。

## 用户答复（原文）

`A`

用户原文仅为上述选项标识。按本次 Decision Checkpoint 中给出的选项定义，
Option A 对应：继续冻结 Cm，保留 C1 substrate，使用 0 GPU，并等待一个可预声明的
新机制。这一解释是选项映射，不是用户逐字原话。

## 决定

- 保留已接受且任务限定的 C1 substrate；不重跑 C1，不扩大其 claim。
- Cm campaign 维持 `FROZEN`；C1/C2/C3 scoreboard、全部 HF 状态和 budget 不变。
- 不登记 HF06，不恢复 HF01–HF05，不启动 logging patch、collection、fit、训练、
  collector、PPO、online Probe 或任何 GPU 工作。
- 未来只有出现区别于 HF01–HF05、能够事前声明的新高层机制，并建立独立的
  high-level goal、完成新的 Decision Checkpoint 后，才重新评估 Cm 路线。

## 阶段性 handoff

当前为 `UNCLEAR/NOT READY`。该标签只表示尚无获准继续执行的新机制，不是新的
实验结果或科学证据，也不升级任何 claim。

资源：`GPU=0; new_research_data=0; experiment=NONE`。

状态入口见 [`STATE.md`](../STATE.md)，机器搜索状态见
[`RESEARCH_QUEUE.yaml`](../RESEARCH_QUEUE.yaml)。
