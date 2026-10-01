# Ref2Dex Research Mission

## 1. Core idea

本项目希望研究：

> 是否可以学习一种以 Cm 为核心、动作条件的短期手物交互转移表示，并利用其物理预测帮助机器人强化学习策略完成灵巧手抓取。

短期转移可以是一步或短多步；物理预测与长期动作价值是不同概念。
物体运动幅度不能直接替代抓取动作价值，后者取决于任务目标和后续执行策略。

Cm 可以重新设计、重新训练或改变接入方式。

不要求沿用当前 CmResidual、CmLite 或历史 Cmv2 的具体实现。

最终评价的是研究思想是否成立，而不是是否保持某个旧版本实现。

---

## 2. Final objectives

项目最终需要同时解决两个问题。

### Objective A — Self-trained manipulation

在 DExplore 的物理环境中，不直接使用官方 DExplore actor checkpoint 作为最终策略基础，自行训练得到能够完成目标抓取/抬升任务的策略。

---

### Objective B — Demonstrate that Cm matters

仅仅训练出一个能够抓取的策略不足以完成本项目。

还必须回答：

> Cm 是否对最终策略产生了真实、可验证的增益？

最终需要一个 matched causal comparison：

Cm-on

vs.

Cm-off

并尽量保持其他训练条件一致。

可验证的收益包括：固定环境交互预算下持续抬升成功率提高，或达到预先固定
成功率所需的环境交互减少。预训练数据、额外计算和运行时间须单独报告，
不能把在线样本效率收益等同于总数据或计算成本降低。

冻结 actor 后加入候选动作选择器的收益可以作为机制 Probe，但当前交付还要求
将 Cm 用于策略训练，并独立评估训练所得策略；仅有控制器收益不能替代 RL 学习收益。
持续抬升评价须包含明确的保持时长及后续掉落检查，具体协议在实验前固定；
历史五步 held-lift 指标保留作比较，不能单独代表新的稳定抓取标准。

最终 Cm 可以通过不同方式参与，包括但不限于：

* representation；
* critic；
* action ranking；
* model-based planning；
* candidate scoring；
* auxiliary objective；
* reward；
* teacher/distillation；
* residual guidance。

当前不预先规定哪一种接入方式必须成功。

---

## 3. What does NOT count as success

以下结果本身不能完成最终目标：

1. 只有官方 DExplore actor 能抓取；
2. 只有 supervised/BC policy 能抓取，但 Cm 没有因果贡献；
3. Cm 离线预测误差较低，但不能改善实际策略；
4. 某一次 rollout 成功；
5. 某个 reward 或 Cm 指标上升，但抓取没有改善；
6. 只有 Cm-on 结果，没有 matched Cm-off；
7. Cm-off 同样好，但仅通过解释认为 Cm “可能有帮助”。

---

## 4. Research priorities

当前优先级：

### P0

建立可靠的 self-trained manipulation baseline。

### P1

寻找一种能够让 Cm 对实际 policy decision 产生信息增益的接入方式。

### P2

进行 matched Cm-on/off Validation，判断 Cm 是否具有因果价值。

### P3

在核心方法成立后，再扩大：

* 多轨迹；
* 多物体；
* 多 hand embodiment；
* 泛化；
* 更完整 statistical validation。

P3 不应阻塞 P1/P2。

Cm 可以利用完整交互轨迹和兼容的多物体、多数据集数据预训练，再用当前机器人
的真实仿真转移适配；广泛训练数据不等同于已证明泛化。跨数据集预训练是可选
研究方向，不是当前交付前置条件。首次新路线策略收益验证限定 airplane，使用
相同自训练 actor 初始化及 matched 条件。

---

## 5. North-star scoreboard

| 目标                      | 达成条件                      | 当前状态          |
| ----------------------- | ------------------------- | ------------- |
| Self-trained grasp      | 自训练策略在固定任务上稳定抓取           | `PARTIAL`：任务限定接受冻结 12-motion 任务上的自训练六专家初始观测路由；单一观测驱动 actor 仍未证 |
| Cm one-step information | Cm 对真实物理转移具有可用预测/排序信息     | 部分成立，但依赖表示和分布 |
| Cm policy utility       | matched Cm-on 明显优于 Cm-off | 尚未证明          |
| Generalization          | 多轨迹/多场景保持效果               | 未见物体迁移弱，Cm 收益未证 |

最终论文价值主要取决于第三项：

> Cm policy utility。

因此当 self-trained baseline 已足够用于实验时，不应继续无限优化无 Cm baseline，而应将主要研究预算转向 Cm 如何真正影响策略。
