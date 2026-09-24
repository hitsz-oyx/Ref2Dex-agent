# Ref2Dex Research Mission

## 1. Core idea

本项目希望研究：

> 是否可以学习一种以 Cm 为核心的局部交互/一步物体效应表示，并利用它帮助机器人强化学习策略完成灵巧手抓取。

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

---

## 5. North-star scoreboard

| 目标                      | 达成条件                      | 当前状态          |
| ----------------------- | ------------------------- | ------------- |
| Self-trained grasp      | 自训练策略在固定任务上稳定抓取           | 已取得强阶段证据      |
| Cm one-step information | Cm 对真实物理转移具有可用预测/排序信息     | 部分成立，但依赖表示和分布 |
| Cm policy utility       | matched Cm-on 明显优于 Cm-off | 尚未证明          |
| Generalization          | 多轨迹/多场景保持效果               | 后续目标          |

最终论文价值主要取决于第三项：

> Cm policy utility。

因此当 self-trained baseline 已足够用于实验时，不应继续无限优化无 Cm baseline，而应将主要研究预算转向 Cm 如何真正影响策略。
